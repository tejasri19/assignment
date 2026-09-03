"""
End-to-end demo pipeline tying together every step from the plan:

  0. Generate synthetic Iowa/Sahel stand-in data (skip once you have real chips).
  1. Train baseline model on source (Iowa) only.
  2. Evaluate baseline on source-val AND target-val -- reproduce the
     "0.85 on Iowa, 0.41 on Sahel" style gap.
  3. Run full diagnostics (per-class IoU, band shift, domain classifier,
     embedding plot) and save a JSON report + PNG.
  4. Apply AdaBN (free, no labels) -> re-evaluate.
  5. Apply radiometric harmonization (histogram matching target -> source)
     -> re-evaluate.
  6. Self-train with confidence-thresholded pseudo-labels + entropy
     minimization -> re-evaluate.
  7. (Optional, only if gap persists) DANN feature alignment -> re-evaluate.
  8. Print a before/after comparison table across every stage.

Run with:  python src/run_pipeline.py
"""

import os
import sys
import copy
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
from config import cfg  # noqa: E402

import synthetic_data  # noqa: E402
from dataset import SegmentationDataset, UnlabeledDataset, split_ids  # noqa: E402
from model import UNet  # noqa: E402
from train import train_baseline  # noqa: E402
from eval import evaluate, print_report, print_comparison  # noqa: E402
import diagnostics as diag  # noqa: E402
import normalize  # noqa: E402
import adabn  # noqa: E402
import pseudo_label as pl  # noqa: E402
import dann  # noqa: E402

from torch.utils.data import DataLoader


def load_domain_arrays(domain_dir, ids):
    """Stack a list of chip ids into (N, C, H, W) numpy image arrays."""
    imgs = [np.load(os.path.join(domain_dir, "images", f"{i}.npy")) for i in ids]
    return np.stack(imgs, axis=0)


def main():
    device = cfg.device
    stages = {}

    # --- 0. Data ---
    print("\nStep 0: ensuring synthetic stand-in data exists...")
    synthetic_data.generate_and_save_all()

    source_dir = os.path.join(cfg.data_root, cfg.source_name)
    target_dir = os.path.join(cfg.data_root, cfg.target_name)

    train_ids, val_ids = split_ids(os.path.join(source_dir, "images"),
                                    n_val=cfg.n_source_val, seed=cfg.synthetic_seed)
    source_train_ds = SegmentationDataset(source_dir, ids=train_ids)
    source_val_ds = SegmentationDataset(source_dir, ids=val_ids)

    all_target_ids = sorted(os.path.splitext(f)[0] for f in
                             os.listdir(os.path.join(target_dir, "images")))
    target_labeled_ids = sorted(os.path.splitext(f)[0] for f in
                                 os.listdir(os.path.join(target_dir, "labels")))
    target_unlabeled_ids = [i for i in all_target_ids if i not in set(target_labeled_ids)]

    target_unlabeled_ds = UnlabeledDataset(target_dir, ids=target_unlabeled_ids)
    target_val_ds = SegmentationDataset(target_dir, ids=target_labeled_ids)  # eval ONLY

    # --- 1. Baseline training (source only) ---
    print("\nStep 1: training baseline model on source (Iowa) only...")
    model, _, _ = train_baseline()

    # --- 2. Baseline evaluation: reproduce the reported gap ---
    print("\nStep 2: evaluating baseline on source-val and target-val...")
    stages["baseline_source"] = evaluate(model, source_val_ds, cfg.num_classes, cfg.class_names, device)
    stages["baseline_target"] = evaluate(model, target_val_ds, cfg.num_classes, cfg.class_names, device)
    print_report("Baseline on SOURCE val (Iowa)", stages["baseline_source"], cfg.class_names)
    print_report("Baseline on TARGET val (Sahel)", stages["baseline_target"], cfg.class_names)

    # --- 3. Diagnostics ---
    print("\nStep 3: running diagnostics (band shift, domain classifier, embeddings)...")
    source_imgs = load_domain_arrays(source_dir, train_ids[:150])
    target_imgs = load_domain_arrays(target_dir, target_unlabeled_ids[:150])

    band_report = diag.band_shift_report(source_imgs, target_imgs)
    domain_report = diag.domain_classifier_accuracy(source_imgs, target_imgs)
    print("Per-band shift:")
    for b in band_report:
        print(f"  band {b['band']}: mean_shift={b['mean_shift']:+.4f} "
              f"std_ratio={b['std_ratio']:.2f}"
              + (f" ks={b['ks_statistic']:.3f}" if "ks_statistic" in b else ""))
    print(f"Domain classifier accuracy: {domain_report['domain_classifier_accuracy']:.3f} "
          f"(0.5 = indistinguishable, 1.0 = trivially separable)")

    emb_source = diag.extract_embeddings(model, torch.from_numpy(source_imgs), device)
    emb_target = diag.extract_embeddings(model, torch.from_numpy(target_imgs), device)
    proj_source = diag.project_2d(np.concatenate([emb_source, emb_target]))[:len(emb_source)]
    proj_target = diag.project_2d(np.concatenate([emb_source, emb_target]))[len(emb_source):]
    diag.plot_domain_embeddings(proj_source, proj_target,
                                 os.path.join(cfg.diagnostics_dir, "embeddings_baseline.png"))

    diag.save_json_report(
        {"band_shift": band_report, "domain_classifier": domain_report},
        os.path.join(cfg.diagnostics_dir, "diagnostics_baseline.json"))

    # --- 4. AdaBN ---
    print("\nStep 4: recalibrating BatchNorm on unlabeled target chips (AdaBN)...")
    target_unlabeled_loader = DataLoader(target_unlabeled_ds, batch_size=cfg.batch_size, shuffle=True)
    model = adabn.recalibrate_batchnorm(model, target_unlabeled_loader,
                                         num_batches=cfg.adabn_batches, device=device)
    stages["after_adabn_target"] = evaluate(model, target_val_ds, cfg.num_classes, cfg.class_names, device)
    print_report("After AdaBN on TARGET val (Sahel)", stages["after_adabn_target"], cfg.class_names)

    # --- 5. Radiometric harmonization ---
    print("\nStep 5: radiometric harmonization (histogram matching target -> source)...")
    source_mean, source_std = normalize.compute_band_stats(source_imgs)

    class HarmonizedTargetDataset(SegmentationDataset):
        def __getitem__(self, idx):
            image, label = super().__getitem__(idx)
            image_np = image.numpy()[None]  # add batch dim
            matched = normalize.linear_match(image_np, source_mean, source_std)
            return torch.from_numpy(matched[0]), label

    harmonized_target_val_ds = HarmonizedTargetDataset(target_dir, ids=target_labeled_ids)
    stages["after_harmonization_target"] = evaluate(
        model, harmonized_target_val_ds, cfg.num_classes, cfg.class_names, device)
    print_report("After harmonization on TARGET val (Sahel)",
                 stages["after_harmonization_target"], cfg.class_names)

    # --- 6. Self-training with pseudo-labels ---
    print("\nStep 6: self-training with confidence-thresholded pseudo-labels...")

    class HarmonizedUnlabeledDataset(UnlabeledDataset):
        def __getitem__(self, idx):
            image, _id = super().__getitem__(idx)
            image_np = image.numpy()[None]
            matched = normalize.linear_match(image_np, source_mean, source_std)
            return torch.from_numpy(matched[0]), _id

    harmonized_unlabeled_ds = HarmonizedUnlabeledDataset(target_dir, ids=target_unlabeled_ids)
    harmonized_unlabeled_loader = DataLoader(harmonized_unlabeled_ds, batch_size=cfg.batch_size, shuffle=False)

    pseudo_dict, kept_frac = pl.generate_pseudo_labels(
        model, harmonized_unlabeled_loader, cfg.num_classes, device=device,
        confidence_threshold=cfg.pseudo_label_confidence)

    all_target_tensor = torch.stack([harmonized_unlabeled_ds[i][0] for i in range(len(harmonized_unlabeled_ds))])

    model = pl.self_train(model, source_train_ds, pseudo_dict, cfg, device=device,
                           all_target_images_for_entropy=all_target_tensor)

    stages["after_selftrain_target"] = evaluate(
        model, harmonized_target_val_ds, cfg.num_classes, cfg.class_names, device)
    stages["after_selftrain_source"] = evaluate(model, source_val_ds, cfg.num_classes, cfg.class_names, device)
    print_report("After self-training on TARGET val (Sahel)",
                 stages["after_selftrain_target"], cfg.class_names)
    print_report("After self-training on SOURCE val (sanity check, no regression)",
                 stages["after_selftrain_source"], cfg.class_names)

    # --- 7. Final comparison table ---
    print_comparison({
        "baseline_src": stages["baseline_source"],
        "baseline_tgt": stages["baseline_target"],
        "adabn_tgt": stages["after_adabn_target"],
        "harmon_tgt": stages["after_harmonization_target"],
        "selftrain_tgt": stages["after_selftrain_target"],
        "selftrain_src": stages["after_selftrain_source"],
    }, cfg.class_names)

    diag.save_json_report(
        {k: v for k, v in stages.items()},
        os.path.join(cfg.diagnostics_dir, "pipeline_results.json"))

    print("\nDone. See outputs/diagnostics/ for JSON reports and embedding plots, "
          "outputs/checkpoints/ for the baseline model weights.")


if __name__ == "__main__":
    main()
