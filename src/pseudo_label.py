"""
Self-training with confidence-thresholded pseudo-labels, plus optional
entropy minimization on all target pixels.

This is the highest-leverage intervention when you have unlabeled target
chips: it lets the model directly absorb the ACTUAL Sahel pixel statistics
and class distribution, instead of relying on indirect proxies.

Loop:
  1. Run current model (post AdaBN + radiometric harmonization ideally) on
     unlabeled target chips.
  2. Keep per-pixel argmax predictions where softmax confidence exceeds
     `confidence_threshold`; mask out (ignore_index) the rest.
  3. Fine-tune on source (real labels) + target (pseudo-labels) jointly,
     with class-balanced weighting and an entropy-minimization term on all
     target pixels (encourages confident, decisive predictions even where
     unmasked).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

IGNORE_INDEX = -1


@torch.no_grad()
def generate_pseudo_labels(model, unlabeled_loader, num_classes, device="cpu",
                            confidence_threshold=0.9):
    """Returns dict: id -> (image numpy, pseudo_label numpy with IGNORE_INDEX
    at low-confidence pixels), plus the fraction of pixels kept (a useful
    diagnostic: too low means threshold is too strict / model too uncertain)."""
    model.eval()
    pseudo = {}
    kept_total, pixel_total = 0, 0

    for images, ids in unlabeled_loader:
        images_dev = images.to(device)
        logits = model(images_dev)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        conf = conf.cpu().numpy()
        pred = pred.cpu().numpy()
        images_np = images.numpy()

        mask = conf >= confidence_threshold
        kept_total += mask.sum()
        pixel_total += mask.size

        for i, _id in enumerate(ids):
            lbl = pred[i].copy()
            lbl[~mask[i]] = IGNORE_INDEX
            pseudo[_id] = (images_np[i], lbl)

    kept_frac = kept_total / max(1, pixel_total)
    print(f"Pseudo-labeling: kept {kept_frac:.1%} of target pixels at "
          f"confidence >= {confidence_threshold}")
    return pseudo, kept_frac


class PseudoLabeledDataset(Dataset):
    def __init__(self, pseudo_dict):
        self.items = list(pseudo_dict.values())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image, label = self.items[idx]
        return torch.from_numpy(image), torch.from_numpy(label)


def class_balanced_weights(label_arrays, num_classes):
    """Inverse-frequency class weights computed over a list of label arrays,
    ignoring IGNORE_INDEX pixels. Mitigates class-prior shift."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for lbl in label_arrays:
        valid = lbl[lbl != IGNORE_INDEX]
        counts += np.bincount(valid, minlength=num_classes)
    freq = counts / max(1, counts.sum())
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def entropy_loss(logits):
    """Mean per-pixel entropy of the softmax distribution -- minimizing this
    sharpens predictions on unlabeled target pixels (entropy minimization,
    a standard UDA regularizer)."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    ent = -(probs * log_probs).sum(dim=1)
    return ent.mean()


def self_train(model, source_dataset, pseudo_dict, cfg, device="cpu",
               all_target_images_for_entropy=None):
    """One round of self-training fine-tuning.
    source_dataset: labeled SegmentationDataset (real labels).
    pseudo_dict: output of generate_pseudo_labels().
    all_target_images_for_entropy: optional (N, C, H, W) tensor of ALL
        unlabeled target images (not just confident ones) for the entropy
        minimization term.
    """
    model.to(device)
    model.train()

    pseudo_ds = PseudoLabeledDataset(pseudo_dict)
    combined = ConcatDataset([source_dataset, pseudo_ds])
    loader = DataLoader(combined, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers)

    all_labels = [source_dataset[i][1].numpy() for i in range(len(source_dataset))]
    all_labels += [lbl for _, lbl in pseudo_dict.values()]
    class_weights = class_balanced_weights(all_labels, cfg.num_classes).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr * 0.3)  # gentle LR

    for epoch in range(cfg.finetune_epochs):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = ce_loss(logits, labels)

            if all_target_images_for_entropy is not None and cfg.entropy_loss_weight > 0:
                idx = torch.randint(0, all_target_images_for_entropy.shape[0],
                                     (min(cfg.batch_size, all_target_images_for_entropy.shape[0]),))
                ent_batch = all_target_images_for_entropy[idx].to(device)
                ent_logits = model(ent_batch)
                loss = loss + cfg.entropy_loss_weight * entropy_loss(ent_logits)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[self-train] epoch {epoch + 1}/{cfg.finetune_epochs} "
              f"loss={running_loss / max(1, len(loader)):.4f}")

    model.eval()
    return model
