"""
Evaluation utilities: run a model over a labeled dataset and report mIoU +
per-class IoU, formatted as a readable table. Used to produce honest
before/after comparisons for every intervention.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnostics import per_class_iou, mean_iou, confusion_matrix


@torch.no_grad()
def evaluate(model, dataset, num_classes, class_names, device="cpu", batch_size=8):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    ious = per_class_iou(preds, labels, num_classes)
    miou = mean_iou(ious)
    cm = confusion_matrix(preds, labels, num_classes)

    return {
        "miou": miou,
        "per_class_iou": {name: (float(iou) if not np.isnan(iou) else None)
                           for name, iou in zip(class_names, ious)},
        "confusion_matrix": cm.tolist(),
    }


def print_report(title, report, class_names):
    print(f"\n=== {title} ===")
    print(f"mIoU: {report['miou']:.4f}")
    for name in class_names:
        val = report["per_class_iou"][name]
        val_str = f"{val:.4f}" if val is not None else "  n/a "
        print(f"  {name:>18s}: {val_str}")


def print_comparison(reports_by_stage, class_names):
    """reports_by_stage: OrderedDict-like {stage_name: report_dict}."""
    stages = list(reports_by_stage.keys())
    print("\n=== mIoU across pipeline stages ===")
    header = f"{'class':>18s} | " + " | ".join(f"{s:>14s}" for s in stages)
    print(header)
    print("-" * len(header))
    for name in class_names:
        row = f"{name:>18s} | "
        cells = []
        for s in stages:
            v = reports_by_stage[s]["per_class_iou"][name]
            cells.append(f"{v:14.4f}" if v is not None else f"{'n/a':>14s}")
        print(row + " | ".join(cells))
    print("-" * len(header))
    miou_row = f"{'mIoU':>18s} | " + " | ".join(
        f"{reports_by_stage[s]['miou']:14.4f}" for s in stages)
    print(miou_row)
