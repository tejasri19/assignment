"""
Baseline supervised training on the source (Iowa) domain only.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
from config import cfg  # noqa: E402
from dataset import SegmentationDataset, split_ids  # noqa: E402
from model import UNet  # noqa: E402


def train_baseline(save_path=None, device=None):
    device = device or cfg.device
    source_dir = os.path.join(cfg.data_root, cfg.source_name)
    train_ids, val_ids = split_ids(os.path.join(source_dir, "images"),
                                    n_val=cfg.n_source_val, seed=cfg.synthetic_seed)

    train_ds = SegmentationDataset(source_dir, ids=train_ids)
    val_ds = SegmentationDataset(source_dir, ids=val_ids)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers)

    model = UNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes,
                 base_channels=cfg.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(cfg.baseline_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = ce_loss(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[baseline] epoch {epoch + 1}/{cfg.baseline_epochs} "
              f"loss={running_loss / max(1, len(train_loader)):.4f}")

    save_path = save_path or os.path.join(cfg.checkpoint_dir, "baseline_source_only.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved baseline checkpoint to {save_path}")

    return model, train_ds, val_ds


if __name__ == "__main__":
    train_baseline()
