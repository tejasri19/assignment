"""
AdaBN: recalibrate BatchNorm running statistics using unlabeled target chips.

This is the cheapest possible intervention -- no gradients, no labels, just
forward passes -- and isolates how much of the domain gap is caused by the
encoder's BatchNorm layers carrying Iowa-specific activation statistics.

Reference: Li et al., "Revisiting Batch Normalization for Practical Domain
Adaptation" (AdaBN).
"""

import torch
import torch.nn as nn


@torch.no_grad()
def recalibrate_batchnorm(model, target_loader, num_batches=30, device="cpu"):
    """Resets BN running stats then re-estimates them purely from target data.
    All non-BN parameters stay frozen (no weight updates occur at all)."""
    model.to(device)

    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bn_layers:
        print("No BatchNorm2d layers found in model -- AdaBN is a no-op here.")
        return model

    for bn in bn_layers:
        bn.reset_running_stats()
        bn.momentum = None  # use cumulative moving average over all batches seen

    model.train()  # BN layers update running stats only in train() mode
    for p in model.parameters():
        p.requires_grad_(False)  # safety: guarantee no weight updates happen

    seen = 0
    for images, _ids in target_loader:
        images = images.to(device)
        model(images)
        seen += 1
        if seen >= num_batches:
            break

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    print(f"AdaBN: recalibrated {len(bn_layers)} BatchNorm layers using "
          f"{seen} unlabeled target batches.")
    return model
