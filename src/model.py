"""
Compact U-Net for chip-level land-cover segmentation.

Exposes the bottleneck features via `forward(x, return_features=True)` so
diagnostics.py (embedding visualization, domain classifier) and dann.py
(domain-adversarial head) can hook into the same encoder representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=5, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.bottleneck = conv_block(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv2d(c, num_classes, 1)

        self.feature_dim = c * 8

    def forward(self, x, return_features=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.out_conv(d1)

        if return_features:
            # Global-average-pooled bottleneck feature: a compact per-chip
            # embedding used for t-SNE plots and the domain classifier.
            feat = F.adaptive_avg_pool2d(b, 1).flatten(1)
            return logits, feat
        return logits
