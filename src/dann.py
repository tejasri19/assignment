"""
DANN-style domain-adversarial feature alignment (fallback intervention).

Use this only if diagnostics show the domain classifier can still trivially
separate source/target embeddings AFTER AdaBN + radiometric harmonization +
self-training. Adds a small domain discriminator on the encoder bottleneck
features and trains the encoder, via a gradient-reversal layer, to make
those features domain-indistinguishable while the segmentation head keeps
learning from source labels (and optionally target pseudo-labels).

Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by
Backpropagation" (DANN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """Small MLP head on the pooled bottleneck feature -> P(domain = target)."""

    def __init__(self, feature_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)  # logits, shape (N,)


def train_dann_epoch(model, discriminator, source_loader, target_loader,
                      optimizer, cfg, device="cpu", lambd=1.0):
    """One epoch of joint segmentation + domain-adversarial training.
    source_loader yields (image, label); target_loader yields (image, id)."""
    model.train()
    discriminator.train()

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    bce_loss = nn.BCEWithLogitsLoss()

    target_iter = iter(target_loader)
    running_seg, running_dom = 0.0, 0.0
    n_batches = 0

    for src_images, src_labels in source_loader:
        try:
            tgt_images, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_images, _ = next(target_iter)

        src_images = src_images.to(device)
        src_labels = src_labels.to(device)
        tgt_images = tgt_images.to(device)

        optimizer.zero_grad()

        # Segmentation loss on source only (we have no target labels).
        seg_logits, src_feat = model(src_images, return_features=True)
        seg_loss = ce_loss(seg_logits, src_labels)

        # Domain-adversarial loss on both domains' pooled features.
        _, tgt_feat = model(tgt_images, return_features=True)
        src_feat_rev = grad_reverse(src_feat, lambd)
        tgt_feat_rev = grad_reverse(tgt_feat, lambd)

        domain_logits = torch.cat([
            discriminator(src_feat_rev),
            discriminator(tgt_feat_rev),
        ], dim=0)
        domain_targets = torch.cat([
            torch.zeros(src_feat.size(0), device=device),  # 0 = source
            torch.ones(tgt_feat.size(0), device=device),   # 1 = target
        ], dim=0)
        domain_loss = bce_loss(domain_logits, domain_targets)

        loss = seg_loss + cfg.dann_loss_weight * domain_loss
        loss.backward()
        optimizer.step()

        running_seg += seg_loss.item()
        running_dom += domain_loss.item()
        n_batches += 1

    return running_seg / max(1, n_batches), running_dom / max(1, n_batches)


def train_dann(model, source_dataset, target_dataset, cfg, device="cpu", epochs=None):
    from torch.utils.data import DataLoader
    discriminator = DomainDiscriminator(model.feature_dim).to(device)
    model.to(device)

    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=cfg.batch_size, shuffle=True)

    params = list(model.parameters()) + list(discriminator.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr * 0.3)

    epochs = epochs or cfg.finetune_epochs
    for epoch in range(epochs):
        # Ramp up lambda over training, standard DANN schedule.
        p = epoch / max(1, epochs - 1)
        lambd = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0

        seg_l, dom_l = train_dann_epoch(model, discriminator, source_loader,
                                         target_loader, optimizer, cfg,
                                         device=device, lambd=lambd)
        print(f"[DANN] epoch {epoch + 1}/{epochs} seg_loss={seg_l:.4f} "
              f"domain_loss={dom_l:.4f} lambda={lambd:.2f}")

    model.eval()
    return model, discriminator
