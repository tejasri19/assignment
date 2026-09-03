"""
PyTorch Dataset classes for the source (labeled) and target (unlabeled /
partially labeled) domains. Works transparently for BOTH the synthetic
stand-in data and real chips, as long as real chips are exported into the
same folder layout:

    data/<domain>/images/<id>.npy   float32, shape (C, H, W), values in [0, 1]
    data/<domain>/labels/<id>.npy   int64,   shape (H, W)      (if labeled)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_ids(images_dir):
    paths = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


class SegmentationDataset(Dataset):
    """Labeled dataset: returns (image, label) tensors."""

    def __init__(self, domain_dir, ids=None, transform=None):
        self.images_dir = os.path.join(domain_dir, "images")
        self.labels_dir = os.path.join(domain_dir, "labels")
        self.ids = ids if ids is not None else _list_ids(self.images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        image = np.load(os.path.join(self.images_dir, f"{_id}.npy")).astype(np.float32)
        label = np.load(os.path.join(self.labels_dir, f"{_id}.npy")).astype(np.int64)
        if self.transform is not None:
            image, label = self.transform(image, label)
        return torch.from_numpy(image), torch.from_numpy(label)


class UnlabeledDataset(Dataset):
    """Unlabeled dataset: returns image tensor only (+ id, useful for
    caching pseudo-labels back to disk by id)."""

    def __init__(self, domain_dir, ids=None, transform=None):
        self.images_dir = os.path.join(domain_dir, "images")
        self.ids = ids if ids is not None else _list_ids(self.images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        image = np.load(os.path.join(self.images_dir, f"{_id}.npy")).astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)
        return torch.from_numpy(image), _id


def split_ids(images_dir, n_val, seed=0):
    """Deterministic train/val split by id, used for the source domain."""
    ids = _list_ids(images_dir)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    ids = [ids[i] for i in perm]
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids
