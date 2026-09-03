"""
Radiometric harmonization: fixes the "same sensor, different season" part of
the domain gap by rescaling target chip statistics onto source statistics,
with two interchangeable methods:

  1. linear_match  - per-band mean/std rescale (fast, robust, good default).
  2. histogram_match - full per-band CDF matching (numpy-only implementation,
     no scikit-image dependency), captures non-linear shifts (gamma, etc.)
     better than linear rescaling.

Apply this to target images at both inference time and before self-training.
"""

import numpy as np


def compute_band_stats(images):
    """images: (N, C, H, W) -> (mean[C], std[C])"""
    mean = images.mean(axis=(0, 2, 3))
    std = images.std(axis=(0, 2, 3))
    return mean, std


def linear_match(images, source_mean, source_std, own_mean=None, own_std=None):
    """Rescale `images` so each band's mean/std matches source_mean/source_std.
    If own_mean/own_std not given, they are computed from `images` itself."""
    if own_mean is None or own_std is None:
        own_mean, own_std = compute_band_stats(images)
    own_mean = own_mean.reshape(1, -1, 1, 1)
    own_std = own_std.reshape(1, -1, 1, 1) + 1e-8
    source_mean = source_mean.reshape(1, -1, 1, 1)
    source_std = source_std.reshape(1, -1, 1, 1)

    normalized = (images - own_mean) / own_std
    matched = normalized * source_std + source_mean
    return np.clip(matched, 0, 1).astype(np.float32)


def _single_band_histogram_match(source_band_pixels, target_band):
    """Map target_band's pixel values onto source distribution via CDF matching.
    source_band_pixels: 1D reference sample. target_band: array of any shape."""
    src_sorted = np.sort(source_band_pixels)
    src_quantiles = np.linspace(0, 1, len(src_sorted))

    flat = target_band.ravel()
    tgt_sorted_idx = np.argsort(flat)
    tgt_ranks = np.empty_like(tgt_sorted_idx)
    tgt_ranks[tgt_sorted_idx] = np.arange(len(flat))
    tgt_quantiles = tgt_ranks / max(1, len(flat) - 1)

    matched_flat = np.interp(tgt_quantiles, src_quantiles, src_sorted)
    return matched_flat.reshape(target_band.shape)


def histogram_match(images, source_reference_images):
    """images, source_reference_images: (N, C, H, W). Matches each band of
    `images` to the pooled per-band distribution of source_reference_images."""
    C = images.shape[1]
    out = np.empty_like(images)
    for c in range(C):
        ref_pixels = source_reference_images[:, c].ravel()
        # Subsample reference for speed if very large.
        if ref_pixels.size > 200_000:
            rng = np.random.default_rng(0)
            ref_pixels = rng.choice(ref_pixels, size=200_000, replace=False)
        for n in range(images.shape[0]):
            out[n, c] = _single_band_histogram_match(ref_pixels, images[n, c])
    return np.clip(out, 0, 1).astype(np.float32)
