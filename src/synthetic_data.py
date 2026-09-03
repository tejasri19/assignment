"""
Synthetic stand-in for real satellite chips.

Until real Iowa (temperate North America) and Sahel (semi-arid) chips are
available, this module generates numpy arrays that reproduce the SAME KINDS
of domain shift described in the problem statement, so that every diagnostic
and every intervention downstream is exercised meaningfully:

  1. Class-prior shift:
       - source ("Iowa")  -> dominated by regular rectangular cropland fields
                              plus some water/built-up, little bare soil.
       - target ("Sahel") -> dominated by irregular bare-soil/shrub patches,
                              little cropland/water.
  2. Radiometric / seasonal shift:
       - target chips get a brightness/contrast/gamma shift and a different
         per-band offset to emulate a different acquisition season on the
         same sensor (sun angle, atmospheric correction, phenology).
  3. Textural shift:
       - source field boundaries are axis-aligned rectangles (like center-
         pivot / row-crop cadastral patterns); target regions are organic
         Voronoi-like blobs (more irregular smallholder plots / natural
         vegetation patches).

Once you have real chips, replace calls to `generate_and_save_all()` with
your own ingestion script that writes into the same folder layout:

    data/<domain>/images/<id>.npy   float32 array, shape (C, H, W)
    data/<domain>/labels/<id>.npy   int64 array,  shape (H, W)   (omit for
                                     unlabeled target chips)
"""

import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
from config import cfg  # noqa: E402

WATER, BUILT_UP, CROPLAND, GRASS_SHRUB, BARE_SOIL = range(5)


def _voronoi_label_map(size, n_seeds, rng):
    """Organic irregular regions via nearest-seed assignment."""
    ys, xs = np.mgrid[0:size, 0:size]
    seeds = rng.integers(0, size, size=(n_seeds, 2))
    dist = np.zeros((n_seeds, size, size), dtype=np.float32)
    for i, (sy, sx) in enumerate(seeds):
        dist[i] = (ys - sy) ** 2 + (xs - sx) ** 2
    return np.argmin(dist, axis=0)  # (H, W) region id per pixel


def _rectangular_label_map(size, n_fields, rng):
    """Axis-aligned rectangular fields, like cadastral cropland patterns."""
    region = np.zeros((size, size), dtype=np.int64)
    region_id = 1
    for _ in range(n_fields):
        w = rng.integers(size // 8, size // 3)
        h = rng.integers(size // 8, size // 3)
        x0 = rng.integers(0, max(1, size - w))
        y0 = rng.integers(0, max(1, size - h))
        region[y0:y0 + h, x0:x0 + w] = region_id
        region_id += 1
    return region


def generate_source_chip(size, rng):
    """Iowa-like chip: regular fields, cropland-dominant, one season."""
    region = _rectangular_label_map(size, n_fields=rng.integers(6, 12), rng=rng)
    label = np.full((size, size), CROPLAND, dtype=np.int64)

    n_regions = region.max() + 1
    for r in range(1, n_regions):
        mask = region == r
        roll = rng.random()
        if roll < 0.08:
            label[mask] = WATER
        elif roll < 0.15:
            label[mask] = BUILT_UP
        elif roll < 0.20:
            label[mask] = GRASS_SHRUB
        else:
            label[mask] = CROPLAND  # dominant class

    image = _render_reflectance(label, domain="source", rng=rng, size=size)
    return image, label


def generate_target_chip(size, rng, labeled=False):
    """Sahel-like chip: irregular patches, bare-soil/shrub dominant,
    different season -> different radiometry, same sensor bands."""
    region = _voronoi_label_map(size, n_seeds=rng.integers(10, 18), rng=rng)
    label = np.full((size, size), BARE_SOIL, dtype=np.int64)

    n_regions = region.max() + 1
    for r in range(n_regions):
        mask = region == r
        roll = rng.random()
        if roll < 0.03:
            label[mask] = WATER
        elif roll < 0.08:
            label[mask] = BUILT_UP
        elif roll < 0.15:
            label[mask] = CROPLAND       # occasional small irrigated plot
        elif roll < 0.55:
            label[mask] = GRASS_SHRUB
        else:
            label[mask] = BARE_SOIL      # dominant class

    image = _render_reflectance(label, domain="target", rng=rng, size=size)
    return image, (label if labeled else None)


# Per-class base reflectance signature, roughly (R, G, B, NIR) in [0, 1].
# Vegetation classes get high NIR / low R (classic vegetation contrast);
# bare soil / built-up get flatter, brighter spectra.
_BASE_SPECTRA = {
    WATER:        np.array([0.05, 0.07, 0.10, 0.03]),
    BUILT_UP:     np.array([0.35, 0.33, 0.30, 0.32]),
    CROPLAND:     np.array([0.12, 0.22, 0.10, 0.55]),
    GRASS_SHRUB:  np.array([0.20, 0.25, 0.15, 0.35]),
    BARE_SOIL:    np.array([0.32, 0.28, 0.24, 0.30]),
}


def _render_reflectance(label, domain, rng, size):
    c = cfg.in_channels
    image = np.zeros((c, size, size), dtype=np.float32)
    for cls, spectrum in _BASE_SPECTRA.items():
        mask = label == cls
        if not mask.any():
            continue
        spec = spectrum[:c] if c <= 4 else np.pad(spectrum, (0, c - 4), constant_values=0.2)
        noise = rng.normal(0, 0.02, size=(mask.sum(), c)).astype(np.float32)
        image[:, mask] = (spec[None, :] + noise).T

    if domain == "target":
        # Simulate a different acquisition season on the SAME sensor:
        # brightness/contrast/gamma shift + per-band offset (atmospheric /
        # illumination difference), applied uniformly to the whole chip.
        gamma = rng.uniform(1.15, 1.45)
        brightness = rng.uniform(0.08, 0.18)
        contrast = rng.uniform(0.75, 0.9)
        band_offset = rng.normal(0.03, 0.015, size=(c, 1, 1)).astype(np.float32)

        image = np.clip(image, 0, 1) ** gamma
        image = image * contrast + brightness
        image = image + band_offset

    image = np.clip(image, 0, 1).astype(np.float32)
    return image


def generate_and_save_all(force=False):
    """Populate data/source_iowa and data/target_sahel with synthetic chips
    following the exact folder layout expected by dataset.py."""
    rng = np.random.default_rng(cfg.synthetic_seed)

    src_img_dir = os.path.join(cfg.data_root, cfg.source_name, "images")
    src_lbl_dir = os.path.join(cfg.data_root, cfg.source_name, "labels")
    tgt_img_dir = os.path.join(cfg.data_root, cfg.target_name, "images")
    tgt_lbl_dir = os.path.join(cfg.data_root, cfg.target_name, "labels")
    for d in (src_img_dir, src_lbl_dir, tgt_img_dir, tgt_lbl_dir):
        os.makedirs(d, exist_ok=True)

    if not force and len(os.listdir(src_img_dir)) > 0:
        print("Synthetic data already present, skipping generation "
              "(pass force=True to regenerate).")
        return

    n_source = cfg.n_source_train + cfg.n_source_val
    for i in range(n_source):
        img, lbl = generate_source_chip(cfg.chip_size, rng)
        np.save(os.path.join(src_img_dir, f"src_{i:04d}.npy"), img)
        np.save(os.path.join(src_lbl_dir, f"src_{i:04d}.npy"), lbl)

    for i in range(cfg.n_target_unlabeled):
        img, _ = generate_target_chip(cfg.chip_size, rng, labeled=False)
        np.save(os.path.join(tgt_img_dir, f"tgt_{i:04d}.npy"), img)

    # Small held-out labeled target set -- used ONLY for honest evaluation,
    # never for training, mirroring a real "spend a little labeling budget
    # purely to measure progress" strategy.
    for i in range(cfg.n_target_val_labeled):
        j = cfg.n_target_unlabeled + i
        img, lbl = generate_target_chip(cfg.chip_size, rng, labeled=True)
        np.save(os.path.join(tgt_img_dir, f"tgt_{j:04d}.npy"), img)
        np.save(os.path.join(tgt_lbl_dir, f"tgt_{j:04d}.npy"), lbl)

    print(f"Wrote {n_source} source chips, {cfg.n_target_unlabeled} unlabeled "
          f"target chips, {cfg.n_target_val_labeled} labeled target "
          f"validation chips under {cfg.data_root}")


if __name__ == "__main__":
    generate_and_save_all()
