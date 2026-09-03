"""
Central configuration for the Iowa -> Sahel domain-shift project.

No YAML/JSON dependency is used on purpose so this file works with a bare
numpy+torch+torchvision environment. Edit values directly.
"""

from dataclasses import dataclass, field
from typing import List
import os

# Repo root = parent of this file's parent (pix_code/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # --- data ---
    data_root: str = os.path.join(ROOT, "data")
    source_name: str = "source_iowa"
    target_name: str = "target_sahel"
    chip_size: int = 128
    in_channels: int = 4          # e.g. R, G, B, NIR (same sensor both domains)
    class_names: List[str] = field(default_factory=lambda: [
        "water", "built_up", "cropland", "grassland_shrub", "bare_soil"
    ])

    # --- synthetic data generation (only used until you plug in real chips) ---
    n_source_train: int = 300
    n_source_val: int = 60
    n_target_unlabeled: int = 300
    n_target_val_labeled: int = 40   # small held-out target set, labels used ONLY for evaluation
    synthetic_seed: int = 0

    # --- model ---
    base_channels: int = 32

    # --- training ---
    batch_size: int = 8
    lr: float = 1e-3
    baseline_epochs: int = 15
    finetune_epochs: int = 8
    num_workers: int = 0
    device: str = "cpu"   # set to "cuda" if available

    # --- domain adaptation knobs ---
    pseudo_label_confidence: float = 0.9
    entropy_loss_weight: float = 0.05
    dann_loss_weight: float = 0.1
    adabn_batches: int = 30

    # --- outputs ---
    output_root: str = os.path.join(ROOT, "outputs")
    checkpoint_dir: str = os.path.join(ROOT, "outputs", "checkpoints")
    diagnostics_dir: str = os.path.join(ROOT, "outputs", "diagnostics")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


cfg = Config()
