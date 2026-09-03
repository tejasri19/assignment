# pix_code — Iowa → Sahel domain-shift diagnosis & fix

A land-cover segmentation model trained on temperate North America (Iowa)
chips scores 0.85 mIoU there and drops to 0.41 mIoU on semi-arid (Sahel)
chips from a different acquisition season, same sensor. This project:

1. **Diagnoses** the failure with concrete, falsifiable evidence (per-class
   IoU, per-band radiometric shift, domain classifier separability, encoder
   embedding visualization).
2. **Intervenes** with a cheapest-first stack: AdaBN → radiometric
   harmonization → self-training with pseudo-labels → (fallback) DANN
   feature alignment.
3. **Re-evaluates** honestly after each step so you can see exactly how much
   of the gap each intervention closes.

Until real chips are available, `src/synthetic_data.py` generates a
stand-in dataset that reproduces the same three shift mechanisms (class
prior shift, radiometric/seasonal shift, textural shift) so every script
below is fully runnable and the numbers it prints are meaningful evidence of
the *mechanism*, not just a demo.

## Setup

Create a virtual environment and install required packages:
```pip install -r requirements.txt
```

## Run the full pipeline

```powershell
python src\run_pipeline.py
```

This single command:
- generates synthetic source/target chips under `data/` (skipped if already present),
- trains the baseline model on source only and saves it to `outputs/checkpoints/`,
- reproduces the "high source mIoU, low target mIoU" gap,
- runs the diagnostic suite and saves JSON + a PNG embedding plot to `outputs/diagnostics/`,
- applies AdaBN, then radiometric harmonization, then pseudo-label self-training,
- re-evaluates target mIoU after each step,
- prints a before/after comparison table across every stage.

## File-by-file map

| File | Role |
|---|---|
| `configs/config.py` | All hyperparameters/paths in one dataclass, no YAML dependency. |
| `src/synthetic_data.py` | Generates the synthetic Iowa/Sahel stand-in chips. **Replace this with a real ingestion script once you have real data** (same `.npy` folder layout). |
| `src/dataset.py` | PyTorch `Dataset` classes for labeled/unlabeled chips. |
| `src/model.py` | Compact U-Net; exposes bottleneck features for diagnostics/DANN. |
| `src/train.py` | Baseline supervised training on source only. |
| `src/eval.py` | mIoU / per-class IoU / confusion matrix, comparison tables. |
| `src/diagnostics.py` | Per-class IoU, per-band KS-test/mean-std shift, domain-classifier separability ("proxy A-distance"), t-SNE/PCA embedding plots. |
| `src/normalize.py` | Radiometric harmonization: linear stat matching + full histogram (CDF) matching. |
| `src/adabn.py` | BatchNorm statistics recalibration on unlabeled target chips. |
| `src/pseudo_label.py` | Confidence-thresholded pseudo-labeling + class-balanced self-training + entropy minimization. |
| `src/dann.py` | Gradient-reversal-layer domain-adversarial training (fallback intervention). |
| `src/run_pipeline.py` | Orchestrates everything above end to end. |

## Swapping in real chips

Once you have real Iowa and Sahel chips, export each chip as a pair of
`.npy` files matching this layout (values normalized to `[0, 1]`, band order
consistent across both domains since it's the same sensor):

```
data/source_iowa/images/<id>.npy   float32, shape (C, H, W)
data/source_iowa/labels/<id>.npy   int64,   shape (H, W)
data/target_sahel/images/<id>.npy  float32, shape (C, H, W)
data/target_sahel/labels/<id>.npy  int64,   shape (H, W)   
                                                            
```

Then just delete/skip `synthetic_data.generate_and_save_all()` in
`run_pipeline.py` (or point `cfg.data_root` at your real folder) — nothing
else needs to change. 
