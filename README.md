# Label-Efficient Learning with Self-Training for Medical Image Classification

> Achieving **AUROC 0.807** on ChestMNIST using only **1% of available labels** (784 of 78,468 images) through iterative self-training with adaptive pseudo-labeling.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange?logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

Medical imaging AI requires large annotated datasets — but expert annotation (e.g., a radiologist labeling chest X-rays) is expensive and slow. This project explores a **label-efficient pipeline** that combines self-supervised pretraining with iterative pseudo-labeling to approach fully-supervised performance with dramatically fewer labels.

**Core idea:** Train a backbone with no labels at all (SimCLR contrastive learning), then iteratively expand the labeled pool by using the model's own high-confidence predictions as pseudo-labels — growing from 784 images to 43,000+ over 10 rounds without any additional human annotation.

### Key Results

| Method | Test AUROC | mAP | Labels Used |
|--------|-----------|-----|-------------|
| Supervised Baseline (ImageNet) | 0.6562 | 0.094 | 784 (1%) |
| Rotation Pretext → Fine-tune | 0.6288 | 0.082 | 784 (1%) |
| SimCLR → Fine-tune | 0.6602 | 0.092 | 784 (1%) |
| **Self-Train (SimCLR backbone)** | **0.7928** | **0.199** | 784 seed |
| **Self-Train (ImageNet backbone)** | **0.8070** | **0.220** | 784 seed |

> **+23% AUROC improvement** over the supervised baseline, with zero additional human annotations.

---

## Pipeline Architecture

```
Phase 1 — Backbone Pretraining (no labels required)
┌─────────────────────────────────────────────────────┐
│  78,468 Unlabeled X-rays                            │
│       │                                             │
│       ├──→ SimCLR (contrastive, NT-Xent loss)       │
│       └──→ Rotation Prediction (pretext, 0/90/180/270°) │
│                    │                                │
│             Pretrained Backbone                     │
└─────────────────────────────────────────────────────┘
                     │
Phase 2 — Self-Training Loop (784 labeled seed)
                     │
┌────────────────────▼────────────────────────────────┐
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐  │
│  │ Train on │───→│ Predict  │───→│ Adaptive      │  │
│  │ labeled  │    │ unlabeled│    │ threshold     │  │
│  │ + pseudo │    │ pool     │    │ (97th pctile) │  │
│  └──────────┘    └──────────┘    └──────┬────────┘  │
│       ▲                                 │           │
│       └─────────── Add pseudo-labels ───┘           │
│              Repeat for 10 rounds                   │
└─────────────────────────────────────────────────────┘
```

---

## Dataset

**ChestMNIST** (MedMNIST v2 benchmark) — derived from NIH ChestX-ray14:
- **78,468** training images / **11,219** validation / **22,433** test
- **14 disease classes** (multi-label): Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
- **224×224 pixels**, automatically downloaded (~3.9 GB)

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Krish171205/Label-efficient-learning-with-self-training-for-medical-or-vision-data.git
cd Label-efficient-learning-with-self-training-for-medical-or-vision-data

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/dataset_download.py
python scripts/convert_to_npy.py   # One-time conversion for fast multi-worker loading
```

### 3. Run Full Pipeline (Automated)

```bash
python scripts/run_full_pipeline.py --config configs/default.yaml
```

This runs all steps in sequence with automatic checkpointing. If interrupted, re-run the same command to resume from the last completed step.

### 4. Run Individual Steps

```bash
# Step 1: Supervised baseline (transfer learning)
python scripts/train_baseline.py

# Step 2: SimCLR self-supervised pretraining
python scripts/train_simclr.py

# Step 3: Fine-tune SimCLR backbone on labeled data
python scripts/train_simclr_finetune.py

# Step 4: Rotation pretext task pretraining
python scripts/train_pretext.py --task rotation

# Step 5: Fine-tune rotation backbone
python scripts/train_pretext_finetune.py --task rotation

# Step 6: Self-training with pseudo-labeling (choose backbone)
python scripts/train_self_training.py --backbone imagenet
python scripts/train_self_training.py --backbone simclr

# Step 7: Generate comparison plots and results JSON
python scripts/run_comparison.py
```

All scripts support `--resume` to continue from the last saved checkpoint:

```bash
python scripts/train_self_training.py --backbone imagenet --resume
```

---

## Project Structure

```
├── configs/
│   └── default.yaml              # All hyperparameters (label fraction, batch size, epochs, etc.)
├── scripts/
│   ├── run_full_pipeline.py      # Orchestrates all steps end-to-end
│   ├── train_baseline.py         # ImageNet transfer learning baseline
│   ├── train_simclr.py           # SimCLR contrastive pretraining
│   ├── train_simclr_finetune.py  # Fine-tune SimCLR backbone
│   ├── train_pretext.py          # Rotation / inpainting pretraining
│   ├── train_pretext_finetune.py # Fine-tune pretext backbone
│   ├── train_self_training.py    # Iterative pseudo-labeling loop
│   ├── run_comparison.py         # Aggregate results + generate plots
│   ├── dataset_download.py       # Download ChestMNIST
│   └── convert_to_npy.py         # Convert .npz → .npy for fast loading
├── src/
│   ├── data/
│   │   ├── chest_mnist.py        # Dataset class with label-fraction splitting
│   │   └── lazy_dataset.py       # Memory-mapped dataset (multi-worker on Windows)
│   ├── models/
│   │   └── classifier.py         # ResNet-18 + 14-class sigmoid head
│   ├── simclr/
│   │   ├── model.py              # SimCLR with projection head
│   │   └── augmentations.py      # Dual-view augmentation pipeline
│   ├── pretext/
│   │   ├── rotation.py           # Rotation prediction (4-class)
│   │   └── inpainting.py         # Masked region reconstruction
│   ├── self_training/
│   │   └── pseudo_labels.py      # Adaptive threshold pseudo-label engine
│   ├── active_learning/
│   │   └── strategies.py         # Uncertainty sampling + core-set selection
│   └── utils/
│       ├── config.py             # YAML config loader
│       ├── device.py             # GPU setup + mixed precision
│       ├── training.py           # Training loop + checkpointing
│       └── metrics.py            # AUROC, mAP, F1 (multi-label)
├── requirements.txt
└── README.md
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
data:
  label_fraction: 0.01    # 1% of training data is labeled
  num_workers: 4          # Set to 0 on Windows if you hit shared-memory errors
  image_size: 224

training:
  batch_size: 64
  epochs_simclr: 100
  self_training_rounds: 10
  lr: 0.0001

self_training:
  confidence_percentile: 97   # Top 3% most confident per class → pseudo-label
  max_per_round_fraction: 0.20
```

---

## Methods

### SimCLR Contrastive Pretraining
Trains ResNet-18 using the NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss on pairs of augmented views of the same image, with no labels. The loss for a positive pair (i, j) in a batch of N images:

```
L = -log [ exp(sim(zᵢ, zⱼ)/τ) / Σₖ₌₁²ᴺ 1[k≠i] exp(sim(zᵢ, zₖ)/τ) ]
```

After pretraining, the projection head is discarded and the backbone is fine-tuned on labeled data.

### Adaptive Pseudo-Labeling
Standard fixed-threshold pseudo-labeling fails on imbalanced multi-label datasets (common diseases flood the pseudo-label set; rare diseases are never labeled). We use per-class 97th-percentile thresholds:

```
τ[k] = min(percentile₉₇({p_k(x) : x ∈ D_unlabeled}), 0.95)
```

This accepts the top 3% most confident predictions per class regardless of absolute confidence, ensuring all 14 disease classes receive pseudo-labels proportionally.

### Rotation Pretext Task
Trains the backbone to classify 4 rotations (0°, 90°, 180°, 270°) applied to unlabeled images. Forces the model to learn anatomical orientation — which lobes are upper/lower, where the diaphragm sits — without any disease labels.

---

## Requirements

| Package | Version |
|---------|---------|
| torch | 2.5.1 (CUDA 12.4) |
| torchvision | 0.20.1 |
| lightly | 1.5.14 |
| medmnist | 3.0.2 |
| timm | 1.0.9 |
| scikit-learn | 1.5.2 |
| numpy | 1.26.4 |

Full list: [`requirements.txt`](requirements.txt)

**Hardware:** Tested on NVIDIA RTX 4060 (8 GB VRAM). Minimum 6 GB VRAM recommended. For GPUs with ≥ 16 GB, increase `batch_size` to 128+ for better SimCLR performance.

---

## Reproducing Results

The full pipeline takes approximately **15 hours** on an RTX 4060 (8 GB). On a cloud GPU (T4/A100), run with `configs/default.yaml` and increase `num_workers` to 4.

Expected outputs after full run:
```
outputs/
├── checkpoints/              # Saved model weights per step
└── results/
    ├── comparison_summary.json
    └── plots/
        ├── auroc_comparison.png
        ├── self_train_imagenet_progression.png
        └── self_train_simclr_progression.png
```

---

## References

1. T. Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
2. K. Sohn et al. "FixMatch: Simplifying Semi-Supervised Learning." NeurIPS 2020. [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)
3. B. Zhang et al. "FlexMatch: Curriculum Pseudo Labeling." NeurIPS 2021. [arXiv:2110.08263](https://arxiv.org/abs/2110.08263)
4. J. Yang et al. "MedMNIST v2." Scientific Data, 2023. [DOI:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)
5. S. Azizi et al. "Big Self-Supervised Models Advance Medical Image Classification." ICCV 2021. [arXiv:2101.05224](https://arxiv.org/abs/2101.05224)
6. X. Wang et al. "NIH ChestX-ray14." CVPR 2017.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
