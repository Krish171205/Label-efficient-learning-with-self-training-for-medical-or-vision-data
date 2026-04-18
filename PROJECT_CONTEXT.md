# 📋 Project Context — Label-Efficient Learning with Self-Training

> **Read this file first.** It contains the complete context for this research project.
> If you are an AI assistant being asked to help with this project, read this entire document before doing anything else.

---

## 1. Project Identity

| Field | Value |
|-------|-------|
| **Title** | Label-Efficient Learning with Self-Training for Medical / Vision Data |
| **Type** | Research-based internship project (SEM 6, PICT) |
| **Hardware** | NVIDIA GeForce RTX 4060 (8 GB VRAM, Ada Lovelace SM 8.9) |
| **Software Stack** | Python 3.10–3.12, PyTorch 2.5.1, CUDA 12.4, PyTorch Lightning, timm, lightly |
| **Primary Language** | Python |

---

## 2. Abstract

Large annotated medical/vision datasets are costly to create — every image requires expert annotation (e.g., a radiologist identifying pneumonia and marking its location). This project explores **self-supervised and label-efficient learning** to maximise the utility of unlabelled data. The goal is to build a **self-training pipeline** that achieves near fully-supervised performance using only **1% of the labels**.

---

## 3. The Core Problem

In standard supervised learning, every image needs an expert annotation. For medical data, this means:
- **Thousands of dollars** per dataset
- **Months of time** for radiologist review
- **Bottleneck** that limits how much data can be used for training

**The fundamental question this project answers:** Can a model learn useful structure from raw, unlabeled images, and then use just a handful of labels to get close to fully-supervised performance?

---

## 4. Motivation

Surveys in medical image analysis show that self-training and pseudo-labeling **outperform** standard supervised learning in low-label regimes. The key insight is that **structure in unlabeled data** — how images cluster, what features repeat — is rich signal that labels alone cannot capture.

---

## 5. Related Work

- **Self-training**: Uses a model's own predictions (pseudo-labels) on unlabeled data to expand the training set iteratively.
- **Entropy minimization**: Pushes the model to make sharper, more committed predictions on unlabeled data, which improves generalization.
- **Pseudo-labels**: Treating high-confidence predictions as ground truth for retraining.
- **Gap**: Medical dataset classification using these techniques remains **sparse** in the literature — this project fills that gap.

---

## 6. Technical Deep-Dive: Every Component Explained

### 6a. SimCLR (Contrastive Self-Supervised Pretraining)

**What it does:** Takes one image, creates two distorted versions (random crop, color jitter, Gaussian blur), and trains the backbone to produce the **same embedding** for both views.

**The only training signal:** "These two views came from the same image."

**Why it works:** To satisfy that constraint across 100,000 images and 256 negative pairs per batch, the backbone is **forced to discover truly invariant features** — lung texture, opacity patterns, anatomical shape — rather than superficial properties like lighting or crop position.

**Result:** Without ever seeing a label, similar pathologies cluster together in embedding space.

**Why SimCLR matters for this project:** SimCLR pretraining tackles the **cold-start problem**. Without it, the model in Round 1 of self-training has only 1,000 examples to learn from — not enough to generalize. With a SimCLR backbone, the model already knows that "opacity in the lower lobe looks different from fluid around the heart," even **before seeing a single label**. This means:
- Round 1 predictions on unlabeled data are **less noisy**
- Higher-quality pseudo-labels from the **very first iteration**
- The entire self-training loop starts from a **much stronger position**

**Implementation:** Uses the `lightly` library (v1.5.14) which provides ready-made SimCLR, MoCo, and BYOL frameworks.

---

### 6b. Pretext Tasks (Rotation Prediction & Inpainting)

Cheaper alternatives to SimCLR for bootstrapping the backbone with structural knowledge from unlabeled data.

**Rotation Prediction:**
- Rotate images by 0°, 90°, 180°, 270°
- Train the model to predict which rotation was applied
- Forces the model to understand **orientation-invariant anatomy** — it must know what "upright" looks like

**Inpainting:**
- Mask a random region of the image
- Train the model to reconstruct the masked pixels
- Forces the model to understand **spatial coherence** — what a masked region of a lung should look like given its neighbors
- Uses OpenCV for the masking operations

**The project compares all three** (SimCLR, rotation, inpainting) to find which gives the best downstream representation for medical images specifically.

---

### 6c. The Self-Training Loop (Pseudo-Labeling Pipeline)

This is the **core algorithm** of the project. It works in iterative rounds:

#### Step 0: Starting Condition
- **1,000 labeled images** (radiologist-verified)
- **99,000 unlabeled images**
- Standard supervised learning would ignore the unlabeled pool entirely
- Self-training uses **all of it**

#### Step 1: Train on Labeled Seed
- Train a model on the 1,000 labeled images only → produces a **weak but real classifier**
- The model is initialized with **SimCLR-pretrained weights** (backbone already knows what features matter)
- Result: Model M₀ with ~62% accuracy

#### Step 2: Predict on Unlabeled Data (Inference Pass)
- Run M₀ on all 99,000 unlabeled images
- For each image, get a **sigmoid probability vector** — e.g., `[pneumonia: 0.91, effusion: 0.03, nodule: 0.82, ...]` (14 independent probabilities since ChestMNIST is multi-label)
- Each class has its own confidence score independently
- Nothing is labeled yet — we're just measuring: how confident is the current model on each unlabeled image?

#### Step 3: Threshold + Pseudo-Label (The Filtering Gate)
- Apply confidence threshold **τ = 0.95** per class (multi-label: each of the 14 disease labels is thresholded independently)
- For each class, if `sigmoid(logit) > 0.95` → assign pseudo-label 1 (disease present); if `sigmoid(logit) < 0.05` → assign pseudo-label 0 (disease absent); otherwise → mask that class (too uncertain)
- The model says: "I am confident enough about this specific disease to treat my prediction as ground truth"
- ~18K images have at least one confidently pseudo-labeled class; many classes remain masked per image

**Why τ = 0.95?**
- Too low (0.7) → floods the training set with noisy, wrong labels — the model starts **learning its own mistakes** (confirmation bias)
- Too high (0.999) → almost nothing passes, no benefit
- 0.95 is the **empirically validated sweet spot**
- **Adaptive per-class thresholds** improve on this further (see Section 6e)

#### Step 4: Retrain on Expanded Set (The Feedback Loop)
- Combine: 1,000 labeled + 18,000 pseudo-labeled = **19,000 total training images**
- Retrain from scratch (or fine-tune) → Model M₁
- M₁ is now stronger because it has seen more diverse examples
- Even 5% noise in pseudo-labels is tolerable because errors average out across many examples
- Result: M₁ accuracy ~71% (up from 62%)

#### Step 5: Repeat Until Convergence
- Run M₁ on remaining ~81K unlabeled → more images now pass threshold (model is better)
- Add new pseudo-labels, retrain → M₂
- Repeat for 3–5 rounds until accuracy plateaus

**The Compounding Effect:**
```
Round 0: 1,000 labeled → M₀ 62%
Round 1: 19,000 total  → M₁ 71%  (+9%)
Round 2: 42,000 total  → M₂ 77%  (+6%)
Round 3: 70,000 total  → M₃ 81%  (+4%)
Round 4: 90,000 total  → M₄ 84%  (+3%) — convergence
Ceiling (100% labels): 88%

Result: 95% of fully-supervised performance using only 1% of labels
```

The labeled set grows **exponentially** per round because each improved model → more confident predictions → more pass threshold → bigger training set → better model.

---

### 6d. Entropy Minimization

**The problem it solves:** A model that has learned something true will often still output **soft, uncertain probabilities** (e.g., `[0.6, 0.3, 0.1]`) rather than sharp decisions.

**How it works:** Add an **entropy penalty** on unlabeled predictions. This pushes the model to commit — to output `[0.97, 0.02, 0.01]` instead of `[0.6, 0.3, 0.1]`.

**Why it helps:**
- Doesn't just help thresholding — it actually **improves generalization** by making decision boundaries sharper
- Forces the model to pick a side on ambiguous examples
- Reduces the number of uncertain predictions near the threshold

---

### 6e. Adaptive Thresholds (Per-Class)

**The problem:** In ChestX-ray14, pneumonia is **far more common** than pneumothorax. A fixed τ=0.95 will:
- **Flood** the pseudo-label set with pneumonia examples (easy, high-confidence class)
- Include **almost nothing** from rare classes (the model is less confident on them)
- Result: severe class imbalance in pseudo-labels

**The solution:** Per-class adaptive thresholds:
- **Scale τ down** for rare classes (letting more uncertain examples in)
- **Scale τ up** for common classes (being stricter to avoid flooding)
- This **balances the training signal** across all disease categories

---

### 6f. Annotation Selection Metrics (Active Learning)

**The question:** If you have a budget of N annotations (e.g., 1,000 images a radiologist can label), **which 1,000 images** should you label?

**Why it matters:** You want images that are **maximally informative** — covering the full distribution of the unlabeled data, not just the easy cases. This is what separates a research system from a naive one.

**Two strategies implemented:**

1. **Uncertainty Sampling:**
   - Select images where the model is **most uncertain** (highest entropy in predictions)
   - These are the images at decision boundaries — labeling them provides maximum information
   - Uses `scikit-learn` for implementation

2. **Core-Set Selection:**
   - Select images that are **most representative** of the full unlabeled distribution
   - Ensures coverage of the entire data manifold, not just the hard cases
   - Finds images that maximize the minimum distance to already-labeled points in feature space

---

## 7. Methodology Summary

The project compares four approaches against a transfer learning baseline:

| Method | Type | What it does |
|--------|------|-------------|
| **Transfer Learning** | Baseline | ImageNet-pretrained backbone + fine-tune on labeled data only |
| **SimCLR** | Self-supervised | Contrastive pretraining on unlabeled data → fine-tune |
| **Rotation Prediction** | Pretext task | Predict rotation angle → fine-tune |
| **Inpainting** | Pretext task | Reconstruct masked regions → fine-tune |
| **Pseudo-Labeling** | Self-training | Iterative: predict → threshold → retrain loop |

All methods are evaluated in the **self-training loop** context (combined with pseudo-labeling) and compared on:
- Accuracy at various label fractions (1%, 5%, 10%, 25%)
- AUROC for multi-label classification
- Convergence speed (rounds to plateau)
- Pseudo-label quality (noise rate)

---

## 8. Dataset

### Primary (and only): ChestMNIST

| Property | Value |
|----------|-------|
| **Source** | NIH ChestX-ray14, downscaled to 224×224 via MedMNIST benchmark |
| **Total Images** | 112,120 (train: 78,468 / val: 11,219 / test: 22,433) |
| **Classes** | 14 disease categories (multi-label) |
| **Task** | Multi-label binary classification |
| **Download Size** | ~3.9 GB at 224×224 resolution (compressed .npz) |
| **Download** | Auto-downloads via `python dataset_download.py` |
| **Storage** | `Internship/data/chestmnist_224.npz` (project-local) |

**The 14 disease categories:**
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

**Multi-label implications for the pipeline:**
- Use **sigmoid + BCE loss** (not softmax + cross-entropy)
- Pseudo-label thresholding is applied **per-class independently** (each of 14 classes gets its own threshold)
- Evaluation uses **AUROC per class** rather than simple accuracy
- Adaptive thresholds are especially important — disease prevalence varies wildly across the 14 categories

### Optional / Future
- **ChestX-ray14 full resolution** (~45 GB) — same data at original resolution
- **COCO Unlabeled** (~40 GB) — for object detection extension, not in current scope

---

## 9. Expected Outcomes

1. **Prototype** yielding state-of-the-art results in label-scarce settings
2. Achieve **~95% of fully-supervised performance** using only **1% of labels**
3. **Robust annotation selection metrics** that identify the most informative images to label
4. **Comparative analysis** of SimCLR vs. pretext tasks vs. transfer learning for medical image features
5. Evidence that **self-training + self-supervised pretraining** is the optimal combination for medical imaging

---

## 10. Project Structure

```
Internship/
├── venv/                          ← virtual environment
├── data/                          ← datasets
│   ├── chestmnist_224.npz         ← compressed dataset (3.9 GB)
│   └── chestmnist_npy/            ← uncompressed .npy files (for multi-worker loading)
│       ├── train_images.npy       ← created by convert_to_npy.py
│       ├── train_labels.npy
│       ├── val_images.npy
│       ├── val_labels.npy
│       ├── test_images.npy
│       └── test_labels.npy
├── configs/
│   └── default.yaml               ← centralized hyperparameters
├── src/
│   ├── data/
│   │   ├── chest_mnist.py         ← data loading, label-fraction splitting
│   │   └── lazy_dataset.py        ← memory-mapped dataset (Windows multiprocessing fix)
│   ├── models/
│   │   └── classifier.py          ← ChestClassifier (ResNet-18 + 14-class head)
│   ├── simclr/
│   │   ├── model.py               ← SimCLR architecture (backbone + projection head)
│   │   └── augmentations.py       ← dual-view augmentation pipeline
│   ├── pretext/
│   │   ├── rotation.py            ← rotation prediction (4-class)
│   │   └── inpainting.py          ← masked region reconstruction
│   ├── self_training/
│   │   └── pseudo_labels.py       ← pseudo-label generation + adaptive thresholds
│   ├── active_learning/
│   │   └── strategies.py          ← uncertainty sampling + core-set selection
│   └── utils/
│       ├── config.py              ← YAML config loader
│       ├── device.py              ← GPU setup, seed, mixed precision
│       ├── training.py            ← training engine (train/eval/checkpoints)
│       └── metrics.py             ← AUROC, mAP, F1 metrics
├── train_baseline.py              ← supervised transfer learning baseline
├── train_simclr.py                ← SimCLR contrastive pretraining
├── train_simclr_finetune.py       ← fine-tune SimCLR backbone on labeled data
├── train_pretext.py               ← rotation/inpainting pretraining
├── train_pretext_finetune.py      ← fine-tune pretext backbone on labeled data
├── train_self_training.py         ← iterative pseudo-labeling pipeline
├── run_comparison.py              ← collect results + generate comparison plots
├── convert_to_npy.py              ← one-time: npz → npy conversion for multi-worker loading
├── dataset_download.py            ← ChestMNIST downloader
├── check_setup.py                 ← dependency/GPU verification
├── PROJECT_CONTEXT.md             ← THIS FILE
├── SETUP_GUIDE.md                 ← environment setup instructions
└── requirements.txt               ← pinned dependencies
```

---

## 11. Key Dependencies

```
# Core DL
torch==2.5.1 (CUDA 12.4)    torchvision==0.20.1    lightning==2.4.0
timm==1.0.9                  torchmetrics==1.4.3

# Self-supervised
lightly==1.5.14              # SimCLR, MoCo, BYOL

# Datasets
medmnist==3.0.2

# Augmentation
albumentations==1.4.21       opencv-python==4.10.0.84    Pillow==11.0.0

# Scientific
numpy==1.26.4    pandas==2.2.3    scikit-learn==1.5.2    scipy==1.14.1

# Visualization & Tracking
matplotlib==3.9.3    seaborn==0.13.2    tensorboard==2.18.0    wandb==0.18.7

# Utilities
tqdm==4.67.1    pyyaml==6.0.2    rich==13.9.4    einops==0.8.0
```

---

## 12. Hardware Constraints & Tips

| Setting | Recommended Value |
|---------|------------------|
| Batch size (supervised) | 32–64 |
| Batch size (SimCLR) | 128–256 (gradient accumulation if OOM) |
| Image resolution | 224×224 |
| Backbone | ResNet-18 or ResNet-50 (avoid ViT-Large) |
| Mixed precision | **Always ON** (`torch.amp.autocast` + `GradScaler`) |

**OOM mitigation:**
1. Reduce batch size
2. `torch.cuda.empty_cache()` between phases
3. Gradient accumulation (effective batch=256, actual batch=64)
4. Gradient checkpointing via `timm`

---

## 13. Conceptual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: BACKBONE PRETRAINING                   │
│                                                                     │
│   100K Unlabeled Images ──→ SimCLR / Rotation / Inpainting         │
│                              ──→ Pretrained Backbone               │
│                                   (knows features, no labels)       │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: SELF-TRAINING LOOP                     │
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐ │
│   │ Train on │    │ Predict  │    │ Threshold│    │ Add pseudo-  │ │
│   │ labeled  │───→│ unlabeled│───→│ + filter │───→│ labels to    │ │
│   │ + pseudo │    │ data     │    │ τ > 0.95 │    │ training set │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────┬───────┘ │
│        ▲                                                  │         │
│        └──────────────────────────────────────────────────┘         │
│                         repeat 3–5 rounds                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: EVALUATION                             │
│                                                                     │
│   Compare: SimCLR vs Rotation vs Inpainting vs Transfer Learning   │
│   Metrics: Accuracy, AUROC, F1 at 1%, 5%, 10%, 25% label fractions│
│   Active Learning: Uncertainty Sampling vs Core-Set Selection      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 14. Key Terminology Quick Reference

| Term | Meaning |
|------|---------|
| **Pseudo-label** | A predicted label assigned by the model (not a human) with high confidence |
| **Self-training** | Iterative process: train → predict → threshold → retrain |
| **SimCLR** | Contrastive learning: pull augmented views of same image together, push different images apart |
| **Pretext task** | Auxiliary task (rotation, inpainting) that forces the model to learn useful representations without labels |
| **Entropy minimization** | Loss penalty that forces the model to make sharper, more confident predictions |
| **Confidence threshold (τ)** | Minimum softmax probability required to accept a pseudo-label (default 0.95) |
| **Adaptive threshold** | Per-class threshold that adjusts τ based on class frequency to handle imbalance |
| **Uncertainty sampling** | Active learning strategy: label the images the model is most confused about |
| **Core-set selection** | Active learning strategy: label images that maximally cover the data distribution |
| **Cold-start problem** | The difficulty of making good predictions in Round 1 with very few labels |
| **Confirmation bias** | Risk of the model reinforcing its own mistakes through noisy pseudo-labels |
| **Label fraction** | Percentage of total data that has human annotations (this project targets 1%) |

---

## 15. Final Results ✅

All experiments completed. **Best method: Self-Training with ImageNet backbone.**

| Method | Test AUROC | mAP | Training Time |
|--------|-----------|-----|---------------|
| Baseline (ImageNet transfer, 784 labels) | 0.6738 | 0.102 | 9.4 min |
| SimCLR → Fine-tune (784 labels) | 0.6702 | 0.097 | 3.7 min |
| Rotation → Fine-tune (784 labels) | 0.6558 | 0.096 | 2.8 min |
| **Self-Train (ImageNet backbone)** | **0.7807** 🏆 | **0.199** | 55.7 min |
| **Self-Train (SimCLR backbone)** | **0.7626** | **0.171** | 63.7 min |

### Key Findings:
1. Self-training with pseudo-labeling is the **clear winner** — +15.9% AUROC over baseline
2. Both self-training methods reached the "clinically useful" range (>0.75) using **only 1% labels**
3. ImageNet backbone beat SimCLR (limited training: 20 epochs/40K vs ImageNet's 1.2M images)
4. Self-training expanded labeled pool: **784 → 43,020 images** (55x growth)
5. Model correctly rejected ~35K uncertain images — prevents confirmation bias

### Self-Training Progression (ImageNet backbone):
```
Round 0: AUROC 0.6711  (pool: 784 labeled)
Round 1: AUROC 0.7740  (pool: 9,236)       ← biggest jump
Round 2: AUROC 0.7829  (pool: 23,082)
Round 3: AUROC 0.7870  (pool: 34,159)
Round 4: AUROC 0.7902  (pool: 43,020)
Test:    AUROC 0.7807
```

### Generated Outputs:
- `outputs/results/comparison_summary.json` — all results
- `outputs/results/plots/auroc_comparison.png` — bar chart comparing methods
- `outputs/results/plots/self-train_imagenet_progression.png` — AUROC progression
- `outputs/results/plots/self-train_simclr_progression.png` — AUROC progression
- `outputs/checkpoints/` — saved model weights

---

## 16. All Tasks Complete

- ✅ Environment setup, dataset download + .npy conversion
- ✅ Baseline (ImageNet transfer, 1% labels) — AUROC: 0.6738
- ✅ SimCLR pretraining (20 epochs, 40K images) + fine-tune — AUROC: 0.6702
- ✅ Rotation pretraining (15 epochs, 40K images) + fine-tune — AUROC: 0.6558
- ✅ Self-training with ImageNet backbone (5 rounds) — AUROC: **0.7807** 🏆
- ✅ Self-training with SimCLR backbone (5 rounds) — AUROC: 0.7626
- ✅ Final comparison + plot generation
- ⏭ Inpainting skipped (rotation comparable, saved ~3 hours)

### Bug Fixes Applied:
- Unpicklable lambdas → `ConvertToRGB` class
- Windows multiprocessing OOM → `lazy_dataset.py` with memory-mapped .npy
- SimCLR GPU OOM → batch_size 128→64
- Pseudo-label overflow → `.astype(int).sum()`
- Pseudo-label thresholds → percentile-based (top 3% per class)
- Pseudo-label flooding → 20% per-round cap

---

---

## 17. Known Issues & Fixes

### Windows Multiprocessing (`num_workers > 0`)
- **Problem**: MedMNIST loads all images into RAM → pickle crash.
- **Fix**: `src/data/lazy_dataset.py` memory-maps .npy files. Run `python convert_to_npy.py` once.
- **Setting**: `num_workers: 4`, `persistent_workers=True`.

### Low Disk Space
- **Problem**: <10 GB free → page file can't expand → error 1455.
- **Fix**: Need ~30+ GB free on C: drive.

### GPU OOM with SimCLR
- **Problem**: batch_size=128 OOM on 8 GB VRAM (2 forward passes).
- **Fix**: SimCLR batch_size=64 in config.

---

## 18. Run Order (Verified — All Completed)

```bash
# Activate venv
venv\Scripts\activate

# Step 0: One-time data prep (~2 min)
python scripts/dataset_download.py                        # Download ChestMNIST
python scripts/convert_to_npy.py                          # .npz → .npy for workers

# Step 1: Baseline (~10 min)
python scripts/train_baseline.py                          # → AUROC: 0.6738

# Step 2: SimCLR (~3.5h total)
python scripts/train_simclr.py                            # Contrastive pretrain
python scripts/train_simclr_finetune.py                   # → AUROC: 0.6702

# Step 3: Rotation (~2.5h total)
python scripts/train_pretext.py --task rotation           # Rotation pretrain
python scripts/train_pretext_finetune.py --task rotation  # → AUROC: 0.6558

# Step 4: Self-training (~1h each)
python scripts/train_self_training.py --backbone imagenet # → AUROC: 0.7807 🏆
python scripts/train_self_training.py --backbone simclr   # → AUROC: 0.7626

# Step 5: Compare + plots (~1 min)
python scripts/run_comparison.py                          # Generates plots + JSON
```

---

## 19. File Organization

```
Internship/
├── scripts/                       ← All executable scripts
│   ├── train_baseline.py          ← Supervised baseline
│   ├── train_simclr.py            ← SimCLR pretraining
│   ├── train_simclr_finetune.py   ← Fine-tune SimCLR backbone
│   ├── train_pretext.py           ← Rotation/inpainting pretrain
│   ├── train_pretext_finetune.py  ← Fine-tune pretext backbone
│   ├── train_self_training.py     ← Self-training loop
│   ├── run_comparison.py          ← Collect results + plots
│   ├── convert_to_npy.py          ← One-time data conversion
│   ├── dataset_download.py        ← Download ChestMNIST
│   └── check_setup.py             ← Verify environment
├── src/                           ← Library code (imported, NOT run directly)
│   ├── data/                      ← Dataset classes + transforms
│   ├── models/                    ← Classifier architecture
│   ├── simclr/                    ← SimCLR model + augmentations
│   ├── pretext/                   ← Rotation + inpainting models
│   ├── self_training/             ← Pseudo-label generation engine
│   ├── active_learning/           ← Uncertainty + core-set strategies
│   └── utils/                     ← Config, device, training loop, metrics
├── configs/default.yaml           ← Hyperparameters
├── outputs/
│   ├── checkpoints/               ← Saved model weights (per experiment)
│   └── results/                   ← JSON results + PNG comparison plots
├── data/                          ← Dataset files
│   ├── chestmnist_224.npz         ← Compressed (3.9 GB)
│   └── chestmnist_npy/            ← Uncompressed .npy (for mmap workers)
├── docs/                          ← Documentation
│   ├── EXPLAINED.md               ← Plain-English explanation of everything
│   └── WEBSITE_IDEAS.md           ← Frontend/backend roadmap
├── PROJECT_CONTEXT.md             ← THIS FILE
├── SETUP_GUIDE.md                 ← Environment setup instructions
└── requirements.txt               ← Pinned Python dependencies
```

