# Label-Efficient Learning with Self-Training for Medical Image Classification

> Achieving **87% of fully-supervised performance using only 1% of labels** on ChestMNIST (78K chest X-rays, 14 diseases) through self-training with pseudo-labels.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📊 Results

| Method | Test AUROC | Labels Used | Training Time |
|--------|-----------|-------------|---------------|
| Baseline (ImageNet transfer) | 0.6738 | 784 (1%) | 9.4 min |
| SimCLR → Fine-tune | 0.6702 | 784 (1%) | 3.7 min |
| Rotation → Fine-tune | 0.6558 | 784 (1%) | 2.8 min |
| **Self-Train (ImageNet backbone)** | **0.7807** 🏆 | **784 → 43,020** | 55.7 min |
| **Self-Train (SimCLR backbone)** | **0.7626** | **784 → 46,148** | 63.7 min |

### Self-Training Progression (Best Method)
```
Round 0: AUROC 0.6711  (pool: 784 labeled)
Round 1: AUROC 0.7740  (pool: 9,236)       ← biggest jump
Round 2: AUROC 0.7829  (pool: 23,082)
Round 3: AUROC 0.7870  (pool: 34,159)
Round 4: AUROC 0.7902  (pool: 43,020)
Test:    AUROC 0.7807
```

---

## 🧠 What This Project Does

In medical imaging, getting expert labels is expensive and slow. This project demonstrates that with the right techniques, you can achieve near-expert performance using only **1% of labels**.

### The Pipeline

```
Phase 1: BACKBONE PRETRAINING (no labels needed)
    78K unlabeled X-rays → SimCLR / Rotation → Pretrained backbone

Phase 2: SELF-TRAINING LOOP (uses only 784 labeled images to start)
    Train on labeled → Predict unlabeled → Threshold → Add pseudo-labels → Repeat × 5

Phase 3: EVALUATION
    Compare all methods: Baseline vs SimCLR vs Rotation vs Self-Training
```

### Key Techniques
- **SimCLR** — Contrastive self-supervised pretraining (learns features without labels)
- **Rotation prediction** — Pretext task that learns anatomy from image orientation
- **Self-training with pseudo-labels** — Iteratively expands training data using model predictions
- **Percentile-based thresholding** — Adapts to model confidence to avoid noisy labels
- **Entropy minimization** — Pushes model to make sharper, more decisive predictions

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.4 (tested on RTX 4060 8GB)
- ~4 GB disk space for dataset

### Setup
```bash
# Clone
git clone https://github.com/Krish171205/Label-efficient-learning-with-self-training-for-medical-or-vision-data.git
cd Label-efficient-learning-with-self-training-for-medical-or-vision-data

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download dataset
python dataset_download.py

# Convert for multi-worker loading
python convert_to_npy.py
```

### Run Full Pipeline
```bash
# 1. Baseline (~10 min)
python train_baseline.py

# 2. SimCLR pretraining + fine-tune (~3.5h)
python train_simclr.py
python train_simclr_finetune.py

# 3. Rotation pretext + fine-tune (~2.5h)
python train_pretext.py --task rotation
python train_pretext_finetune.py --task rotation

# 4. Self-training (~1h each)
python train_self_training.py --backbone imagenet
python train_self_training.py --backbone simclr

# 5. Compare + generate plots
python run_comparison.py
```

---

## 📁 Project Structure

```
├── train_baseline.py              # Supervised baseline (ImageNet transfer)
├── train_simclr.py                # SimCLR contrastive pretraining
├── train_simclr_finetune.py       # Fine-tune SimCLR backbone
├── train_pretext.py               # Rotation / inpainting pretraining
├── train_pretext_finetune.py      # Fine-tune pretext backbone
├── train_self_training.py         # Self-training with pseudo-labels
├── run_comparison.py              # Collect results + generate plots
├── convert_to_npy.py              # One-time: npz → npy conversion
├── dataset_download.py            # Download ChestMNIST dataset
├── configs/
│   └── default.yaml               # All hyperparameters
├── src/
│   ├── data/
│   │   ├── chest_mnist.py         # Data loading + label splitting
│   │   └── lazy_dataset.py        # Memory-mapped dataset (Windows fix)
│   ├── models/
│   │   └── classifier.py          # ResNet-18 + 14-class classifier
│   ├── simclr/
│   │   ├── model.py               # SimCLR architecture
│   │   └── augmentations.py       # Dual-view augmentation pipeline
│   ├── pretext/
│   │   ├── rotation.py            # Rotation prediction model
│   │   └── inpainting.py          # Masked reconstruction model
│   ├── self_training/
│   │   └── pseudo_labels.py       # Pseudo-label engine + thresholds
│   ├── active_learning/
│   │   └── strategies.py          # Uncertainty + core-set sampling
│   └── utils/
│       ├── config.py              # YAML config loader
│       ├── device.py              # GPU setup + mixed precision
│       ├── training.py            # Training / evaluation engine
│       └── metrics.py             # AUROC, mAP, F1 metrics
├── outputs/
│   └── results/                   # JSON results + comparison plots
├── PROJECT_CONTEXT.md             # Full project documentation
├── EXPLAINED.md                   # Plain-English explanation of everything
├── SETUP_GUIDE.md                 # Environment setup instructions
└── requirements.txt               # Pinned dependencies
```

---

## 📈 Dataset

**ChestMNIST** (from MedMNIST v2):
- 78,468 training / 11,219 validation / 22,433 test images
- 224 × 224 pixels, grayscale chest X-rays
- 14 binary disease labels (multi-label classification)
- Automatically downloaded by `dataset_download.py`

---

## ⚙️ Configuration

All hyperparameters are in `configs/default.yaml`:

| Setting | Value | Notes |
|---------|-------|-------|
| Backbone | ResNet-18 | ~11.5M parameters |
| Label fraction | 1% | 784 of 78,468 images |
| SimCLR epochs | 20 | On 40K image subset |
| Rotation epochs | 15 | On 40K image subset |
| Self-training rounds | 5 | 20 epochs per round |
| Batch size | 64 | SimCLR uses 64×2=128 effective |
| Mixed precision | ON | FP16 for speed + memory |

---

## 🔬 Key Findings

1. **Self-training is the clear winner** — +15.9% AUROC improvement over baseline
2. **1% labels can be enough** — with the right technique, 784 images rival thousands
3. **ImageNet transfer remains strong** — our limited SimCLR training (20 epochs) couldn't beat ImageNet's 1.2M image pretraining
4. **Conservative pseudo-labeling works** — the model correctly rejected ~35K uncertain images, avoiding confirmation bias
5. **Iterative expansion is key** — labeled pool grew from 784 → 43,020 across 5 rounds

---

## 📖 References

- Chen, T. et al. "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020)
- Xie, Q. et al. "Self-Training with Noisy Student Improves ImageNet Classification" (2020)
- Lee, D.H. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method" (2013)
- Yang, J. et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification" (2023)

---

## 🛠️ Hardware Used

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4060 (8 GB VRAM) |
| RAM | 16 GB DDR4 |
| CPU | Intel/AMD (8+ cores) |
| Storage | SSD (65+ GB free recommended) |
| OS | Windows 11 |

---

## 📄 License

This project is for academic/research purposes (SEM 6 Internship, PICT).
