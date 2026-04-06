# 🛠️ Complete Setup Guide — Self-Training Pipeline

Everything you need to install, download, and configure **before writing a single line of training code**.

---

## 1. Prerequisites (one-time system setup)

### 1a. Python 3.10, 3.11, or 3.12

Check if you have it:
```
python --version
```
If not installed → download from https://www.python.org/downloads/  
**Tick "Add Python to PATH"** during installation.

---

### 1b. NVIDIA CUDA Toolkit 12.4

Your RTX 4060 driver likely already supports CUDA 12.x. Verify:
```
nvidia-smi
```
Look at the top-right corner — it shows `CUDA Version: 12.x`.  

- **If 12.4 or higher** → you're good, skip this.
- **If lower or missing** → download CUDA Toolkit 12.4 from:  
  👉 https://developer.nvidia.com/cuda-12-4-0-download-archive  
  Choose: **Windows → x86_64 → 11 → exe (local)**  
  Install with default options. Restart your PC after.

---

### 1c. cuDNN (Optional but recommended for speed)

cuDNN accelerates convolutions on your GPU.

1. Go to https://developer.nvidia.com/cudnn-downloads
2. Sign in / create a free NVIDIA account
3. Download **cuDNN 9.x for CUDA 12** (Windows, zip)
4. Extract and copy the contents into your CUDA install folder:
   - Copy `bin\*.dll` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\`
   - Copy `include\*.h` → `...\include\`
   - Copy `lib\x64\*.lib` → `...\lib\x64\`

> **Note:** PyTorch bundles its own cuDNN, so this is optional. It ensures system-wide acceleration.

---

## 2. Python Environment Setup

Run these commands **in order** from your project directory:

```bash
# Step 1: Create virtual environment
python -m venv venv

# Step 2: Activate it
venv\Scripts\activate

# Step 3: Upgrade pip
python -m pip install --upgrade pip

# Step 4: Install PyTorch with CUDA 12.4 support (THIS MUST COME FIRST)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Step 5: Install all other dependencies
pip install -r requirements.txt
```

### Verify GPU Works

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), 'GB')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
VRAM: 8.0 GB
```

If `CUDA available: False` → your CUDA toolkit version doesn't match. Re-install PyTorch with the correct CUDA version from https://pytorch.org/get-started/locally/

---

## 3. Force All Training to Use Your RTX 4060

CUDA only sees NVIDIA GPUs (your integrated Intel/AMD GPU is invisible to it), so your RTX 4060 is **CUDA GPU 0**. Add this to the top of every training script:
 
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # RTX 4060 (only CUDA GPU)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} — {torch.cuda.get_device_name(0)}")
```

Or set it globally in your terminal before running anything:
```bash
set CUDA_VISIBLE_DEVICES=0
```

---

## 4. Dataset Download

### Primary Dataset: ChestMNIST (Auto-download, ~3.9 GB)

We use **ChestMNIST** from the MedMNIST benchmark as our sole primary dataset.

| Property | Value |
|----------|-------|
| **Source** | NIH ChestX-ray14, downscaled to 224×224 |
| **Images** | 112,120 (train: 78,468 / val: 11,219 / test: 22,433) |
| **Classes** | 14 disease categories |
| **Task** | Multi-label binary classification |
| **Size** | ~3.9 GB at 224×224 resolution (compressed .npz) |

**Download with one command:**
```bash
python dataset_download.py
```

Or in Python directly:
```python
import medmnist
from medmnist import INFO

DataClass = getattr(medmnist, INFO["chestmnist"]["python_class"])
for split in ["train", "val", "test"]:
    dataset = DataClass(split=split, download=True, size=224, root="data")
    print(f"✓ chestmnist [{split}] — {len(dataset)} images")
```

Data saves to `Internship/data/chestmnist_224.npz`. No sign-up, no manual downloads.

**The 14 disease categories:**
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

> **Note:** ChestMNIST is multi-label — a single image can have multiple diseases. This means we use **sigmoid + BCE loss** (not softmax + cross-entropy) and per-class thresholding for pseudo-labels. This makes the project's adaptive threshold component even more relevant.

---

### Optional / Future: ChestX-ray14 Full Resolution (~45 GB)

The full-resolution version of the same dataset. Only needed if you want to scale up after validating everything on ChestMNIST.

Download from: 👉 https://nihcc.app.box.com/v/ChestXray-NIHCC or https://www.kaggle.com/datasets/nih-chest-xrays/data

---

### Optional / Future: COCO Unlabeled (~40 GB)

Only needed if you extend the project to object detection. Not part of the current scope.

---

## 5. Project Directory Structure

After setup, your project should look like this:

```
Internship/
├── venv/                          ← virtual environment (created by you)
├── data/                          ← datasets go here
│   ├── chestxray14/               ← (optional full-res)
│   └── coco/                      ← (optional object detection)
├── configs/                       ← YAML config files for experiments
├── src/                           ← all source code
│   ├── models/                    ← model architectures
│   ├── pretext/                   ← rotation, inpainting pretext tasks
│   ├── simclr/                    ← SimCLR pretraining
│   ├── self_training/             ← pseudo-labeling loop
│   ├── active_learning/           ← annotation selection strategies
│   └── utils/                     ← helpers, metrics, visualization
├── notebooks/                     ← Jupyter exploration
├── outputs/                       ← checkpoints, logs, results
├── requirements.txt
├── SETUP_GUIDE.md                 ← this file
└── self_training_loop_stepper.html
```

Create the skeleton:
```bash
mkdir data configs src notebooks outputs
mkdir src\models src\pretext src\simclr src\self_training src\active_learning src\utils
```

---

## 6. RTX 4060 — Memory Tips (8 GB VRAM)

Your RTX 4060 has **8 GB VRAM**. This is enough for everything in this project, but keep these tips in mind:

| Setting | Recommended Value |
|---------|------------------|
| Batch size (supervised) | 32–64 |
| Batch size (SimCLR) | 128–256 (use gradient accumulation if OOM) |
| Image resolution | 224×224 (standard) |
| Backbone | ResNet-18 or ResNet-50 (avoid ViT-Large) |
| Mixed precision | **Always ON** — halves memory usage |

Enable mixed precision in PyTorch:
```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    output = model(images)
    loss = criterion(output, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

If you hit **Out of Memory (OOM)**:
1. Reduce batch size
2. Use `torch.cuda.empty_cache()` between phases
3. Use gradient accumulation (train with effective batch=256, actual batch=64)
4. Use gradient checkpointing: `model.set_grad_checkpointing(enable=True)` (works with timm models)

---

## 7. Quick Sanity Check

After completing all setup, run this to verify everything works:

```python
import torch
import torchvision
import medmnist
import timm
import lightly
import albumentations
import lightning

print("=" * 50)
print(f"PyTorch:      {torch.__version__}")
print(f"Torchvision:  {torchvision.__version__}")
print(f"CUDA:         {torch.version.cuda}")
print(f"cuDNN:        {torch.backends.cudnn.version()}")
print(f"GPU:          {torch.cuda.get_device_name(0)}")
print(f"VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"MedMNIST:     {medmnist.__version__}")
print(f"timm:         {timm.__version__}")
print(f"lightly:      {lightly.__version__}")
print(f"Lightning:    {lightning.__version__}")
print("=" * 50)
print("✅ All good — ready to train!")
```

Save this as `check_setup.py` and run: `python check_setup.py`

---

## Summary of Commands (Copy-Paste Ready)

```bash
# 1. Create and activate venv
python -m venv venv
venv\Scripts\activate

# 2. Install PyTorch with CUDA
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Create project folders
mkdir data configs src notebooks outputs
mkdir src\models src\pretext src\simclr src\self_training src\active_learning src\utils

# 5. Verify
python check_setup.py
```
