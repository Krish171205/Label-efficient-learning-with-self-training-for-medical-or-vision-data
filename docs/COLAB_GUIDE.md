# 🚀 Google Colab Guide — Step by Step

## What to Upload

**You DON'T upload files manually.** Clone from GitHub:

```python
# Cell 1: Clone repo
!git clone https://github.com/Krish171205/Label-efficient-learning-with-self-training-for-medical-or-vision-data.git
%cd Label-efficient-learning-with-self-training-for-medical-or-vision-data
```

---

## Colab Settings

1. Go to **Runtime → Change runtime type**
2. Set **Hardware accelerator: T4 GPU**
3. Click **Save**

Verify GPU:
```python
# Cell 2: Check GPU
!nvidia-smi
```
You should see: **Tesla T4, 16 GB VRAM**

---

## Install Dependencies

```python
# Cell 3: Install
!pip install -r requirements.txt -q
```

---

## Download Dataset

```python
# Cell 4: Download + convert
!python scripts/dataset_download.py
!python scripts/convert_to_npy.py
```

---

## Run Order (use `--config configs/colab.yaml` for every command)

### Session 1: SimCLR Pretraining (~3-4 hours)
```python
# Cell 5: SimCLR pretrain (50 epochs, all 78K images, batch 128)
!python scripts/train_simclr.py --config configs/colab.yaml
```

**If session disconnects**, reconnect and resume:
```python
# Re-clone, reinstall, re-download data, then:
!python scripts/train_simclr.py --config configs/colab.yaml --resume
```

### Session 2: Fine-tuning + Rotation (~2 hours)
```python
# Cell 6: SimCLR fine-tune
!python scripts/train_simclr_finetune.py --config configs/colab.yaml

# Cell 7: Rotation pretraining (30 epochs, all 78K)
!python scripts/train_pretext.py --task rotation --config configs/colab.yaml

# Cell 8: Rotation fine-tune
!python scripts/train_pretext_finetune.py --task rotation --config configs/colab.yaml
```

### Session 3: Self-Training (~2 hours)
```python
# Cell 9: Baseline
!python scripts/train_baseline.py --config configs/colab.yaml

# Cell 10: Self-training (ImageNet backbone)
!python scripts/train_self_training.py --backbone imagenet --config configs/colab.yaml

# Cell 11: Self-training (SimCLR backbone)
!python scripts/train_self_training.py --backbone simclr --config configs/colab.yaml

# Cell 12: Final comparison
!python scripts/run_comparison.py --config configs/colab.yaml
```

### Check Progress Anytime:
```python
# Cell: View training log
!python scripts/training_log.py
```

---

## Colab vs Local Comparison

| Setting | Local (RTX 4060) | Colab (T4) |
|---------|-----------------|------------|
| VRAM | 8 GB | **16 GB** |
| Workers | 4 | 2 |
| Batch size | 64 | **128** |
| SimCLR epochs | 20 | **50** |
| SimCLR data | 40K subset | **All 78K** |
| Rotation epochs | 15 | **30** |
| Rotation data | 40K subset | **All 78K** |
| Session limit | Unlimited | ~4 hours |

---

## Save Results Before Session Ends

Colab deletes files when the session ends! Save to Google Drive:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints + results to Drive
!cp -r outputs/ /content/drive/MyDrive/internship_outputs/
!cp outputs/training_log.csv /content/drive/MyDrive/internship_outputs/
```

To restore in a new session:
```python
# Copy back from Drive
!cp -r /content/drive/MyDrive/internship_outputs/ outputs/
```

---

## If Session Disconnects Mid-Training

1. Reconnect to Colab
2. Set runtime to T4 GPU again
3. Re-clone repo + install deps + download data
4. Copy checkpoints back from Drive
5. Run with `--resume` flag

```python
# Full recovery sequence:
!git clone https://github.com/Krish171205/Label-efficient-learning-with-self-training-for-medical-or-vision-data.git
%cd Label-efficient-learning-with-self-training-for-medical-or-vision-data
!pip install -r requirements.txt -q
!python scripts/dataset_download.py
!python scripts/convert_to_npy.py

# Restore checkpoints from Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/internship_outputs/ outputs/

# Resume training
!python scripts/train_simclr.py --config configs/colab.yaml --resume
```
