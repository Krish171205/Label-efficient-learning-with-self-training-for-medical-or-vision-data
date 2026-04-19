# 🚀 Kaggle GPU Guide — Step by Step

> Kaggle free tier: T4 GPU (16 GB VRAM), 30 hrs/week, 12-hour sessions.
> Your entire pipeline fits in one session — no resume needed.

---

## Step 1: Create a Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com/) → Sign in
2. Click **"Create"** → **"New Notebook"**
3. On the right sidebar:
   - **Accelerator**: Select **GPU T4 x2**
   - **Internet**: Toggle **ON** (needed to clone repo + download dataset)
4. Click **Save & Run All** settings icon → set **Persistence**: **Files only**

---

## Step 2: Clone Repo + Install

```python
# Cell 1: Clone from GitHub
!git clone https://github.com/Krish171205/Label-efficient-learning-with-self-training-for-medical-or-vision-data.git
%cd Label-efficient-learning-with-self-training-for-medical-or-vision-data
```

```python
# Cell 2: Install dependencies
!pip install -r requirements.txt -q
```

```python
# Cell 3: Verify GPU
!nvidia-smi
# Should show: Tesla T4, 16 GB VRAM
```

---

## Step 3: Download Dataset

```python
# Cell 4: Download + convert (~5 min)
!python scripts/dataset_download.py
!python scripts/convert_to_npy.py
```

---

## Step 4: Run Full Pipeline

All commands use `--config configs/colab.yaml` (Kaggle has same T4 GPU specs).

```python
# Cell 5: Baseline (~15 min)
!python scripts/train_baseline.py --config configs/colab.yaml
```

```python
# Cell 6: SimCLR pretraining — 50 epochs, all 78K images (~3-4 hours)
!python scripts/train_simclr.py --config configs/colab.yaml
```

```python
# Cell 7: SimCLR fine-tune (~20 min)
!python scripts/train_simclr_finetune.py --config configs/colab.yaml
```

```python
# Cell 8: Rotation pretraining — 30 epochs, all 78K images (~2 hours)
!python scripts/train_pretext.py --task rotation --config configs/colab.yaml
```

```python
# Cell 9: Rotation fine-tune (~20 min)
!python scripts/train_pretext_finetune.py --task rotation --config configs/colab.yaml
```

```python
# Cell 10: Self-training — ImageNet backbone (~1 hour)
!python scripts/train_self_training.py --backbone imagenet --config configs/colab.yaml
```

```python
# Cell 11: Self-training — SimCLR backbone (~1 hour)
!python scripts/train_self_training.py --backbone simclr --config configs/colab.yaml
```

```python
# Cell 12: Final comparison + plots
!python scripts/run_comparison.py --config configs/colab.yaml
```

```python
# Cell 13: View training log
!python scripts/training_log.py
```

**Total time: ~8-9 hours — fits in one 12-hour Kaggle session!**

---

## Step 5: Download Results

After training completes, download your results:

```python
# Cell 14: View comparison
import json
with open("outputs/results/comparison_summary.json") as f:
    print(json.dumps(json.load(f), indent=2))
```

```python
# Cell 15: Display plots
from IPython.display import Image, display
display(Image("outputs/results/plots/auroc_comparison.png"))
display(Image("outputs/results/plots/self-train_imagenet_progression.png"))
display(Image("outputs/results/plots/self-train_simclr_progression.png"))
```

To download files locally:
- Click the **folder icon** (📁) on the left sidebar
- Navigate to `outputs/results/`
- Right-click any file → **Download**

Or zip everything:
```python
# Cell 16: Zip results for download
!zip -r results.zip outputs/results/ outputs/training_log.csv
# Then download results.zip from the file browser
```

---

## Kaggle vs Local Comparison

| Setting | Local (RTX 4060) | Kaggle (T4) |
|---------|-----------------|-------------|
| VRAM | 8 GB | **16 GB** |
| Workers | 4 | **2** |
| Batch size | 64 | **128** |
| SimCLR epochs | 20 | **50** |
| SimCLR data | 40K subset | **All 78K** |
| Rotation epochs | 15 | **30** |
| Rotation data | 40K subset | **All 78K** |
| Session limit | Unlimited | 12 hours |
| Weekly GPU quota | Unlimited | 30 hours |

---

## If Session Disconnects (unlikely — 12hr limit)

Kaggle preserves output files. If you need to resume:

```python
# Re-clone and install
!git clone https://github.com/Krish171205/Label-efficient-learning-with-self-training-for-medical-or-vision-data.git
%cd Label-efficient-learning-with-self-training-for-medical-or-vision-data
!pip install -r requirements.txt -q
!python scripts/dataset_download.py
!python scripts/convert_to_npy.py

# Resume from checkpoint
!python scripts/train_simclr.py --config configs/colab.yaml --resume
```

---

## Tips

- **Don't close the browser tab** — Kaggle may timeout idle sessions after ~60 min
- **Keep a cell running** to prevent idle timeout (e.g., training cell)
- **30 hrs/week resets** every Saturday — plan your sessions accordingly
- **Save the notebook** regularly — your code cells persist even if the session ends
