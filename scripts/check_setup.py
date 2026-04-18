import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 4060 (only CUDA GPU)

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
print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"MedMNIST:     {medmnist.__version__}")
print(f"timm:         {timm.__version__}")
print(f"lightly:      {lightly.__version__}")
print(f"Lightning:    {lightning.__version__}")
print("=" * 50)
print("✅ All good — ready to train!")
