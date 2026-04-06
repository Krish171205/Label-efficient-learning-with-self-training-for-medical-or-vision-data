"""
Device setup and configuration utilities.

Handles GPU selection, mixed precision setup, and reproducibility.
"""

import os
import random
import numpy as np
import torch


def setup_device(cuda_visible_devices: str = "0") -> torch.device:
    """
    Configure CUDA device and return the torch device.
    
    Args:
        cuda_visible_devices: Which GPU index to expose to CUDA.
                              "0" for RTX 4060 (only CUDA GPU on this system).
    
    Returns:
        torch.device — either 'cuda' or 'cpu'
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    else:
        device = torch.device("cpu")
        print("⚠ No CUDA GPU found — using CPU (training will be very slow)")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for full reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


def get_amp_context(enabled: bool = True):
    """
    Get the mixed precision autocast context manager.
    
    Args:
        enabled: Whether to use FP16 mixed precision.
    
    Returns:
        torch.amp.autocast context manager
    """
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def get_grad_scaler(enabled: bool = True):
    """
    Get gradient scaler for mixed precision training.
    
    Args:
        enabled: Whether to use gradient scaling.
    
    Returns:
        torch.amp.GradScaler instance
    """
    return torch.amp.GradScaler(enabled=enabled)
