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
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"✓ GPU {i}: {name} ({vram:.1f} GB)")
        if n_gpus > 1:
            print(f"  → DataParallel will use all {n_gpus} GPUs")
        print(f"  CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    else:
        device = torch.device("cpu")
        print("⚠ No CUDA GPU found — using CPU (training will be very slow)")
    
    return device


def wrap_model(model):
    """Wrap model in DataParallel if multiple GPUs are available."""
    if torch.cuda.device_count() > 1:
        print(f"✓ Wrapping model in DataParallel ({torch.cuda.device_count()} GPUs)")
        model = torch.nn.DataParallel(model)
    return model


def unwrap_model(model):
    """Get the underlying model (strips DataParallel wrapper if present)."""
    if hasattr(model, 'module'):
        return model.module
    return model


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
