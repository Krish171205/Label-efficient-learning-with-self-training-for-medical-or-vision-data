"""
Reusable training engine.

Contains train_one_epoch() and evaluate() functions used by ALL training
scripts (baseline, SimCLR fine-tune, pretext fine-tune, self-training).
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import get_amp_context, get_grad_scaler


def train_one_epoch(model, dataloader: DataLoader, optimizer, device: torch.device,
                    epoch: int, total_epochs: int, use_amp: bool = True) -> dict:
    """
    Train the model for one epoch.
    
    Args:
        model: ChestClassifier instance
        dataloader: Training DataLoader (labeled data)
        optimizer: Optimizer instance
        device: torch.device
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        use_amp: Whether to use mixed precision
    
    Returns:
        Dict with training metrics (loss)
    """
    model.train()
    amp_context = get_amp_context(enabled=use_amp)
    scaler = get_grad_scaler(enabled=use_amp)
    
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", 
                leave=False, ncols=100)
    
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()
        
        # MedMNIST labels come as (B, 14, 1) or (B, 14) — ensure shape is (B, 14)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
        
        optimizer.zero_grad()
        
        with amp_context:
            logits = model(images)
            loss = model.compute_loss(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = running_loss / max(num_batches, 1)
    return {"train_loss": avg_loss}


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device,
             use_amp: bool = True) -> dict:
    """
    Evaluate model on validation/test data.
    
    Args:
        model: ChestClassifier instance
        dataloader: Validation or test DataLoader
        device: torch.device
        use_amp: Whether to use mixed precision
    
    Returns:
        Dict with:
            - val_loss: Average BCE loss
            - predictions: Sigmoid probabilities, shape (N, 14)
            - targets: Ground truth labels, shape (N, 14)
    """
    model.eval()
    amp_context = get_amp_context(enabled=use_amp)
    
    running_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False, ncols=100):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()
        
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
        
        with amp_context:
            logits = model(images)
            loss = model.compute_loss(logits, targets)
        
        probs = torch.sigmoid(logits)
        
        running_loss += loss.item()
        num_batches += 1
        all_predictions.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    avg_loss = running_loss / max(num_batches, 1)
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return {
        "val_loss": avg_loss,
        "predictions": predictions,
        "targets": targets,
    }


def build_optimizer(model, cfg) -> torch.optim.Optimizer:
    """
    Build optimizer from config.
    
    Args:
        model: PyTorch model
        cfg: Config object with training.optimizer, training.learning_rate, etc.
    
    Returns:
        Optimizer instance
    """
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    
    if cfg.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif cfg.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif cfg.training.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, 
                                     momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")
    
    print(f"✓ Optimizer: {cfg.training.optimizer} (lr={lr}, wd={wd})")
    return optimizer


def build_scheduler(optimizer, cfg, steps_per_epoch: int = None):
    """
    Build learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer instance
        cfg: Config object
        steps_per_epoch: Number of batches per epoch (for OneCycleLR)
    
    Returns:
        Scheduler instance or None
    """
    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=1e-6
        )
        print(f"✓ Scheduler: CosineAnnealing (T_max={cfg.training.epochs})")
        return scheduler
    elif cfg.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.1
        )
        print(f"✓ Scheduler: StepLR (step=15, gamma=0.1)")
        return scheduler
    else:
        print("✓ No scheduler")
        return None


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dict of metrics to save alongside
        path: File path to save to
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(model, path: str, optimizer=None, device: torch.device = None) -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to .pth checkpoint file
        optimizer: Optional optimizer to restore state
        device: Device to map weights to
    
    Returns:
        Dict with saved metrics and epoch
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"✓ Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint.get("metrics", {})
