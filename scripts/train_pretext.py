"""
Pretext task pretraining: Rotation prediction OR Inpainting.

Trains a ResNet-18 backbone on ALL chest X-ray images using a pretext task
(no disease labels needed). The task provides FREE supervisory signal.

Usage:
    python train_pretext.py --task rotation
    python train_pretext.py --task inpainting
    python train_pretext.py --task rotation --epochs 30
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed, wrap_model, get_amp_context, get_grad_scaler, unwrap_model
from src.pretext.rotation import RotationModel, RotationDataset
from src.pretext.inpainting import InpaintingModel, InpaintingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Pretext Task Pretraining")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task", type=str, required=True, choices=["rotation", "inpainting"],
                        help="Which pretext task to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def train_rotation(cfg, device, epochs, batch_size):
    """Train rotation prediction pretext task."""
    
    # Load raw dataset (no transforms — RotationDataset handles it)
    from src.data.lazy_dataset import LazyChestMNIST
    from torch.utils.data import Subset
    import numpy as np
    
    raw_dataset = LazyChestMNIST(
        root=cfg.data.root, split="train",
        transform=None, size=cfg.data.image_size
    )
    
    # Use subset for faster pretraining
    subset_size = getattr(cfg.pretext.rotation, 'pretrain_subset', len(raw_dataset))
    if subset_size < len(raw_dataset):
        rng = np.random.RandomState(cfg.seed)
        indices = rng.choice(len(raw_dataset), size=subset_size, replace=False)
        raw_dataset = Subset(raw_dataset, indices)
    
    rotation_dataset = RotationDataset(raw_dataset, image_size=cfg.data.image_size)
    train_loader = DataLoader(
        rotation_dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True, persistent_workers=(cfg.data.num_workers > 0),
    )
    
    print(f"✓ Rotation dataset: {len(rotation_dataset)} images")
    
    # Model
    model = RotationModel(backbone_name=cfg.training.backbone)
    model = model.to(device)
    model = wrap_model(model)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pretext.rotation.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    amp_context = get_amp_context(enabled=cfg.device.mixed_precision)
    scaler = get_grad_scaler(enabled=cfg.device.mixed_precision)
    
    checkpoint_dir = os.path.join(cfg.logging.checkpoint_dir, "pretext_rotation")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_loss = float("inf")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Rotation]",
                    leave=False, ncols=100)
        
        for images, rot_labels in pbar:
            images = images.to(device, non_blocking=True)
            rot_labels = rot_labels.to(device, non_blocking=True).long()
            
            optimizer.zero_grad()
            
            with amp_context:
                logits = model(images)
                loss = unwrap_model(model).compute_loss(logits, rot_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == rot_labels).sum().item()
            total += rot_labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Rotation Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_acc = acc
            torch.save({
                "backbone": unwrap_model(model).get_backbone_state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "accuracy": best_acc,
            }, os.path.join(checkpoint_dir, "rotation_backbone.pth"))
            print(f"  💾 Best model saved (loss: {best_loss:.4f}, acc: {best_acc:.4f})")
        
        # Unconditional save: protects against crashes
        torch.save({
            "backbone": unwrap_model(model).get_backbone_state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": acc,
        }, os.path.join(checkpoint_dir, "rotation_latest.pth"))
    
    return best_loss, best_acc, checkpoint_dir


def train_inpainting(cfg, device, epochs, batch_size):
    """Train inpainting pretext task."""
    
    from src.data.lazy_dataset import LazyChestMNIST
    raw_dataset = LazyChestMNIST(
        root=cfg.data.root, split="train",
        transform=None, size=cfg.data.image_size
    )
    
    inpainting_dataset = InpaintingDataset(
        raw_dataset, image_size=cfg.data.image_size,
        mask_size=cfg.pretext.inpainting.mask_size,
    )
    train_loader = DataLoader(
        inpainting_dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    
    print(f"✓ Inpainting dataset: {len(inpainting_dataset)} images")
    print(f"  Mask size: {cfg.pretext.inpainting.mask_size}x{cfg.pretext.inpainting.mask_size}")
    
    # Model
    model = InpaintingModel(
        backbone_name=cfg.training.backbone,
        image_size=cfg.data.image_size,
    )
    model = model.to(device)
    model = wrap_model(model)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pretext.inpainting.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    amp_context = get_amp_context(enabled=cfg.device.mixed_precision)
    scaler = get_grad_scaler(enabled=cfg.device.mixed_precision)
    
    checkpoint_dir = os.path.join(cfg.logging.checkpoint_dir, "pretext_inpainting")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Inpainting]",
                    leave=False, ncols=100)
        
        for masked_input, target, mask in pbar:
            masked_input = masked_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with amp_context:
                reconstructed = model(masked_input)
                loss = unwrap_model(model).compute_loss(reconstructed, target, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Recon Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "backbone": unwrap_model(model).get_backbone_state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, os.path.join(checkpoint_dir, "inpainting_backbone.pth"))
            print(f"  💾 Best model saved (loss: {best_loss:.4f})")
    
    return best_loss, 0.0, checkpoint_dir


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = setup_device(cfg.device.cuda_visible_devices)
    set_seed(cfg.seed)
    
    task = args.task
    
    if task == "rotation":
        epochs = args.epochs or cfg.pretext.rotation.epochs
        batch_size = args.batch_size or cfg.pretext.rotation.batch_size
    else:
        epochs = args.epochs or cfg.pretext.inpainting.epochs
        batch_size = args.batch_size or cfg.pretext.inpainting.batch_size
    
    print(f"\n{'='*60}")
    print(f"  PRETEXT TASK: {task.upper()}")
    print(f"  Training on ALL chest X-rays (no labels needed)")
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    if task == "rotation":
        best_loss, best_acc, ckpt_dir = train_rotation(cfg, device, epochs, batch_size)
    else:
        best_loss, best_acc, ckpt_dir = train_inpainting(cfg, device, epochs, batch_size)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  PRETEXT ({task.upper()}) COMPLETE")
    print(f"  Best loss: {best_loss:.4f}")
    if task == "rotation":
        print(f"  Best rotation accuracy: {best_acc:.4f}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Backbone saved to: {ckpt_dir}/")
    print(f"{'='*60}")
    print(f"\nNext step: Fine-tune for classification:")
    print(f"  python train_pretext_finetune.py --task {task}")


if __name__ == "__main__":
    main()

