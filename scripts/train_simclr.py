"""
SimCLR pretraining script.

Trains a ResNet-18 backbone on ALL chest X-ray images (no labels needed)
using contrastive learning. The trained backbone is then used to initialize
the classifier for downstream supervised training.

Usage:
    python train_simclr.py
    python train_simclr.py --epochs 50 --batch_size 64
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightly.loss import NTXentLoss

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed, get_amp_context, get_grad_scaler, wrap_model, unwrap_model
from src.simclr.model import SimCLRModel
from src.simclr.augmentations import get_simclr_transform, KorniaDualViewTransform


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR Pretraining")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ---- Setup ----
    cfg = load_config(args.config)
    device = setup_device(cfg.device.cuda_visible_devices)
    set_seed(cfg.seed)
    
    epochs = args.epochs or cfg.simclr.epochs
    batch_size = args.batch_size or cfg.simclr.batch_size
    
    print(f"\n{'='*60}")
    print(f"  SimCLR CONTRASTIVE PRETRAINING")
    print(f"  Training on ALL unlabeled chest X-rays (no labels used)")
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # ---- Data ----
    # This base transform merely converts to RGB and creates a tensor.
    # We offload the heavy augmentations to Kornia on the GPU.
    base_transform = get_simclr_transform(image_size=cfg.data.image_size)
    
    from src.data.lazy_dataset import LazyChestMNIST
    from torch.utils.data import Subset
    import numpy as np
    
    full_dataset = LazyChestMNIST(
        root=cfg.data.root, split="train",
        transform=base_transform, size=cfg.data.image_size
    )
    
    # Use a random subset for faster pretraining (contrastive learning
    # has diminishing returns on dataset size — batch diversity matters more)
    subset_size = getattr(cfg.simclr, 'pretrain_subset', len(full_dataset))
    if subset_size < len(full_dataset):
        rng = np.random.RandomState(cfg.seed)
        indices = rng.choice(len(full_dataset), size=subset_size, replace=False)
        train_dataset = Subset(full_dataset, indices)
    else:
        train_dataset = full_dataset
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True, persistent_workers=(cfg.data.num_workers > 0),
    )
    
    print(f"✓ Training data: {len(train_dataset)} of {len(full_dataset)} images (no labels needed)")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # ---- Model ----
    model = SimCLRModel(
        backbone_name=cfg.training.backbone,
        projection_dim=cfg.simclr.projection_dim,
    )
    model = model.to(device)
    model = wrap_model(model)
    
    # ---- Kornia Augmentation Module ----
    kornia_transform = KorniaDualViewTransform(image_size=cfg.data.image_size)
    kornia_transform = kornia_transform.to(device)
    kornia_transform = wrap_model(kornia_transform)
    
    # ---- Loss, Optimizer, Scheduler ----
    criterion = NTXentLoss(temperature=cfg.simclr.temperature)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.simclr.learning_rate,
        weight_decay=cfg.simclr.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    print(f"✓ Loss: NT-Xent (temperature={cfg.simclr.temperature})")
    print(f"✓ Optimizer: Adam (lr={cfg.simclr.learning_rate})")
    print(f"✓ Scheduler: CosineAnnealing (T_max={epochs})")
    
    # ---- Training Loop ----
    amp_context = get_amp_context(enabled=cfg.device.mixed_precision)
    scaler = get_grad_scaler(enabled=cfg.device.mixed_precision)
    
    checkpoint_dir = os.path.join(cfg.logging.checkpoint_dir, "simclr")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_loss = float("inf")
    start_epoch = 0
    start_time = time.time()
    
    # ---- Resume from checkpoint ----
    resume_path = os.path.join(checkpoint_dir, "simclr_full.pth")
    if args.resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        print(f"✓ Resumed from epoch {start_epoch} (loss: {best_loss:.4f})")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [SimCLR]",
                    leave=False, ncols=100)
        
        for batch in pbar:
            # Baseline dataloader returns (images, labels)
            images, _ = batch
            
            # Immediately move raw tensors to GPU
            images = images.to(device, non_blocking=True)
            
            # Kornia applies massive parallel data augmentations directly on GPU
            with torch.no_grad():
                view1, view2 = kornia_transform(images)
            
            optimizer.zero_grad()
            
            with amp_context:
                # Get projected embeddings for both views
                z1 = model(view1)  # (B, 128)
                z2 = model(view2)  # (B, 128)
                
                # NT-Xent contrastive loss:
                # Pull z1[i] and z2[i] together (same image)
                # Push z1[i] and z2[j] apart (different images)
                loss = criterion(z1, z2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = running_loss / max(num_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch+1}/{epochs} | SimCLR Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Save the full model (backbone + projection head)
            torch.save({
                "epoch": epoch,
                "model_state_dict": unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, os.path.join(checkpoint_dir, "simclr_full.pth"))
            
            # Save JUST the backbone (this is what we load for classification)
            torch.save({
                "backbone": unwrap_model(model).get_backbone_state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, os.path.join(checkpoint_dir, "simclr_backbone.pth"))
            
            print(f"  💾 Best model saved (loss: {best_loss:.4f})")
    
    elapsed = time.time() - start_time
    
    # ---- Log result ----
    from scripts.training_log import log_result
    log_result(
        script="train_simclr.py", config=args.config,
        backbone="resnet18", epochs_trained=epochs - start_epoch,
        loss=best_loss, data_size=len(train_dataset),
        batch_size=batch_size, time_minutes=elapsed / 60,
        notes=f"SimCLR pretrain {'(resumed)' if start_epoch > 0 else ''}"
    )
    
    print(f"\n{'='*60}")
    print(f"  SimCLR PRETRAINING COMPLETE")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Backbone saved to: {checkpoint_dir}/simclr_backbone.pth")
    print(f"{'='*60}")
    print(f"\nNext step: python scripts/train_simclr_finetune.py")


if __name__ == "__main__":
    main()
