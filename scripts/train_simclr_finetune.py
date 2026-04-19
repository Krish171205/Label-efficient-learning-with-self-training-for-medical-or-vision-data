"""
Fine-tune a SimCLR-pretrained backbone on labeled data.

This script:
1. Loads the backbone weights from SimCLR pretraining (no ImageNet)
2. Attaches a fresh classification head (14 outputs)
3. Fine-tunes on the small labeled pool (1% by default)
4. Compares against the baseline (ImageNet transfer learning)

Usage:
    python train_simclr_finetune.py
    python train_simclr_finetune.py --label_fraction 0.05
"""

import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed, wrap_model
from src.utils.metrics import compute_multilabel_metrics, print_metrics
from src.utils.training import (
    train_one_epoch, evaluate, build_optimizer,
    build_scheduler, save_checkpoint, load_checkpoint
)
from src.data.chest_mnist import ChestMNISTDataset
from src.models.classifier import ChestClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--label_fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--backbone_path", type=str, default=None,
                        help="Path to SimCLR backbone checkpoint")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Number of epochs to freeze backbone (linear probe) before unfreezing")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ---- Setup ----
    cfg = load_config(args.config)
    device = setup_device(cfg.device.cuda_visible_devices)
    set_seed(cfg.seed)
    
    label_fraction = args.label_fraction or cfg.label_fractions[0]
    epochs = args.epochs or cfg.training.epochs
    freeze_epochs = args.freeze_epochs
    
    backbone_path = args.backbone_path or os.path.join(
        cfg.logging.checkpoint_dir, "simclr", "simclr_backbone.pth"
    )
    
    print(f"\n{'='*60}")
    print(f"  SimCLR FINE-TUNING")
    print(f"  Backbone: SimCLR-pretrained (from {backbone_path})")
    print(f"  Label fraction: {label_fraction*100:.0f}%")
    print(f"  Strategy: {freeze_epochs} frozen epochs → then full fine-tune")
    print(f"  Epochs: {epochs}  |  Total: {freeze_epochs + epochs}")
    print(f"{'='*60}\n")
    
    # ---- Data ----
    dataset = ChestMNISTDataset(
        data_root=cfg.data.root,
        image_size=cfg.data.image_size,
        label_fraction=label_fraction,
        seed=cfg.seed,
    )
    
    train_loader = dataset.get_labeled_loader(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )
    val_loader = dataset.get_val_loader(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )
    test_loader = dataset.get_test_loader(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )
    
    # ---- Model ----
    # Create classifier with NO ImageNet weights (we'll load SimCLR instead)
    model = ChestClassifier(
        backbone_name=cfg.training.backbone,
        num_classes=cfg.data.num_classes,
        pretrained_imagenet=False,  # Don't use ImageNet — use SimCLR
    )
    
    # Load SimCLR backbone
    model.load_backbone_weights(backbone_path)
    model = model.to(device)
    model = wrap_model(model)
    
    # ---- Phase 1: Linear Probing (frozen backbone) ----
    if freeze_epochs > 0:
        print(f"\n--- Phase 1: Linear Probe ({freeze_epochs} epochs, backbone frozen) ---")
        model.freeze_backbone()
        
        optimizer = build_optimizer(model, cfg)
        
        for epoch in range(freeze_epochs):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device,
                epoch, freeze_epochs, use_amp=cfg.device.mixed_precision
            )
            val_results = evaluate(model, val_loader, device, use_amp=cfg.device.mixed_precision)
            val_metrics = compute_multilabel_metrics(
                val_results["predictions"], val_results["targets"]
            )
            print(f"[Frozen] Epoch {epoch+1}/{freeze_epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val AUROC: {val_metrics['mean_auroc']:.4f}")
    
    # ---- Phase 2: Full Fine-tuning (unfrozen) ----
    print(f"\n--- Phase 2: Full Fine-tuning ({epochs} epochs, backbone unfrozen) ---")
    model.unfreeze_backbone()
    
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    best_auroc = 0.0
    patience_counter = 0
    results_history = []
    
    checkpoint_dir = os.path.join(
        cfg.logging.checkpoint_dir,
        f"simclr_finetune_lf{label_fraction}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, epochs, use_amp=cfg.device.mixed_precision
        )
        
        val_results = evaluate(model, val_loader, device, use_amp=cfg.device.mixed_precision)
        val_metrics = compute_multilabel_metrics(
            val_results["predictions"], val_results["targets"]
        )
        
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val Loss: {val_results['val_loss']:.4f} | "
              f"Val AUROC: {val_metrics['mean_auroc']:.4f} | "
              f"LR: {current_lr:.6f}")
        
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_results["val_loss"],
            "val_auroc": val_metrics["mean_auroc"],
            "val_mAP": val_metrics["mAP"],
            "lr": current_lr,
        }
        results_history.append(epoch_result)
        
        if val_metrics["mean_auroc"] > best_auroc:
            best_auroc = val_metrics["mean_auroc"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"auroc": best_auroc, **val_metrics},
                os.path.join(checkpoint_dir, "best_model.pth")
            )
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.training.early_stopping_patience:
            print(f"\n⏹ Early stopping after {epoch+1} epochs")
            break
    
    elapsed = time.time() - start_time
    
    # ---- Test Evaluation ----
    print(f"\n{'='*60}")
    print(f"  FINAL TEST EVALUATION (SimCLR Fine-tune)")
    print(f"{'='*60}")
    
    load_checkpoint(model, os.path.join(checkpoint_dir, "best_model.pth"), device=device)
    
    test_results = evaluate(model, test_loader, device, use_amp=cfg.device.mixed_precision)
    test_metrics = compute_multilabel_metrics(
        test_results["predictions"], test_results["targets"]
    )
    print_metrics(test_metrics)
    
    # ---- Save Results ----
    results_dir = os.path.join(cfg.logging.results_dir, f"simclr_finetune_lf{label_fraction}")
    os.makedirs(results_dir, exist_ok=True)
    
    final_results = {
        "experiment": "simclr_finetune",
        "label_fraction": label_fraction,
        "n_labeled": len(dataset.labeled_indices),
        "freeze_epochs": freeze_epochs,
        "best_val_auroc": best_auroc,
        "test_metrics": {
            "mean_auroc": test_metrics["mean_auroc"],
            "mAP": test_metrics["mAP"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_micro": test_metrics["f1_micro"],
            "per_class_auroc": test_metrics["per_class_auroc"],
        },
        "training_history": results_history,
        "training_time_minutes": elapsed / 60,
    }
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  SimCLR FINE-TUNE COMPLETE")
    print(f"  Label fraction: {label_fraction*100:.0f}% ({len(dataset.labeled_indices)} images)")
    print(f"  Best Val AUROC:  {best_auroc:.4f}")
    print(f"  Test AUROC:      {test_metrics['mean_auroc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

