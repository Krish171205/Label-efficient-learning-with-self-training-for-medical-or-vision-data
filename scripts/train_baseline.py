"""
Baseline: Transfer Learning (Supervised Only)

This is the BASELINE experiment. It answers the question:
"How well can we do with just labeled data + an ImageNet backbone?"

Everything else in the project (SimCLR, pretext, self-training) must beat this.

Usage:
    python train_baseline.py
    python train_baseline.py --label_fraction 0.05
    python train_baseline.py --label_fraction 0.10 --epochs 30
"""

import os
import sys
import argparse
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed, wrap_model
from src.utils.metrics import compute_multilabel_metrics, print_metrics
from src.utils.training import (
    train_one_epoch, evaluate, build_optimizer, 
    build_scheduler, save_checkpoint
)
from src.data.chest_mnist import ChestMNISTDataset
from src.models.classifier import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline: Transfer Learning")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--label_fraction", type=float, default=None,
                        help="Override label fraction (e.g., 0.01, 0.05, 0.10)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ---- Setup ----
    cfg = load_config(args.config)
    device = setup_device(cfg.device.cuda_visible_devices)
    set_seed(cfg.seed)
    
    label_fraction = args.label_fraction or cfg.label_fractions[0]  # Default: 1%
    epochs = args.epochs or cfg.training.epochs
    batch_size = args.batch_size or cfg.training.batch_size
    
    print(f"\n{'='*60}")
    print(f"  BASELINE: Transfer Learning (Supervised Only)")
    print(f"  Label fraction: {label_fraction*100:.0f}%")
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # ---- Data ----
    dataset = ChestMNISTDataset(
        data_root=cfg.data.root,
        image_size=cfg.data.image_size,
        label_fraction=label_fraction,
        seed=cfg.seed,
    )
    
    train_loader = dataset.get_labeled_loader(
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
    )
    val_loader = dataset.get_val_loader(
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
    )
    test_loader = dataset.get_test_loader(
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
    )
    
    # ---- Model ----
    model = build_classifier(cfg)
    model = model.to(device)
    model = wrap_model(model)
    
    # ---- Optimizer & Scheduler ----
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    # ---- Training Loop ----
    best_auroc = 0.0
    patience_counter = 0
    results_history = []
    
    checkpoint_dir = os.path.join(
        cfg.logging.checkpoint_dir, 
        f"baseline_lf{label_fraction}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, epochs, use_amp=cfg.device.mixed_precision
        )
        
        # Validate
        val_results = evaluate(model, val_loader, device, use_amp=cfg.device.mixed_precision)
        val_metrics = compute_multilabel_metrics(
            val_results["predictions"], val_results["targets"]
        )
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val Loss: {val_results['val_loss']:.4f} | "
              f"Val AUROC: {val_metrics['mean_auroc']:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Track history
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_results["val_loss"],
            "val_auroc": val_metrics["mean_auroc"],
            "val_mAP": val_metrics["mAP"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": current_lr,
        }
        results_history.append(epoch_result)
        
        # Save best model
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
        
        # Early stopping
        if patience_counter >= cfg.training.early_stopping_patience:
            print(f"\n⏹ Early stopping after {epoch+1} epochs (no improvement for "
                  f"{cfg.training.early_stopping_patience} epochs)")
            break
    
    elapsed = time.time() - start_time
    print(f"\n⏱ Training completed in {elapsed/60:.1f} minutes")
    
    # ---- Final Test Evaluation ----
    print(f"\n{'='*60}")
    print(f"  FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    # Load best model for test
    from src.utils.training import load_checkpoint
    load_checkpoint(model, os.path.join(checkpoint_dir, "best_model.pth"), device=device)
    
    test_results = evaluate(model, test_loader, device, use_amp=cfg.device.mixed_precision)
    test_metrics = compute_multilabel_metrics(
        test_results["predictions"], test_results["targets"]
    )
    print_metrics(test_metrics)
    
    # ---- Save Results ----
    results_dir = os.path.join(cfg.logging.results_dir, f"baseline_lf{label_fraction}")
    os.makedirs(results_dir, exist_ok=True)
    
    final_results = {
        "experiment": "baseline_transfer_learning",
        "label_fraction": label_fraction,
        "n_labeled": len(dataset.labeled_indices),
        "n_total_train": len(dataset.train_dataset),
        "epochs_trained": len(results_history),
        "best_val_auroc": best_auroc,
        "test_metrics": {
            "mean_auroc": test_metrics["mean_auroc"],
            "mAP": test_metrics["mAP"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_micro": test_metrics["f1_micro"],
            "exact_match": test_metrics["exact_match_accuracy"],
            "per_class_auroc": test_metrics["per_class_auroc"],
        },
        "training_history": results_history,
        "training_time_minutes": elapsed / 60,
    }
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n📊 Results saved to {results_path}")
    
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPLETE")
    print(f"  Label fraction: {label_fraction*100:.0f}% ({len(dataset.labeled_indices)} images)")
    print(f"  Best Val AUROC:  {best_auroc:.4f}")
    print(f"  Test AUROC:      {test_metrics['mean_auroc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

