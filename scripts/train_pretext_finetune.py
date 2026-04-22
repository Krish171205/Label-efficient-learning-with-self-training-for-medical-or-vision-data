"""
Fine-tune a pretext-pretrained backbone on labeled data.

Same structure as train_simclr_finetune.py but loads rotation or inpainting
backbone instead of SimCLR.

Usage:
    python train_pretext_finetune.py --task rotation
    python train_pretext_finetune.py --task inpainting
    python train_pretext_finetune.py --task rotation --label_fraction 0.05
"""

import os
import sys
import argparse
import json
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed, wrap_model, unwrap_model
from src.utils.metrics import compute_multilabel_metrics, print_metrics
from src.utils.training import (
    train_one_epoch, evaluate, build_optimizer,
    build_scheduler, save_checkpoint, load_checkpoint
)
from src.data.chest_mnist import ChestMNISTDataset
from src.models.classifier import ChestClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Pretext Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task", type=str, required=True, choices=["rotation", "inpainting"])
    parser.add_argument("--label_fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--backbone_path", type=str, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = load_config(args.config)
    device = setup_device(cfg.device.cuda_visible_devices)
    set_seed(cfg.seed)
    
    task = args.task
    label_fraction = args.label_fraction or cfg.label_fractions[0]
    epochs = args.epochs or cfg.training.epochs
    freeze_epochs = args.freeze_epochs
    
    # Default backbone path based on task
    if args.backbone_path:
        backbone_path = args.backbone_path
    elif task == "rotation":
        backbone_path = os.path.join(
            cfg.logging.checkpoint_dir, "pretext_rotation", "rotation_backbone.pth"
        )
    else:
        backbone_path = os.path.join(
            cfg.logging.checkpoint_dir, "pretext_inpainting", "inpainting_backbone.pth"
        )
    
    print(f"\n{'='*60}")
    print(f"  PRETEXT ({task.upper()}) FINE-TUNING")
    print(f"  Backbone: {task}-pretrained")
    print(f"  Label fraction: {label_fraction*100:.0f}%")
    print(f"  Strategy: {freeze_epochs} frozen → then full fine-tune")
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
    model = ChestClassifier(
        backbone_name=cfg.training.backbone,
        num_classes=cfg.data.num_classes,
        pretrained_imagenet=False,
    )
    model.load_backbone_weights(backbone_path)
    model = model.to(device)
    model = wrap_model(model)
    
    # ---- Phase 1: Linear Probe ----
    if freeze_epochs > 0:
        print(f"\n--- Phase 1: Linear Probe ({freeze_epochs} epochs, backbone frozen) ---")
        unwrap_model(model).freeze_backbone()
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
    
    # ---- Phase 2: Full Fine-tuning ----
    print(f"\n--- Phase 2: Full Fine-tuning ({epochs} epochs) ---")
    unwrap_model(model).unfreeze_backbone()
    
    # Use a 10x smaller learning rate for the backbone to prevent catastrophic forgetting
    # of the precious pretext features, while keeping the standard LR for the head.
    base_lr = cfg.training.learning_rate
    optimizer = torch.optim.Adam([
        {"params": unwrap_model(model).backbone.parameters(), "lr": base_lr * 0.1},
        {"params": unwrap_model(model).classifier.parameters(), "lr": base_lr}
    ], weight_decay=cfg.training.weight_decay)
    
    scheduler = build_scheduler(optimizer, cfg)
    
    best_auroc = 0.0
    patience_counter = 0
    results_history = []
    start_epoch = 0
    
    exp_name = f"pretext_{task}_lf{label_fraction}"
    checkpoint_dir = os.path.join(cfg.logging.checkpoint_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ---- Resume from checkpoint ----
    resume_path = os.path.join(checkpoint_dir, "latest_model.pth")
    if args.resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auroc = ckpt.get("metrics", {}).get("auroc", 0.0)
        print(f"Resumed pretext finetune from epoch {start_epoch}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, epochs, use_amp=cfg.device.mixed_precision
        )
        
        val_results = evaluate(model, val_loader, device, use_amp=cfg.device.mixed_precision)
        val_metrics = compute_multilabel_metrics(
            val_results["predictions"], val_results["targets"]
        )
        
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val AUROC: {val_metrics['mean_auroc']:.4f} | "
              f"LR: {current_lr:.6f}")
        
        results_history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "val_auroc": val_metrics["mean_auroc"],
        })
        
        if val_metrics["mean_auroc"] > best_auroc:
            best_auroc = val_metrics["mean_auroc"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"auroc": best_auroc},
                os.path.join(checkpoint_dir, "best_model.pth")
            )
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.training.early_stopping_patience:
            print(f"\n⏹ Early stopping after {epoch+1} epochs")
            break
        
        # Unconditional save: protects against crashes
        save_checkpoint(
            model, optimizer, epoch,
            {"auroc": val_metrics["mean_auroc"]},
            os.path.join(checkpoint_dir, "latest_model.pth")
        )
    
    elapsed = time.time() - start_time
    
    # ---- Test ----
    print(f"\n{'='*60}")
    print(f"  FINAL TEST ({task.upper()} Fine-tune)")
    print(f"{'='*60}")
    
    load_checkpoint(model, os.path.join(checkpoint_dir, "best_model.pth"), device=device)
    
    test_results = evaluate(model, test_loader, device, use_amp=cfg.device.mixed_precision)
    test_metrics = compute_multilabel_metrics(
        test_results["predictions"], test_results["targets"]
    )
    print_metrics(test_metrics)
    
    # ---- Save ----
    results_dir = os.path.join(cfg.logging.results_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    final_results = {
        "experiment": f"pretext_{task}_finetune",
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
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  PRETEXT ({task.upper()}) FINE-TUNE COMPLETE")
    print(f"  Label fraction: {label_fraction*100:.0f}%")
    print(f"  Best Val AUROC:  {best_auroc:.4f}")
    print(f"  Test AUROC:      {test_metrics['mean_auroc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

