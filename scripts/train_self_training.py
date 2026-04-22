"""
Self-training loop — the CORE algorithm of the project.

Implements the iterative pseudo-labeling pipeline:
    Round 0: Train on labeled seed (1% of data)
    Round 1: Predict on unlabeled → threshold → add pseudo-labels → retrain
    Round 2: Predict on remaining unlabeled → threshold → retrain
    ...repeat until convergence

Usage:
    python train_self_training.py
    python train_self_training.py --backbone simclr
    python train_self_training.py --backbone rotation
    python train_self_training.py --label_fraction 0.05 --rounds 3
"""

import os
import sys
import argparse
import json
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed, wrap_model, get_amp_context, get_grad_scaler, unwrap_model
from src.utils.metrics import compute_multilabel_metrics, print_metrics
from src.utils.training import (
    train_one_epoch, evaluate, build_optimizer,
    build_scheduler, save_checkpoint, load_checkpoint
)
from src.data.chest_mnist import ChestMNISTDataset
from src.models.classifier import ChestClassifier
from src.self_training.pseudo_labels import (
    generate_pseudo_labels, compute_entropy_loss
)


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Training Loop")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--backbone", type=str, default="imagenet",
                        choices=["imagenet", "simclr", "rotation", "inpainting"],
                        help="Which pretrained backbone to start from")
    parser.add_argument("--label_fraction", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None,
                        help="Number of self-training rounds")
    parser.add_argument("--epochs_per_round", type=int, default=20,
                        help="Training epochs per round")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed round")
    return parser.parse_args()


def get_backbone_path(backbone_type: str, cfg) -> str:
    """Get the path to pretrained backbone checkpoint."""
    ckpt_dir = cfg.logging.checkpoint_dir
    paths = {
        "simclr": os.path.join(ckpt_dir, "simclr", "simclr_backbone.pth"),
        "rotation": os.path.join(ckpt_dir, "pretext_rotation", "rotation_backbone.pth"),
        "inpainting": os.path.join(ckpt_dir, "pretext_inpainting", "inpainting_backbone.pth"),
    }
    return paths.get(backbone_type, None)


def train_one_round(model, train_loader, val_loader, optimizer, scheduler,
                    device, epochs, cfg, round_num) -> dict:
    """
    Train the model for one round of self-training.
    
    Returns dict with best AUROC and training history for this round.
    """
    best_auroc = 0.0
    best_state = None
    history = []
    
    for epoch in range(epochs):
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
        
        lr = optimizer.param_groups[0]["lr"]
        
        print(f"  R{round_num} Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_metrics['train_loss']:.4f} | "
              f"Val AUROC: {val_metrics['mean_auroc']:.4f} | "
              f"LR: {lr:.6f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "val_auroc": val_metrics["mean_auroc"],
        })
        
        if val_metrics["mean_auroc"] > best_auroc:
            best_auroc = val_metrics["mean_auroc"]
            best_state = copy.deepcopy(unwrap_model(model).state_dict())
    
    # Restore best model from this round
    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)
    
    return {"best_auroc": best_auroc, "history": history}


def main():
    args = parse_args()
    
    cfg = load_config(args.config)
    device = setup_device(cfg.device.cuda_visible_devices)
    set_seed(cfg.seed)
    
    label_fraction = args.label_fraction or cfg.label_fractions[0]
    num_rounds = args.rounds or cfg.self_training.num_rounds
    epochs_per_round = args.epochs_per_round
    backbone_type = args.backbone
    
    print(f"\n{'='*60}")
    print(f"  SELF-TRAINING LOOP")
    print(f"  Backbone init: {backbone_type}")
    print(f"  Label fraction: {label_fraction*100:.0f}%")
    print(f"  Rounds: {num_rounds}  |  Epochs/round: {epochs_per_round}")
    print(f"  Threshold: τ={cfg.self_training.confidence_threshold}")
    print(f"  Adaptive thresholds: {cfg.self_training.adaptive_threshold}")
    print(f"{'='*60}\n")
    
    # ---- Data ----
    dataset = ChestMNISTDataset(
        data_root=cfg.data.root,
        image_size=cfg.data.image_size,
        label_fraction=label_fraction,
        seed=cfg.seed,
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
    use_imagenet = (backbone_type == "imagenet")
    model = ChestClassifier(
        backbone_name=cfg.training.backbone,
        num_classes=cfg.data.num_classes,
        pretrained_imagenet=use_imagenet,
    )
    
    # Load self-supervised backbone if specified
    backbone_path = get_backbone_path(backbone_type, cfg)
    if backbone_path and os.path.exists(backbone_path):
        model.load_backbone_weights(backbone_path)
    elif backbone_type != "imagenet":
        print(f"⚠ Backbone {backbone_path} not found — using random init")
    
    model = model.to(device)
    model = wrap_model(model)
    
    # ---- Self-Training Loop ----
    exp_name = f"self_training_{backbone_type}_lf{label_fraction}"
    checkpoint_dir = os.path.join(cfg.logging.checkpoint_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_round_results = []
    start_time = time.time()
    start_round = 0
    
    # ---- Resume from last completed round ----
    if args.resume:
        for r in range(num_rounds - 1, -1, -1):
            rpath = os.path.join(checkpoint_dir, f"round_{r}.pth")
            if os.path.exists(rpath):
                ckpt = torch.load(rpath, map_location=device, weights_only=False)
                unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
                start_round = r + 1
                print(f"Resumed self-training from after round {r} (AUROC: {ckpt.get('metrics',{}).get('auroc','?')})")
                break
    
    for round_num in range(start_round, num_rounds):
        print(f"\n{'─'*60}")
        print(f"  ROUND {round_num}/{num_rounds-1}")
        print(f"  Labeled pool: {len(dataset.labeled_indices)} images")
        print(f"  Unlabeled pool: {len(dataset.unlabeled_indices)} images")
        print(f"{'─'*60}")
        
        # ---- Train on current labeled set ----
        train_loader = dataset.get_labeled_loader(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
        )
        
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        
        round_result = train_one_round(
            model, train_loader, val_loader, optimizer, scheduler,
            device, epochs_per_round, cfg, round_num
        )
        
        print(f"\n  Round {round_num} best Val AUROC: {round_result['best_auroc']:.4f}")
        
        # ---- Generate pseudo-labels on unlabeled data ----
        if round_num < num_rounds - 1 and len(dataset.unlabeled_indices) > 0:
            unlabeled_loader = dataset.get_unlabeled_loader(
                batch_size=cfg.training.batch_size,
                num_workers=cfg.data.num_workers,
            )
            
            pseudo_result = generate_pseudo_labels(
                model, unlabeled_loader, device,
                threshold=cfg.self_training.confidence_threshold,
                adaptive=cfg.self_training.adaptive_threshold,
                use_amp=cfg.device.mixed_precision,
            )
            
            # Map back to original dataset indices
            unlabeled_indices = dataset.unlabeled_indices.copy()
            original_indices = [unlabeled_indices[i] for i in pseudo_result["pseudo_indices"]]
            
            if len(original_indices) > 0:
                dataset.add_pseudo_labeled(original_indices, pseudo_result["pseudo_labels"])
            else:
                print("  ⚠ No pseudo-labels generated this round — stopping early")
                break
            
            round_result["pseudo_stats"] = pseudo_result["stats"]
        
        # Save round checkpoint
        save_checkpoint(
            model, optimizer, round_num,
            {"auroc": round_result["best_auroc"], "round": round_num},
            os.path.join(checkpoint_dir, f"round_{round_num}.pth")
        )
        
        all_round_results.append({
            "round": round_num,
            "labeled_pool_size": len(dataset.labeled_indices),
            "unlabeled_pool_size": len(dataset.unlabeled_indices),
            "best_val_auroc": round_result["best_auroc"],
            "history": round_result["history"],
            "pseudo_stats": round_result.get("pseudo_stats", None),
        })
    
    elapsed = time.time() - start_time
    
    # ---- Final Test ----
    print(f"\n{'='*60}")
    print(f"  FINAL TEST EVALUATION (Self-Training)")
    print(f"{'='*60}")
    
    test_results = evaluate(model, test_loader, device, use_amp=cfg.device.mixed_precision)
    test_metrics = compute_multilabel_metrics(
        test_results["predictions"], test_results["targets"]
    )
    print_metrics(test_metrics)
    
    # ---- Save Results ----
    results_dir = os.path.join(cfg.logging.results_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    final_results = {
        "experiment": "self_training",
        "backbone_init": backbone_type,
        "label_fraction": label_fraction,
        "initial_labeled": int(len(dataset.train_dataset) * label_fraction),
        "final_labeled": len(dataset.labeled_indices),
        "num_rounds": len(all_round_results),
        "epochs_per_round": epochs_per_round,
        "threshold": cfg.self_training.confidence_threshold,
        "adaptive_threshold": cfg.self_training.adaptive_threshold,
        "test_metrics": {
            "mean_auroc": test_metrics["mean_auroc"],
            "mAP": test_metrics["mAP"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_micro": test_metrics["f1_micro"],
            "per_class_auroc": test_metrics["per_class_auroc"],
        },
        "round_results": all_round_results,
        "training_time_minutes": elapsed / 60,
    }
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SELF-TRAINING COMPLETE")
    print(f"  Backbone: {backbone_type}")
    print(f"  Rounds: {len(all_round_results)}")
    print(f"  Labels: {int(len(dataset.train_dataset) * label_fraction)} → {len(dataset.labeled_indices)}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  ")
    print(f"  AUROC progression:")
    for r in all_round_results:
        bar = "█" * int(r['best_val_auroc'] * 30)
        print(f"    Round {r['round']}: {r['best_val_auroc']:.4f}  {bar}  "
              f"(pool: {r['labeled_pool_size']})")
    print(f"  ")
    print(f"  Test AUROC: {test_metrics['mean_auroc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

