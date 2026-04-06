"""
Evaluation metrics for multi-label classification.

ChestMNIST has 14 disease labels — we need per-class and average AUROC,
not simple accuracy.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# The 14 ChestMNIST disease categories (in order)
CHEST_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]


def compute_auroc(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute per-class and mean AUROC for multi-label classification.
    
    Args:
        predictions: Sigmoid probabilities, shape (N, 14)
        targets: Binary ground truth, shape (N, 14)
    
    Returns:
        Dict with per-class AUROC and mean AUROC
    """
    num_classes = targets.shape[1]
    per_class_auroc = {}
    valid_aurocs = []
    
    for i in range(num_classes):
        # Skip classes with only one value in targets (AUROC undefined)
        if len(np.unique(targets[:, i])) < 2:
            per_class_auroc[CHEST_LABELS[i]] = float("nan")
            continue
        
        auc = roc_auc_score(targets[:, i], predictions[:, i])
        per_class_auroc[CHEST_LABELS[i]] = auc
        valid_aurocs.append(auc)
    
    mean_auroc = np.mean(valid_aurocs) if valid_aurocs else 0.0
    
    return {
        "mean_auroc": mean_auroc,
        "per_class_auroc": per_class_auroc,
    }


def compute_multilabel_metrics(predictions: np.ndarray, targets: np.ndarray, 
                                threshold: float = 0.5) -> dict:
    """
    Compute comprehensive multi-label metrics.
    
    Args:
        predictions: Sigmoid probabilities, shape (N, 14)
        targets: Binary ground truth, shape (N, 14)
        threshold: Decision threshold for converting probabilities to binary
    
    Returns:
        Dict with AUROC, mAP, F1, and per-class breakdown
    """
    # AUROC
    auroc_results = compute_auroc(predictions, targets)
    
    # Binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # F1 scores
    f1_macro = f1_score(targets, binary_preds, average="macro", zero_division=0)
    f1_micro = f1_score(targets, binary_preds, average="micro", zero_division=0)
    
    # Mean Average Precision
    try:
        mAP = average_precision_score(targets, predictions, average="macro")
    except ValueError:
        mAP = 0.0
    
    # Exact match accuracy (all 14 labels must match)
    exact_match = np.mean(np.all(binary_preds == targets, axis=1))
    
    return {
        "mean_auroc": auroc_results["mean_auroc"],
        "per_class_auroc": auroc_results["per_class_auroc"],
        "mAP": mAP,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "exact_match_accuracy": exact_match,
    }


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 55)
    print(f"  Mean AUROC:     {metrics['mean_auroc']:.4f}")
    print(f"  mAP:            {metrics['mAP']:.4f}")
    print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):     {metrics['f1_micro']:.4f}")
    print(f"  Exact Match:    {metrics['exact_match_accuracy']:.4f}")
    print("-" * 55)
    print("  Per-class AUROC:")
    for label, auc in metrics["per_class_auroc"].items():
        bar = "█" * int(auc * 20) if not np.isnan(auc) else "N/A"
        print(f"    {label:<22s}  {auc:.4f}  {bar}")
    print("=" * 55)
