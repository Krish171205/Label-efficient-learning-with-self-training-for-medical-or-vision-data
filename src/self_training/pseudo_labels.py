"""
Pseudo-labeling engine for the self-training loop.

This module handles the critical steps:
1. Run model on unlabeled data → get confidence scores
2. Apply threshold τ (fixed or adaptive per-class)
3. Select pseudo-labels that pass the threshold
4. Optionally compute entropy minimization loss

This is where the magic happens — high-confidence predictions become
training data for the next round.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import get_amp_context, unwrap_model
from src.utils.metrics import CHEST_LABELS


@torch.no_grad()
def generate_pseudo_labels(model, unlabeled_loader: DataLoader, device: torch.device,
                           threshold: float = 0.95, adaptive: bool = True,
                           use_amp: bool = True) -> dict:
    """
    Generate pseudo-labels for unlabeled data.
    
    For each unlabeled image, run the model and check if any class
    prediction exceeds the confidence threshold. If yes, assign that
    prediction as a pseudo-label.
    
    Args:
        model: Trained ChestClassifier
        unlabeled_loader: DataLoader for unlabeled pool
        device: torch device
        threshold: Base confidence threshold τ (default 0.95)
        adaptive: If True, use per-class adaptive thresholds
        use_amp: Mixed precision
    
    Returns:
        Dict with:
            - pseudo_indices: list of indices that passed threshold
            - pseudo_labels: numpy array of pseudo-labels (N, 14)
            - pseudo_masks: numpy array of which classes were confident (N, 14)
            - confidences: numpy array of sigmoid probabilities (N, 14)
            - stats: dict with statistics about this round
    """
    model.eval()
    amp_context = get_amp_context(enabled=use_amp)
    
    all_probs = []
    all_indices = []
    
    for batch_idx, (images, _) in enumerate(tqdm(unlabeled_loader, 
                                                   desc="Generating pseudo-labels",
                                                   leave=False, ncols=100)):
        images = images.to(device, non_blocking=True)
        
        with amp_context:
            probs = unwrap_model(model).predict_proba(images)  # (B, 14), sigmoid probabilities
        
        all_probs.append(probs.cpu().numpy())
        
        # Track original dataset indices
        start_idx = batch_idx * unlabeled_loader.batch_size
        batch_indices = list(range(start_idx, start_idx + images.size(0)))
        all_indices.extend(batch_indices)
    
    all_probs = np.concatenate(all_probs, axis=0)  # (N_unlabeled, 14)
    
    # ---- Compute thresholds ----
    if adaptive:
        thresholds = compute_adaptive_thresholds(all_probs, base_threshold=threshold)
    else:
        thresholds = np.full(14, threshold)
    
    # ---- Apply thresholds per class ----
    # Use PERCENTILE-BASED thresholds that adapt to the model's actual confidence.
    # With only 784 labeled images, absolute thresholds (0.95) are too strict —
    # the model is too weak to be that confident about anything.
    # Instead: for each class, the top 5% most confident → pseudo-positive,
    # the bottom 5% → pseudo-negative. Everything else → uncertain (masked).
    
    pseudo_labels = np.zeros_like(all_probs)       # (N, 14)
    pseudo_masks = np.zeros_like(all_probs)         # (N, 14) — 1 if confident, 0 if uncertain
    
    for c in range(14):
        class_probs = all_probs[:, c]
        
        # Percentile-based: adapt to actual model confidence
        upper = min(np.percentile(class_probs, 95), thresholds[c])  # Whichever is LOWER
        lower = max(np.percentile(class_probs, 5), 1.0 - thresholds[c])  # Whichever is HIGHER
        
        # Confident positive
        confident_pos = class_probs > upper
        pseudo_labels[confident_pos, c] = 1.0
        pseudo_masks[confident_pos, c] = 1.0
        
        # Confident negative
        confident_neg = class_probs < lower
        pseudo_labels[confident_neg, c] = 0.0
        pseudo_masks[confident_neg, c] = 1.0
    
    # An image "passes" if it has at least one POSITIVE confident class
    confident_positive_per_image = np.zeros(len(all_probs))
    for c in range(14):
        upper = min(np.percentile(all_probs[:, c], 95), thresholds[c])
        confident_positive_per_image += (all_probs[:, c] > upper).astype(float)
    
    passed_mask = confident_positive_per_image >= 1
    
    # Limit to top 15% most confident per round (gradual expansion)
    max_per_round = max(500, int(0.15 * len(all_probs)))
    if passed_mask.sum() > max_per_round:
        passed_indices = np.where(passed_mask)[0]
        max_confs = all_probs[passed_indices].max(axis=1)
        top_k = np.argsort(max_confs)[-max_per_round:]
        new_mask = np.zeros(len(all_probs), dtype=bool)
        new_mask[passed_indices[top_k]] = True
        passed_mask = new_mask
    
    pseudo_indices = [all_indices[i] for i in range(len(all_indices)) if passed_mask[i]]
    passed_labels = pseudo_labels[passed_mask]
    passed_masks = pseudo_masks[passed_mask]
    
    # ---- Statistics ----
    stats = {
        "total_unlabeled": len(all_probs),
        "passed_threshold": int(passed_mask.sum()),
        "rejected": int((~passed_mask).sum()),
        "pass_rate": float(passed_mask.mean()),
        "per_class_thresholds": {CHEST_LABELS[i]: float(thresholds[i]) for i in range(14)},
        "per_class_confident": {
            CHEST_LABELS[i]: int(pseudo_masks[:, i].astype(int).sum()) for i in range(14)
        },
        "avg_confidence": float(all_probs.max(axis=1).mean()),
    }
    
    print(f"\n  Pseudo-label generation:")
    print(f"    Total unlabeled: {stats['total_unlabeled']}")
    print(f"    Passed threshold: {stats['passed_threshold']} ({stats['pass_rate']*100:.1f}%)")
    print(f"    Rejected: {stats['rejected']}")
    print(f"    Avg max confidence: {stats['avg_confidence']:.4f}")
    
    return {
        "pseudo_indices": pseudo_indices,
        "pseudo_labels": passed_labels,
        "pseudo_masks": passed_masks,
        "confidences": all_probs,
        "stats": stats,
    }


def compute_adaptive_thresholds(probs: np.ndarray, base_threshold: float = 0.95) -> np.ndarray:
    """
    Compute per-class adaptive thresholds.
    
    Rare classes get LOWER thresholds (more lenient) to avoid being
    completely excluded from pseudo-labels.
    Common classes get HIGHER thresholds (stricter) to avoid flooding.
    
    Args:
        probs: Sigmoid probabilities for all unlabeled data, shape (N, 14)
        base_threshold: Starting threshold value
    
    Returns:
        Array of 14 per-class thresholds
    """
    # Estimate class frequency from predictions
    pred_positives = (probs > 0.5).mean(axis=0)  # How often each class is predicted positive
    
    # Normalize: rare classes get lower τ, common classes get higher τ
    median_freq = np.median(pred_positives)
    
    thresholds = np.zeros(14)
    for c in range(14):
        if pred_positives[c] < 1e-6:
            thresholds[c] = base_threshold * 0.8  # Very rare → much more lenient
        else:
            # ratio > 1 for common classes, < 1 for rare classes
            ratio = pred_positives[c] / (median_freq + 1e-8)
            # Clamp the adjustment
            adjustment = np.clip(ratio, 0.85, 1.10)
            thresholds[c] = np.clip(base_threshold * adjustment, 0.7, 0.99)
    
    return thresholds


def compute_entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    """
    Entropy minimization loss for unlabeled data.
    
    Pushes the model to make sharper predictions (closer to 0 or 1)
    instead of uncertain ones (near 0.5).
    
    For multi-label sigmoid outputs, entropy per class is:
        H = -p*log(p) - (1-p)*log(1-p)
    
    Minimizing this encourages the model to commit to 0 or 1.
    
    Args:
        probs: Sigmoid probabilities, shape (B, 14)
    
    Returns:
        Mean entropy loss (scalar)
    """
    eps = 1e-7
    probs = torch.clamp(probs, eps, 1.0 - eps)
    entropy = -probs * torch.log(probs) - (1.0 - probs) * torch.log(1.0 - probs)
    return entropy.mean()
