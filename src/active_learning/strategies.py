"""
Active Learning: Annotation selection strategies.

Given a budget of N images to label, which N should we pick?
Random selection wastes budget. Smart selection maximizes information gain.

Two strategies:
1. Uncertainty Sampling — pick images the model is most confused about
2. Core-Set Selection — pick images that best cover the data distribution
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist

from src.utils.device import get_amp_context


@torch.no_grad()
def get_predictions_and_features(model, dataloader: DataLoader, device: torch.device,
                                  use_amp: bool = True) -> dict:
    """
    Run model on data and collect predictions + backbone features.
    
    Args:
        model: ChestClassifier with extract_features() method
        dataloader: DataLoader for the pool to select from
        device: torch device
        use_amp: Mixed precision
    
    Returns:
        Dict with 'probabilities' (N, 14) and 'features' (N, 512)
    """
    model.eval()
    amp_context = get_amp_context(enabled=use_amp)
    
    all_probs = []
    all_features = []
    
    for images, _ in tqdm(dataloader, desc="Extracting features", leave=False, ncols=100):
        images = images.to(device, non_blocking=True)
        
        with amp_context:
            probs = model.predict_proba(images)           # (B, 14)
            features = model.extract_features(images)     # (B, 512)
        
        all_probs.append(probs.cpu().numpy())
        all_features.append(features.cpu().numpy())
    
    return {
        "probabilities": np.concatenate(all_probs, axis=0),
        "features": np.concatenate(all_features, axis=0),
    }


def uncertainty_sampling(probabilities: np.ndarray, budget: int) -> list:
    """
    Select the most uncertain images (highest entropy).
    
    For multi-label: compute per-class binary entropy, then average across classes.
    Images near the decision boundary (prob ≈ 0.5) have highest entropy.
    
    Args:
        probabilities: Sigmoid probs, shape (N, 14)
        budget: Number of images to select
    
    Returns:
        List of indices (into the pool) to label
    """
    eps = 1e-7
    p = np.clip(probabilities, eps, 1.0 - eps)
    
    # Binary entropy per class: H = -p*log(p) - (1-p)*log(1-p)
    entropy_per_class = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    
    # Average entropy across all 14 classes
    mean_entropy = entropy_per_class.mean(axis=1)  # (N,)
    
    # Select top-k most uncertain
    budget = min(budget, len(mean_entropy))
    selected_indices = np.argsort(mean_entropy)[-budget:][::-1]  # Highest entropy first
    
    print(f"  Uncertainty Sampling:")
    print(f"    Pool size: {len(probabilities)}")
    print(f"    Selected: {budget} most uncertain images")
    print(f"    Entropy range: {mean_entropy[selected_indices[-1]]:.4f} — "
          f"{mean_entropy[selected_indices[0]]:.4f}")
    
    return selected_indices.tolist()


def coreset_selection(features: np.ndarray, labeled_features: np.ndarray,
                      budget: int) -> list:
    """
    Core-set selection: pick images that maximize coverage of the data distribution.
    
    Greedy algorithm: iteratively pick the unlabeled point that is FARTHEST
    from all currently labeled points. This ensures the selected set covers
    the full feature space, not just dense regions.
    
    Args:
        features: Feature vectors of the unlabeled pool, shape (N_unlabeled, 512)
        labeled_features: Feature vectors of already-labeled data, shape (N_labeled, 512)
        budget: Number of images to select
    
    Returns:
        List of indices (into the unlabeled pool) to label
    """
    budget = min(budget, len(features))
    
    # Start with the already-labeled features as our "covered" set
    if len(labeled_features) > 0:
        covered = labeled_features.copy()
    else:
        # If no labeled data yet, start with the medoid
        center = features.mean(axis=0, keepdims=True)
        dists = cdist(features, center, metric="euclidean").squeeze()
        first_idx = np.argmin(dists)
        covered = features[first_idx:first_idx+1]
        budget -= 1  # One already selected
    
    selected = []
    remaining = set(range(len(features)))
    
    print(f"  Core-Set Selection (greedy, budget={budget})...")
    
    for i in tqdm(range(budget), desc="Core-set", leave=False, ncols=100):
        if not remaining:
            break
        
        remaining_list = list(remaining)
        remaining_features = features[remaining_list]
        
        # Distance from each remaining point to the nearest covered point
        dists = cdist(remaining_features, covered, metric="euclidean")
        min_dists = dists.min(axis=1)  # Distance to nearest labeled point
        
        # Pick the farthest point
        farthest_idx = np.argmax(min_dists)
        original_idx = remaining_list[farthest_idx]
        
        selected.append(original_idx)
        remaining.remove(original_idx)
        
        # Add to covered set
        covered = np.vstack([covered, features[original_idx:original_idx+1]])
    
    print(f"    Selected {len(selected)} images covering the feature space")
    
    return selected


def select_annotations(model, unlabeled_loader: DataLoader, 
                       labeled_loader: DataLoader,
                       device: torch.device, strategy: str = "uncertainty",
                       budget: int = 1000, use_amp: bool = True) -> list:
    """
    Main entry point: select which images to annotate.
    
    Args:
        model: Trained ChestClassifier
        unlabeled_loader: DataLoader for unlabeled pool
        labeled_loader: DataLoader for already-labeled data (needed for core-set)
        device: torch device
        strategy: "uncertainty" or "coreset"
        budget: Number of images to select
        use_amp: Mixed precision
    
    Returns:
        List of indices into the unlabeled pool to label
    """
    print(f"\n  Active Learning: {strategy} (budget={budget})")
    
    # Get predictions and features for unlabeled pool
    unlabeled_data = get_predictions_and_features(model, unlabeled_loader, device, use_amp)
    
    if strategy == "uncertainty":
        selected = uncertainty_sampling(unlabeled_data["probabilities"], budget)
    
    elif strategy == "coreset":
        # Also need features for already-labeled data
        labeled_data = get_predictions_and_features(model, labeled_loader, device, use_amp)
        selected = coreset_selection(
            unlabeled_data["features"],
            labeled_data["features"],
            budget
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'uncertainty' or 'coreset'.")
    
    return selected
