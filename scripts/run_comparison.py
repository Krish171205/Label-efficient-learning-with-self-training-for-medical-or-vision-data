"""
Run ALL experiments and compare results.

This is the master script that orchestrates the full pipeline.
It does NOT run training — it collects results from already-completed
experiments and generates comparison tables + plots.

Usage:
    python run_comparison.py

Prerequisites: Run these first (in order):
    1. python train_baseline.py
    2. python train_simclr.py
    3. python train_simclr_finetune.py
    4. python train_pretext.py --task rotation
    5. python train_pretext_finetune.py --task rotation
    6. python train_pretext.py --task inpainting
    7. python train_pretext_finetune.py --task inpainting
    8. python train_self_training.py --backbone imagenet
    9. python train_self_training.py --backbone simclr
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config


def load_results(results_dir: str, experiment_name: str) -> dict:
    """Load results JSON for a given experiment."""
    path = os.path.join(results_dir, experiment_name, "results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    results_dir = cfg.logging.results_dir
    label_fraction = cfg.label_fractions[0]  # 0.01
    
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPARISON — Label Fraction: {label_fraction*100:.0f}%")
    print(f"{'='*70}\n")
    
    # ---- Collect results ----
    experiments = {
        "Baseline (ImageNet)":          f"baseline_lf{label_fraction}",
        "SimCLR → Fine-tune":           f"simclr_finetune_lf{label_fraction}",
        "Rotation → Fine-tune":         f"pretext_rotation_lf{label_fraction}",
        "Inpainting → Fine-tune":       f"pretext_inpainting_lf{label_fraction}",
        "Self-Train (ImageNet)":        f"self_training_imagenet_lf{label_fraction}",
        "Self-Train (SimCLR)":          f"self_training_simclr_lf{label_fraction}",
    }
    
    results = {}
    for name, exp_dir in experiments.items():
        data = load_results(results_dir, exp_dir)
        if data:
            results[name] = data
    
    if not results:
        print("❌ No results found! Run the training scripts first.")
        print("\nRequired run order:")
        print("  1. python train_baseline.py")
        print("  2. python train_simclr.py")
        print("  3. python train_simclr_finetune.py")
        print("  4. python train_pretext.py --task rotation")
        print("  5. python train_pretext_finetune.py --task rotation")
        print("  6. python train_pretext.py --task inpainting")
        print("  7. python train_pretext_finetune.py --task inpainting")
        print("  8. python train_self_training.py --backbone imagenet")
        print("  9. python train_self_training.py --backbone simclr")
        return
    
    # ---- Comparison Table ----
    print(f"{'Method':<30s} {'AUROC':>8s} {'mAP':>8s} {'F1-mac':>8s} {'F1-mic':>8s} {'Time':>8s}")
    print("─" * 72)
    
    sorted_results = sorted(results.items(), 
                            key=lambda x: x[1].get("test_metrics", {}).get("mean_auroc", 0),
                            reverse=True)
    
    best_auroc = 0
    best_method = ""
    
    for name, data in sorted_results:
        tm = data.get("test_metrics", {})
        auroc = tm.get("mean_auroc", 0)
        mAP = tm.get("mAP", 0)
        f1_mac = tm.get("f1_macro", 0)
        f1_mic = tm.get("f1_micro", 0)
        time_min = data.get("training_time_minutes", 0)
        
        marker = " ★" if auroc == max(r.get("test_metrics", {}).get("mean_auroc", 0) 
                                       for r in results.values()) else ""
        
        print(f"{name:<30s} {auroc:>8.4f} {mAP:>8.4f} {f1_mac:>8.4f} {f1_mic:>8.4f} {time_min:>7.1f}m{marker}")
        
        if auroc > best_auroc:
            best_auroc = auroc
            best_method = name
    
    print("─" * 72)
    print(f"{'★ = best':>72s}")
    
    # ---- Improvement over baseline ----
    baseline_data = results.get("Baseline (ImageNet)")
    if baseline_data:
        baseline_auroc = baseline_data["test_metrics"]["mean_auroc"]
        print(f"\n  Baseline AUROC: {baseline_auroc:.4f}")
        print(f"  Best method:    {best_method} ({best_auroc:.4f})")
        improvement = (best_auroc - baseline_auroc) / baseline_auroc * 100
        print(f"  Improvement:    +{improvement:.1f}%")
    
    # ---- Per-class AUROC comparison ----
    print(f"\n{'='*70}")
    print(f"  PER-CLASS AUROC COMPARISON")
    print(f"{'='*70}")
    
    # Header
    method_names = [name for name, _ in sorted_results[:4]]  # Top 4 methods
    header = f"{'Disease':<22s}" + "".join(f"{n[:12]:>14s}" for n in method_names)
    print(header)
    print("─" * (22 + 14 * len(method_names)))
    
    # Get disease labels
    from src.utils.metrics import CHEST_LABELS
    
    for label in CHEST_LABELS:
        row = f"{label:<22s}"
        for name in method_names:
            per_class = results[name].get("test_metrics", {}).get("per_class_auroc", {})
            val = per_class.get(label, 0)
            row += f"{val:>14.4f}"
        print(row)
    
    # ---- Self-training progression ----
    for name, data in results.items():
        if "round_results" in data:
            print(f"\n{'='*70}")
            print(f"  SELF-TRAINING PROGRESSION: {name}")
            print(f"{'='*70}")
            print(f"  {'Round':<8s} {'Pool Size':>12s} {'Val AUROC':>12s} {'Growth':>10s}")
            print("  " + "─" * 44)
            
            prev_auroc = 0
            for r in data["round_results"]:
                auroc = r["best_val_auroc"]
                delta = f"+{auroc-prev_auroc:.4f}" if prev_auroc > 0 else "—"
                pool = r["labeled_pool_size"]
                bar = "█" * int(auroc * 25)
                print(f"  {r['round']:<8d} {pool:>12,d} {auroc:>12.4f} {delta:>10s}  {bar}")
                prev_auroc = auroc
    
    # ---- Generate plots ----
    try:
        generate_plots(results, results_dir, label_fraction)
    except Exception as e:
        print(f"\n⚠ Could not generate plots: {e}")
    
    # ---- Save summary ----
    summary_path = os.path.join(results_dir, "comparison_summary.json")
    summary = {
        "label_fraction": label_fraction,
        "best_method": best_method,
        "best_auroc": best_auroc,
        "all_results": {
            name: {
                "auroc": data.get("test_metrics", {}).get("mean_auroc", 0),
                "mAP": data.get("test_metrics", {}).get("mAP", 0),
                "time_minutes": data.get("training_time_minutes", 0),
            }
            for name, data in results.items()
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Summary saved to {summary_path}")


def generate_plots(results: dict, results_dir: str, label_fraction: float):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # ---- Plot 1: AUROC Bar Chart ----
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    aurocs = []
    colors = []
    color_map = {
        "Baseline": "#6c757d",
        "SimCLR": "#0d6efd",
        "Rotation": "#198754",
        "Inpainting": "#20c997",
        "Self-Train": "#fd7e14",
    }
    
    for name, data in sorted(results.items(), 
                              key=lambda x: x[1]["test_metrics"]["mean_auroc"]):
        methods.append(name)
        aurocs.append(data["test_metrics"]["mean_auroc"])
        
        color = "#6c757d"
        for key, c in color_map.items():
            if key in name:
                color = c
                break
        colors.append(color)
    
    bars = ax.barh(methods, aurocs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Test AUROC")
    ax.set_title(f"Method Comparison — ChestMNIST ({label_fraction*100:.0f}% Labels)")
    ax.set_xlim(0.5, max(aurocs) + 0.05)
    
    for bar, val in zip(bars, aurocs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "auroc_comparison.png"), dpi=150)
    plt.close()
    print(f"  📈 Plot saved: {plots_dir}/auroc_comparison.png")
    
    # ---- Plot 2: Self-training progression ----
    for name, data in results.items():
        if "round_results" in data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            rounds = [r["round"] for r in data["round_results"]]
            aurocs = [r["best_val_auroc"] for r in data["round_results"]]
            pools = [r["labeled_pool_size"] for r in data["round_results"]]
            
            ax1.plot(rounds, aurocs, "o-", color="#fd7e14", linewidth=2, markersize=8)
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Val AUROC")
            ax1.set_title(f"AUROC vs Self-Training Round")
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(rounds, pools, color="#0d6efd", alpha=0.7)
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Labeled Pool Size")
            ax2.set_title("Pool Growth per Round")
            ax2.grid(True, alpha=0.3)
            
            fig.suptitle(name, fontsize=14, fontweight="bold")
            plt.tight_layout()
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(os.path.join(plots_dir, f"{safe_name}_progression.png"), dpi=150)
            plt.close()
            print(f"  📈 Plot saved: {plots_dir}/{safe_name}_progression.png")


if __name__ == "__main__":
    main()
