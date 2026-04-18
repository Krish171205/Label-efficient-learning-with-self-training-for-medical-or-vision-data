"""
Training log — automatically records results from every run.

Every script calls `log_result()` after finishing. Results are appended to
`outputs/training_log.csv` so you can track progress across sessions.

Usage in any script:
    from scripts.training_log import log_result
    log_result(
        script="train_simclr.py",
        config="colab.yaml",
        auroc=0.78,
        epochs=50,
        notes="First Colab run"
    )
"""

import os
import sys
import csv
from datetime import datetime

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_ROOT, "outputs", "training_log.csv")

COLUMNS = [
    "timestamp", "script", "config", "backbone", "label_fraction",
    "epochs_trained", "auroc", "mAP", "loss", "data_size",
    "batch_size", "time_minutes", "notes"
]


def log_result(script="", config="", backbone="", label_fraction="",
               epochs_trained="", auroc="", mAP="", loss="",
               data_size="", batch_size="", time_minutes="", notes=""):
    """Append one row to the training log CSV."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    file_exists = os.path.exists(LOG_PATH)
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": script,
        "config": config,
        "backbone": backbone,
        "label_fraction": label_fraction,
        "epochs_trained": epochs_trained,
        "auroc": f"{auroc:.4f}" if isinstance(auroc, float) else auroc,
        "mAP": f"{mAP:.4f}" if isinstance(mAP, float) else mAP,
        "loss": f"{loss:.4f}" if isinstance(loss, float) else loss,
        "data_size": data_size,
        "batch_size": batch_size,
        "time_minutes": f"{time_minutes:.1f}" if isinstance(time_minutes, float) else time_minutes,
        "notes": notes,
    }
    
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"📝 Result logged to {LOG_PATH}")


def print_log():
    """Print the training log as a formatted table."""
    if not os.path.exists(LOG_PATH):
        print("No training log found yet.")
        return
    
    with open(LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("Training log is empty.")
        return
    
    print(f"\n{'='*100}")
    print(f"  TRAINING LOG ({len(rows)} entries)")
    print(f"{'='*100}")
    print(f"{'Timestamp':<20} {'Script':<28} {'AUROC':<8} {'Epochs':<8} {'Data':<8} {'Time':<8} {'Notes'}")
    print(f"{'-'*100}")
    
    for row in rows:
        print(f"{row['timestamp']:<20} {row['script']:<28} {row['auroc']:<8} "
              f"{row['epochs_trained']:<8} {row['data_size']:<8} "
              f"{row['time_minutes']:<8} {row.get('notes', '')}")
    
    print(f"{'='*100}\n")


if __name__ == "__main__":
    print_log()
