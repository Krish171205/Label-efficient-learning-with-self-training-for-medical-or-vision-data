"""
Master Pipeline — Run EVERYTHING end-to-end on RTX 4060.

Just run this one file and walk away:
    python scripts/run_full_pipeline.py

It will:
0. Install all dependencies (including CUDA PyTorch)
1. Train the ImageNet baseline
2. Pretrain SimCLR (100 epochs) 
3. Pretrain Rotation (30 epochs)
4. Fine-tune SimCLR backbone
5. Fine-tune Rotation backbone
6. Self-train with ImageNet backbone (10 rounds)
7. Self-train with SimCLR backbone (10 rounds)
8. Generate comparison table + plots
"""

import os
import sys
import subprocess
import time

# Ensure we're in the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

CONFIG = "configs/local_4060.yaml"


def install_dependencies():
    """Install PyTorch with CUDA and all other requirements."""
    print(f"\n{'='*70}")
    print(f"  STEP 0: Installing Dependencies")
    print(f"{'='*70}\n")
    
    # Step 0a: Install PyTorch with CUDA 12.4 support
    print("  Installing PyTorch with CUDA 12.4...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
        "--quiet",
    ])
    if result.returncode != 0:
        print("❌ Failed to install PyTorch with CUDA!")
        print("   Try manually: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    
    # Step 0b: Install everything else from requirements.txt
    print("  Installing remaining dependencies...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "-r", "requirements.txt", "--quiet",
    ])
    if result.returncode != 0:
        print("❌ Failed to install requirements.txt!")
        sys.exit(1)
    
    # Step 0c: Verify GPU is actually visible
    print("\n  Verifying GPU access...")
    verify = subprocess.run(
        [sys.executable, "-c", 
         "import torch; assert torch.cuda.is_available(), 'NO GPU'; "
         "print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}'); "
         "print(f'  ✓ VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB'); "
         "print(f'  ✓ CUDA: {torch.version.cuda}')"],
        capture_output=True, text=True,
    )
    if verify.returncode != 0:
        print("❌ PyTorch installed but CANNOT see your GPU!")
        print("   Your PyTorch may still be CPU-only.")
        print(f"   Error: {verify.stderr}")
        sys.exit(1)
    
    print(verify.stdout)
    print("  ✅ All dependencies installed successfully!\n")


def run_step(step_num: int, description: str, command: list):
    """Run a training step and handle errors."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {description}")
    print(f"  Command: {' '.join(command)}")
    print(f"{'='*70}\n")
    
    start = time.time()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    
    result = subprocess.run(
        command,
        env=env,
        cwd=PROJECT_ROOT,
    )
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n❌ STEP {step_num} FAILED after {elapsed/60:.1f} minutes!")
        print(f"   Command: {' '.join(command)}")
        print(f"   Return code: {result.returncode}")
        print(f"\n   The pipeline has STOPPED. Fix the error above and re-run.")
        print(f"   Completed steps will NOT re-run (checkpoints are saved).")
        sys.exit(1)
    
    print(f"\n✅ Step {step_num} completed in {elapsed/60:.1f} minutes\n")
    return elapsed


def main():
    total_start = time.time()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           LABEL-EFFICIENT LEARNING — MASTER PIPELINE            ║
║                   RTX 4060 Local Execution                      ║
║                                                                  ║
║  Config: {CONFIG:<52s} ║
║  No time limit. Let it run until completion.                     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 0: Install dependencies and verify GPU
    install_dependencies()
    
    steps = [
        (1, "Baseline (ImageNet Transfer Learning)", [
            sys.executable, "-u", "scripts/train_baseline.py",
            "--config", CONFIG,
        ]),
        (2, "SimCLR Contrastive Pretraining (100 epochs)", [
            sys.executable, "-u", "scripts/train_simclr.py",
            "--config", CONFIG,
        ]),
        (3, "Rotation Pretext Pretraining (30 epochs)", [
            sys.executable, "-u", "scripts/train_pretext.py",
            "--task", "rotation", "--config", CONFIG,
        ]),
        (4, "SimCLR → Fine-tune on labeled data", [
            sys.executable, "-u", "scripts/train_simclr_finetune.py",
            "--config", CONFIG,
        ]),
        (5, "Rotation → Fine-tune on labeled data", [
            sys.executable, "-u", "scripts/train_pretext_finetune.py",
            "--task", "rotation", "--config", CONFIG,
        ]),
        (6, "Self-Training with ImageNet backbone (10 rounds)", [
            sys.executable, "-u", "scripts/train_self_training.py",
            "--backbone", "imagenet", "--config", CONFIG,
        ]),
        (7, "Self-Training with SimCLR backbone (10 rounds)", [
            sys.executable, "-u", "scripts/train_self_training.py",
            "--backbone", "simclr", "--config", CONFIG,
        ]),
        (8, "Generate Comparison Table + Plots", [
            sys.executable, "-u", "scripts/run_comparison.py",
            "--config", CONFIG,
        ]),
    ]
    
    timings = {}
    
    for step_num, description, command in steps:
        elapsed = run_step(step_num, description, command)
        timings[description] = elapsed
    
    total_elapsed = time.time() - total_start
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  ✅ ALL 8 STEPS COMPLETED SUCCESSFULLY!")
    print(f"  Total time: {total_elapsed/3600:.1f} hours ({total_elapsed/60:.0f} minutes)")
    print(f"{'='*70}")
    print(f"\n  Step timings:")
    for desc, t in timings.items():
        print(f"    {desc}: {t/60:.1f} min")
    print(f"\n  Results are in: outputs/results/")
    print(f"  Plots are in:   outputs/results/plots/")
    print(f"  Checkpoints:    outputs/checkpoints/")
    print(f"\n  🎉 You're done! Check outputs/results/comparison_summary.json")


if __name__ == "__main__":
    main()
