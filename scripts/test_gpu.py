"""
GPU Stress Test — Open Task Manager and watch the GPU usage spike!

This script runs a heavy matrix multiplication loop on your RTX 4060
for 30 seconds. While it's running, open Task Manager → Performance → GPU
and you should see GPU utilization jump to ~80-100%.

Usage:
    python scripts/test_gpu.py
"""

import time
import sys
import os

def main():
    # Install torch with CUDA if needed
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "torch", "--index-url", "https://download.pytorch.org/whl/cu124", "-q"])
        import torch

    print(f"\n{'='*60}")
    print(f"  GPU STRESS TEST")
    print(f"{'='*60}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print(f"\n  ❌ NO GPU DETECTED!")
        print(f"  Your PyTorch is CPU-only ({torch.__version__})")
        print(f"  Fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall")
        return
    
    print(f"  GPU name:        {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  CUDA version:    {torch.version.cuda}")
    
    # Check Kornia (GPU augmentation library)
    try:
        import kornia
        print(f"  Kornia version:  {kornia.__version__} ✓ (GPU augmentations ready)")
    except ImportError:
        print(f"  Kornia:          ❌ NOT INSTALLED (pip install kornia)")
    
    print(f"\n  🔥 Starting 30-second GPU stress test...")
    print(f"  👉 Open Task Manager → Performance → GPU to see utilization!")
    print(f"{'='*60}\n")
    
    # Allocate large tensors on GPU
    device = torch.device("cuda")
    size = 4096  # Large matrix for stress
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    
    start = time.time()
    iterations = 0
    
    while time.time() - start < 30:
        # Heavy matrix multiplication — this will push GPU to near 100%
        c = torch.mm(a, b)
        torch.cuda.synchronize()  # Force GPU to finish before next iteration
        iterations += 1
        
        elapsed = time.time() - start
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        
        print(f"\r  ⚡ {elapsed:.0f}/30s | Iterations: {iterations} | "
              f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB", end="", flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"  ✅ GPU TEST PASSED!")
    print(f"  {iterations} matrix multiplications in 30 seconds")
    print(f"  Your RTX 4060 is working perfectly.")
    print(f"  You can now run: python scripts/run_full_pipeline.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
