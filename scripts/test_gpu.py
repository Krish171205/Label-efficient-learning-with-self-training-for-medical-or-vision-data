"""
GPU Visibility Test — Allocates 6 GB of VRAM so you can SEE it in Task Manager.

Usage:
    python scripts/test_gpu.py

What to look for in Task Manager:
    1. Click on "GPU 1" (NVIDIA GeForce RTX 4060)
    2. Look at the BOTTOM where it says "Dedicated GPU memory"
    3. It will jump from 0.0/8.0 GB to ~6.0/8.0 GB
    4. For the utilization graph: click the small dropdown arrow on
       any graph label and change it from "3D" to "CUDA"
"""

import time
import sys
import subprocess

def main():
    try:
        import torch
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "torch", "--index-url", "https://download.pytorch.org/whl/cu124", "-q"])
        import torch

    if not torch.cuda.is_available():
        print("❌ No CUDA GPU found!")
        return
    
    print(f"\n{'='*60}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}")
    
    print(f"\n  Step 1: Allocating 6 GB of VRAM...")
    print(f"         Open Task Manager → Performance → GPU 1 (NVIDIA)")
    print(f"         Watch 'Dedicated GPU memory' at the bottom!\n")
    
    device = torch.device("cuda:0")
    
    # Allocate 6 GB of VRAM — impossible to miss in Task Manager
    chunks = []
    target_gb = 6.0
    chunk_size = 8192  # 8192 x 8192 x 2 bytes = 128 MB per chunk
    num_chunks = int(target_gb * 1024 / 128)  # ~48 chunks
    
    for i in range(num_chunks):
        chunks.append(torch.randn(chunk_size, chunk_size, device=device, dtype=torch.float16))
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"\r  Allocated: {allocated:.1f} / {target_gb:.0f} GB", end="", flush=True)
    
    allocated = torch.cuda.memory_allocated(0) / 1e9
    print(f"\n\n  ✅ {allocated:.1f} GB allocated on NVIDIA GPU!")
    print(f"  👉 CHECK TASK MANAGER NOW — 'Dedicated GPU memory' should show ~6 GB")
    print(f"\n  Holding for 30 seconds so you can look...\n")
    
    for i in range(30, 0, -1):
        print(f"\r  Releasing in {i} seconds...  ", end="", flush=True)
        time.sleep(1)
    
    # Cleanup
    del chunks
    torch.cuda.empty_cache()
    
    print(f"\n\n  ✅ Memory released. 'Dedicated GPU memory' should drop back to 0.")
    print(f"  Your NVIDIA GPU is confirmed working.\n")


if __name__ == "__main__":
    main()
