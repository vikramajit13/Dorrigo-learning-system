"""Optional environment report; works even before PyTorch is installed."""

import platform
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"OS: {platform.platform()}")
print(f"Machine: {platform.machine()}")
try:
    import torch
except ImportError:
    print("PyTorch: not installed (not needed for Exercise 001)")
else:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Apple MPS available: {torch.backends.mps.is_available()}")
