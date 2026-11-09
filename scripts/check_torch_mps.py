"""Optional Apple Silicon MPS diagnostic; only needed when starting PyTorch exercises."""

import platform
import sys

print(f"Architecture: {platform.machine()}")
if platform.machine() != "arm64":
    print("Warning: expected native Apple Silicon (arm64) Python. Check Rosetta mode.")

try:
    import torch
except ImportError:
    print("PyTorch is not installed. Not needed for Experiment 001.")
    print("When ready: python -m pip install torch")
    sys.exit(0)

print(f"PyTorch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    print("MPS unavailable. Confirm macOS, Apple Silicon, and native PyTorch installation.")
    sys.exit(1)

device = torch.device("mps")
x = torch.randn(16, 16, device=device, requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
if x.grad is None or not torch.isfinite(x.grad).all().item():
    raise RuntimeError("Gradient computation on MPS did not pass validation")
torch.mps.synchronize()
print(f"MPS forward/backward smoke test passed. Example loss: {loss.item():.6f}")
