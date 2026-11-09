# Apple Silicon 16 GB: Dorrigo Development Guide

This is the local-first development profile for Dorrigo Learning Systems. It assumes an Apple Silicon Mac with **16 GB unified memory**. CPU and GPU share memory, so avoid treating 16 GB as dedicated GPU VRAM.

## Goals

1. Learn every fundamental mechanism with small problems before optimizing.
2. Use `cpu` for early reference implementations and `mps` for small PyTorch training runs.
3. Keep experiments reproducible, including environment, random seed, dataset split, model config, device, runtime, and memory/latency results.
4. Do not add cloud spend until benchmark evidence shows local compute is insufficient.

## Host setup

- Use native Apple Silicon (`arm64`) Python, preferably Python 3.12 for broad ecosystem compatibility. Python 3.11 also works for this project.
- Install Xcode command-line tools: `xcode-select --install` (harmless to omit if already present).
- Confirm `uname -m` prints `arm64`. If it prints `x86_64`, check whether the terminal is running under Rosetta.
- Use the Python virtual environment in the project root.

```bash
uname -m
python3.12 --version  # if installed; otherwise see python3 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/check_environment.py
python -m pytest -q
```

If `python3.12` is unavailable, install native Python 3.12 using Homebrew or python.org first, or use an existing compatible native Python executable and replace `python3.12` in the commands. **Do not install PyTorch to complete Experiment 001.** The failing tests in this starter are intentional exercises.

## When you begin PyTorch work (after the autodiff exercise)

```bash
python -m pip install torch
python scripts/check_torch_mps.py
```

The diagnostic script tests that PyTorch sees the MPS backend and can run a small differentiable GPU operation. A successful `import torch` does **not** prove MPS acceleration is available; the backend check does. Use a macOS/PyTorch combination supported by the installed PyTorch release.

A typical device selector for future training scripts:

```python
import torch

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)
# Ensure input and target tensors also use device before the forward pass.
```

## Deliberately conservative training ladder

| Gate | First device | Starting model / shape | Notes |
|---|---|---|---|
| Autodiff engine | CPU | Scalar graph | No third-party numerical libraries required |
| Two-layer MLP | CPU | Small synthetic dataset | Verify your gradients against PyTorch |
| Bigram language model | CPU, then MPS | Small character-level corpus | Establish training/validation loss |
| Decoder transformer | MPS | 4 layers, width 128, 4 heads, context 128, microbatch 4 | An initial experiment, not a fixed architecture |
| TinyStories experiments | MPS | Roughly 5–20M parameters at first; short sequences | Measure memory and throughput before scaling |
| SFT / LoRA | MPS / MLX, later | Start around 0.5B parameters, short context, microbatch 1 | 1.5B may be possible depending on the setup; use cloud if needed |
| Long-context / CUDA-specific research | Cloud GPU | Scale to the question being tested | Do not force unsupported GPU work onto MPS |

The `configs/transformer_16gb_start.json` file records a **future suggested starting point**, not an executable training command. It will be wired into the transformer exercise after you implement its training loop.

## Responsible 16 GB memory use

- Start with `float32` to debug correctness, then explore mixed precision **only with validation** for supported operations and chip types.
- Reduce **microbatch size first**, then context length, then model width/layers if you exhaust available memory.
- Use gradient accumulation for effective batch size when appropriate; it does not eliminate the memory cost of a single microbatch.
- Close heavyweight apps and watch macOS Activity Monitor **Memory Pressure** and swap usage during experiments.
- Run a tiny successful training/evaluation loop before increasing data or model size.
- Record effective batch size, training tokens/sec, validation loss and memory pressure in the experiment report.
- Never disable MPS allocation safety limits just to fit an oversized model. Unexpected CPU fallbacks may be slow; investigate instead of silently masking failures.

## Comparison protocol

When investigating CPU vs MPS performance:

1. Use the same model, workload, seed and precision where possible.
2. Run a warm-up; GPU kernels execute asynchronously.
3. Synchronize the MPS device before and after timing the measured section.
4. Track peak observed memory pressure, tokens/sec and validation loss.
5. Keep output quality/correctness equal before comparing speed.

## Cloud escalation triggers

Move an experiment to a rented NVIDIA GPU only if a clear research question requires CUDA kernels, hardware profiling, multiple GPUs, larger model/sequence length than local memory allows, or impractical run time. Profile locally first; compare cloud cost and expected benefit. No immediate hardware purchase is recommended.

## References

- PyTorch MPS: https://docs.pytorch.org/docs/stable/notes/mps.html
- Hugging Face Apple Silicon guide: https://huggingface.co/docs/transformers/main/apple_silicon
- Apple MLX: https://github.com/ml-explore/mlx
