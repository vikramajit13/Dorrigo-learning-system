# Dorrigo Learning Systems

A long-term research-led ML engineering project focused on understanding, training, evaluating, and improving models. Agent infrastructure comes later, only when needed for experiments.

## What is in this starter?
- Project charter: `docs/project_charter.md`
- Experiment notes template: `docs/experiment_log_template.md`
- First challenge: `experiments/001_autograd/README.md`
- Deliberately incomplete implementation: `src/dorrigo_learning/foundations/autograd.py`
- Acceptance tests: `tests/test_autograd.py`
- Optional device check: `scripts/check_environment.py`
- **Apple Silicon 16 GB setup guide:** `docs/apple_silicon_16gb.md`
- Optional MPS test for later exercises: `scripts/check_torch_mps.py`
- Conservative future transformer configuration: `configs/transformer_16gb_start.json`

## Set up
Requires Python 3.11+. On an Apple Silicon Mac with 16 GB, read `docs/apple_silicon_16gb.md` first. For the initial autodiff exercise, a GPU and PyTorch are not needed.

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
python scripts/check_environment.py
python -m pytest -q
```

**The initial tests are supposed to fail.** Their purpose is to define your first implementation challenge.

## How to work
1. Read the charter and Experiment 001 requirements.
2. Derive expected gradients manually before programming.
3. Implement `Value` without copying an existing autodiff engine.
4. Run and extend the tests. Confirm analytical gradients with finite differences.
5. Complete the explanation and record a technical learning note.

## First milestone
Understand and implement reverse-mode autodiff. The next milestone will be a small neural network trained using that autodiff engine, followed by a bigram language-model baseline and a tiny transformer built in PyTorch.

## Constraints
No cloud spend or GPU is required to complete Experiment 001. Do not commit model weights, training corpora, secrets, or sensitive workplace data.
