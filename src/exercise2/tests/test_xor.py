"""Exercise 003: fast structural tests plus a training-loop acceptance test.

Expected: the training-loop test fails with NotImplementedError until the user
implements `train_full_batch` in experiments/003_xor/train.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from dorrigo_learning.foundations.nn import MLP
from dorrigo_learning.foundations.optim import mse

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "003_xor" / "train.py"
_SPEC = importlib.util.spec_from_file_location("dorrigo_xor_experiment", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
xor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(xor)


def test_xor_has_four_unique_cases() -> None:
    assert len(xor.XOR_DATA) == 4
    assert len({x for x, _ in xor.XOR_DATA}) == 4
    assert [y for _, y in xor.XOR_DATA] == [0.0, 1.0, 1.0, 0.0]


def test_mlp_forward_is_one_value() -> None:
    model = MLP([2, 4, 1], seed=11)
    assert len(model([0.0, 1.0])) == 1
    # 4 hidden neurons * (2 weights + 1 bias) + 1 output * (4 weights + 1 bias)
    assert len(model.parameters()) == 17


def test_all_linear_mlp_obeys_affine_identity() -> None:
    model = MLP([2, 4, 1], hidden_activation="linear", output_activation="linear", seed=11)
    f00 = model([0.0, 0.0])[0].data
    f01 = model([0.0, 1.0])[0].data
    f10 = model([1.0, 0.0])[0].data
    f11 = model([1.0, 1.0])[0].data
    assert f00 + f11 == pytest.approx(f01 + f10, abs=1e-10)


def test_loss_gradient_reaches_all_model_layers() -> None:
    # Using a linear hidden layer avoids ReLU zeros, making it easier to check
    # that gradient paths exist through both layers.
    model = MLP([2, 4, 1], hidden_activation="linear", seed=11)
    predictions = [model(x)[0] for x, _ in xor.XOR_DATA]
    loss = mse(predictions, [target for _, target in xor.XOR_DATA])
    loss.backward()
    assert any(abs(p.grad) > 1e-12 for p in model.layers[0].parameters())
    assert any(abs(p.grad) > 1e-12 for p in model.layers[-1].parameters())


def test_training_loop_records_progress() -> None:
    model = MLP([2, 4, 1], seed=11)
    before = xor.evaluate(model)["mse"]
    history = xor.train_full_batch(model, steps=100, learning_rate=0.05, log_every=25)
    after = xor.evaluate(model)["mse"]
    assert isinstance(history, list) and history
    assert history[-1]["step"] == 100
    assert isinstance(before, float) and isinstance(after, float)
    assert after < before
