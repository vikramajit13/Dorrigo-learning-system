"""Exercise 004 acceptance tests. Initial failures are expected until TODOs are done."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from dorrigo_learning.foundations.autograd import Value
from dorrigo_learning.foundations.classification import binary_cross_entropy_with_logits
from dorrigo_learning.foundations.nn import MLP

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "004_binary_classification" / "train.py"
SPEC = importlib.util.spec_from_file_location("dorrigo_ex004", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


def test_sigmoid_forward_and_backward_at_zero() -> None:
    x = Value(0.0)
    prob = x.sigmoid()
    prob.backward()
    assert prob.data == pytest.approx(0.5)
    assert x.grad == pytest.approx(0.25)


@pytest.mark.parametrize("target, expected_gradient", [(0.0, 0.5), (1.0, -0.5)])
def test_bce_at_zero(target: float, expected_gradient: float) -> None:
    z = Value(0.0)
    loss = binary_cross_entropy_with_logits([z], [target])
    assert loss.data == pytest.approx(math.log(2.0))
    loss.backward()
    assert z.grad == pytest.approx(expected_gradient)


def test_bce_mean_gradient() -> None:
    z1, z2 = Value(0.0), Value(0.0)
    loss = binary_cross_entropy_with_logits([z1, z2], [0.0, 1.0])
    loss.backward()
    assert loss.data == pytest.approx(math.log(2.0))
    assert z1.grad == pytest.approx(0.25)
    assert z2.grad == pytest.approx(-0.25)


def test_bce_shared_input_accumulates_gradients() -> None:
    z = Value(0.0)
    loss = binary_cross_entropy_with_logits([z, z], [0.0, 0.0])
    loss.backward()
    assert z.grad == pytest.approx(0.5)


def test_bce_extreme_logits_are_finite_and_gradients_correct() -> None:
    for value, target, expected_loss, expected_grad in [
        (1000.0, 1.0, 0.0, 0.0),
        (-1000.0, 0.0, 0.0, 0.0),
        (1000.0, 0.0, 1000.0, 1.0),
        (-1000.0, 1.0, 1000.0, -1.0),
    ]:
        z = Value(value)
        loss = binary_cross_entropy_with_logits([z], [target])
        assert math.isfinite(loss.data)
        assert loss.data == pytest.approx(expected_loss, abs=1e-10)
        loss.backward()
        assert math.isfinite(z.grad)
        assert z.grad == pytest.approx(expected_grad, abs=1e-10)


def test_bce_rejects_invalid_inputs() -> None:
    bad_cases = [
        ([], []),
        ([Value(0.0)], []),
        ([Value(0.0)], [0.4]),
        ([Value(float("nan"))], [0.0]),
        ([Value(float("inf"))], [1.0]),
        ([Value(0.0)], [float("nan")]),
    ]
    for logits, labels in bad_cases:
        with pytest.raises(ValueError):
            binary_cross_entropy_with_logits(logits, labels)


def test_training_loop_updates_and_logs_correct_steps() -> None:
    model = MLP([2, 4, 1], hidden_activation="relu", output_activation="linear", seed=11)
    before = exp.evaluate(model)["bce"]
    history = exp.train_full_batch(
        model, loss_kind="bce", steps=100, learning_rate=0.05, log_every=25
    )
    after = exp.evaluate(model)["bce"]
    assert [h["step"] for h in history] == [0, 25, 50, 75, 100]
    assert history[0]["bce"] == pytest.approx(before)
    assert history[-1]["bce"] == pytest.approx(after)
    assert after < before


def test_mse_variant_still_propagates_gradients() -> None:
    model = MLP([2, 4, 1], hidden_activation="relu", output_activation="linear", seed=11)
    before = exp.evaluate(model)["brier"]
    exp.train_full_batch(model, loss_kind="mse", steps=100, learning_rate=0.05, log_every=25)
    after = exp.evaluate(model)["brier"]
    assert after < before
