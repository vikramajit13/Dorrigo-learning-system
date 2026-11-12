"""Acceptance tests for Exercise 001. Expected to fail before implementation."""

import pytest

from exercise2.src.dorrigo_learning.foundations.autograd import Value


def test_forward_arithmetic() -> None:
    x, y = Value(2.0), Value(-3.0)
    z = x * y + x**2
    assert z.data == pytest.approx(-2.0)


def test_gradients_with_reused_value() -> None:
    x, y = Value(2.0), Value(3.0)
    z = x * x + x * y
    z.backward()
    assert x.grad == pytest.approx(7.0)
    assert y.grad == pytest.approx(2.0)


def test_scalar_operands() -> None:
    x = Value(3.0)
    z = 2.0 * x + 4.0
    z.backward()
    assert z.data == pytest.approx(10.0)
    assert x.grad == pytest.approx(2.0)


def test_relu_blocks_negative_branch() -> None:
    x = Value(-2.0)
    z = x.relu()
    z.backward()
    assert z.data == pytest.approx(0.0)
    assert x.grad == pytest.approx(0.0)


def test_gradients_against_finite_differences() -> None:
    def f(a: float, b: float) -> float:
        return a * a + 3.0 * a * b + b * b

    a, b = Value(1.5), Value(-2.0)
    z = a * a + 3.0 * a * b + b * b
    z.backward()
    epsilon = 1e-5
    da = (f(1.5 + epsilon, -2.0) - f(1.5 - epsilon, -2.0)) / (2 * epsilon)
    db = (f(1.5, -2.0 + epsilon) - f(1.5, -2.0 - epsilon)) / (2 * epsilon)
    assert a.grad == pytest.approx(da, abs=1e-5)
    assert b.grad == pytest.approx(db, abs=1e-5)
