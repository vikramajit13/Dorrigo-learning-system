"""Exercise 002: loss function and SGD implemented with Value objects."""
from __future__ import annotations

from collections.abc import Sequence
from .autograd import Value


def mse(predictions: Sequence[Value], targets: Sequence[float]) -> Value:
    """Mean squared error; use Value math so gradients flow to predictions."""
    # TODO: ensure equal nonzero lengths; raise ValueError otherwise.
    # TODO: return mean((prediction - target)**2), with subtraction expressed
    # as prediction + (-1.0 * target) if you haven't added __sub__ to Value.
    if len(predictions) != len(targets) or len(predictions) == 0:
        raise ValueError("Predictions and targets must have the same nonzero length.")
    squared_errors = [
        (p + (-1.0 * t)) ** 2
        for p, t in zip(predictions, targets)
    ]

    return sum(squared_errors, Value(0.0)) * (1.0 / len(predictions))


def sgd_step(parameters: Sequence[Value], learning_rate: float) -> None:
    """Simple gradient descent, without momentum or an optimizer library."""
    # TODO: ensure learning_rate > 0; raise ValueError otherwise.
    # TODO: mutate each parameter.data using its gradient.
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    for p in parameters:
        p.data += -learning_rate * p.grad
        p.grad = 0.0
