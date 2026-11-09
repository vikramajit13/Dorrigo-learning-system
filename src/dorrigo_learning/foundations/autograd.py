"""Exercise 001: reverse-mode autodiff for scalar values.

Implement the TODO methods without importing a neural-network framework.

You will need to:
- track the parents and local backward function of each operation;
- build a topological ordering of the computation graph;
- traverse in reverse topological order to accumulate gradients;
- correctly handle a value that appears in multiple graph branches.

This is intentionally a scaffold, not a completed solution.
"""

from __future__ import annotations


class Value:
    def __init__(self, data: float, *, _children: tuple[Value, ...] = ()) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Value | float) -> Value:
        # TODO: forward result, graph parents, and gradient contributions.
        raise NotImplementedError("Implement Value.__add__")

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        # TODO: forward result, graph parents, and gradient contributions.
        raise NotImplementedError("Implement Value.__mul__")

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __pow__(self, exponent: float) -> Value:
        # TODO: implement scalar-power derivative.
        raise NotImplementedError("Implement Value.__pow__")

    def relu(self) -> Value:
        # TODO: ReLU forward pass and local derivative.
        raise NotImplementedError("Implement Value.relu")

    def backward(self) -> None:
        # TODO: graph traversal, seed gradient=1, reverse-mode accumulation.
        raise NotImplementedError("Implement Value.backward")
