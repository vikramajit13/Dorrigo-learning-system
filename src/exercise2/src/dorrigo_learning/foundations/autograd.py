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
import math


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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other))
        def _backward() -> None:
            # d(self + other)/dself = 1, d(self + other)/dother = 1
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        
        out._backward = _backward
        return out
        #raise NotImplementedError("Implement Value.__add__")

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        # TODO: forward result, graph parents, and gradient contributions.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other))

        def _backward() -> None:
            # d(self * other)/dself = other, d(self * other)/dother = self
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __pow__(self, exponent: float) -> Value:
        # TODO: implement scalar-power derivative.
        out = Value(self.data ** exponent, _children=(self,))
        def _backward() -> None:
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))
        out._backward = _backward
        return out

    def relu(self) -> Value:
        # TODO: ReLU forward pass and local derivative.
        out = Value(max(0, self.data), _children=(self,))
        def _backward() -> None:
            self.grad += out.grad * (1 if self.data > 0 else 0)
        out._backward = _backward
        return out


    def exp(self) -> Value:
        """Exponential function: y = e^x."""

        result = math.exp(self.data)

        out = Value(result, _children=(self,))

        def _backward() -> None:
            # d(e^x)/dx = e^x
            self.grad += out.grad * out.data

        out._backward = _backward
        return out


    def sigmoid(self) -> Value:
        """Numerically stable sigmoid: 1 / (1 + e^-x)."""

        if self.data >= 0:
            z = math.exp(-self.data)
            result = 1.0 / (1.0 + z)
        else:
            z = math.exp(self.data)
            result = z / (1.0 + z)

        out = Value(result, _children=(self,))

        def _backward() -> None:
            # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            self.grad += out.grad * out.data * (1.0 - out.data)

        out._backward = _backward
        return out

    def backward(self) -> None:
        # TODO: graph traversal, seed gradient=1, reverse-mode accumulation.
        ordered_nodes: list[Value] = []
        visited: set[Value] = set()
        
        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                ordered_nodes.append(v)

        build_topo(self)
        # Reverse-mode accumulation.
        # 2. Reset gradients and start the backward pass
        for node in ordered_nodes:
            node.grad = 0.0
            
        # The derivative of the output with respect to itself is always 1
        self.grad = 1.0 
        
        # Traverse backwards through the sorted computational graph
        for node in reversed(ordered_nodes):
            node._backward()
