"""Exercise 002: neural-network primitives built exclusively on your Value class.

Implement TODOs yourself. Do not import PyTorch, NumPy or an ML framework.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from .autograd import Value


class Neuron:
    def __init__(
        self,
        n_inputs: int,
        *,
        activation: str = "relu",
        rng: random.Random | None = None,
    ) -> None:
        if n_inputs < 1:
            raise ValueError("n_inputs must be positive")
        if activation not in ("relu", "linear"):
            raise ValueError("activation must be 'relu' or 'linear'")
        self.activation = activation
        rng = rng if rng is not None else random.Random(42)
        scale = n_inputs ** -0.5
        self.w = [Value(rng.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, inputs: Sequence[Value | float]) -> Value:
        # TODO: validate len(inputs) == len(self.w).
        # TODO: z = sum(w_i * x_i) + b, all via Value operations.
        # TODO: apply relu() if activation == 'relu'; otherwise return z.
        if len(inputs) != len(self.w):
            raise ValueError(f"Expected {len(self.w)} inputs, got {len(inputs)}")
        z = sum(w * x for w, x in zip(self.w, inputs)) + self.b
        if self.activation == "relu":
            return z.relu()
        else:
            return z
        

    def parameters(self) -> list[Value]:
        # TODO: return the trainable weights and bias.
        return self.w + [self.b]
        

class Layer:
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        *,
        activation: str = "relu",
        rng: random.Random | None = None,
    ) -> None:
        if n_outputs < 1:
            raise ValueError("n_outputs must be positive")
        # TODO: create n_outputs independent Neuron objects.
        rng = rng if rng is not None else random.Random(42)
        self.neurons = [
            Neuron(n_inputs=n_inputs, activation=activation, rng=rng)
            for _ in range(n_outputs)
        ]
        # Pass the SAME rng object into each Neuron so their initializations differ.

    def __call__(self, inputs: Sequence[Value | float]) -> list[Value]:
        # TODO: forward each neuron and return all its outputs.
        Neuron_outputs = [neuron(inputs) for neuron in self.neurons]
        return Neuron_outputs

    def parameters(self) -> list[Value]:
        # TODO: flatten parameters from every neuron.
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
    def __init__(
        self,
        sizes: Sequence[int],
        *,
        hidden_activation: str = "relu",
        output_activation: str = "linear",
        seed: int = 42,
    ) -> None:
        if len(sizes) < 2 or any(n < 1 for n in sizes):
            raise ValueError("sizes must contain at least two positive integers")
        # TODO: use rng = random.Random(seed) for repeatable initialization.
        # TODO: build Layer(sizes[i], sizes[i+1]) for each consecutive pair.
        # Use hidden_activation on all but the final layer; use output_activation
        # on the final layer (regression output should usually be linear).
        rng = random.Random(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            activation = (
                hidden_activation if i < len(sizes) - 2 else output_activation
            )
            self.layers.append(
                Layer(n_inputs=sizes[i], n_outputs=sizes[i + 1], activation=activation, rng=rng)
            )
        # raise NotImplementedError

    def __call__(self, inputs: Sequence[Value | float]) -> list[Value]:
        # TODO: feed the output list of each layer to the next layer.
        current: Sequence[Value | float] = inputs
        for layer in self.layers:
            current = layer(current)
        return [v if isinstance(v, Value) else Value(v) for v in current]

    def parameters(self) -> list[Value]:
        # TODO: collect all layer parameters.
        return [param for layer in self.layers for param in layer.parameters()] 
