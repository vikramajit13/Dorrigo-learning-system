"""Exercise 004: binary cross-entropy with logits built on the scalar Value engine.

YOUR TASK: implement the TODO without using NumPy, PyTorch or an ML framework.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

from .autograd import Value


def binary_cross_entropy_with_logits(
    logits: Sequence[Value], targets: Sequence[float]
) -> Value:
    """Return mean BCE-with-logits as a Value connected to every input logit.

    TODO:
      1. Validate equal, non-zero lengths; targets exactly 0.0 or 1.0.
      2. Reject non-finite logit and target values.
      3. For each (logit, target), compute a numerically stable loss:
           max(z, 0) - z * y + log1p(exp(-abs(z)))
         using `math` on logit's numerical value.
      4. Make an output Value with `_children=(logit,)`, and define its
         `_backward` to add upstream_grad * (sigmoid(z) - target) to logit.grad.
         Make sure each closure captures the correct logit and target!
      5. Return the MEAN of all per-example Value losses. Use Value arithmetic
         to preserve graph connections (and an explicit Value sum initializer).

    Think: this is a legitimate primitive custom operation, unlike computing a
    number and returning a disconnected Value(...).
    """
    
    if len(logits) != len(targets) or len(logits) == 0:
        raise ValueError("Logits and targets must have the same non-zero length.")
    for logit in logits:
        if not isinstance(logit, Value):
            raise ValueError("Logits must be Value instances.")
        if not math.isfinite(logit.data):
            raise ValueError("Logit values must be finite.")
    for target in targets:
        if target not in (0.0, 1.0):
         raise ValueError("Each target must be 0.0 or 1.0.")
        if not isinstance(target, float):
            raise ValueError("Targets must be float instances.")
        if not math.isfinite(target):
            raise ValueError("Target values must be finite.")

    # Compute the per-example losses and accumulate them.
         
    losses: list[Value] = []

    for logit, target in zip(logits, targets):
         z = logit.data

         loss_val = (
            max(z, 0.0)
            - z * target
            + math.log1p(math.exp(-abs(z)))
         )

         out = Value(loss_val, _children=(logit,))

         def _backward(
            out: Value = out,
            logit: Value = logit,
            target: float = target,
            z: float = z,
         ) -> None:
            if z >= 0:
                  sig = 1.0 / (1.0 + math.exp(-z))
            else:
                  exp_z = math.exp(z)
                  sig = exp_z / (1.0 + exp_z)

            logit.grad += out.grad * (sig - target)

         out._backward = _backward
         losses.append(out)

    total_loss = sum(losses, Value(0.0))

    return total_loss * (1.0 / len(losses))


