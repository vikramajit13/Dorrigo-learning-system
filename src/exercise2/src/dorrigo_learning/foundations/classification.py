"""Exercise 004: binary cross-entropy with logits built on the scalar Value engine.

YOUR TASK: implement the TODO without using NumPy, PyTorch or an ML framework.
"""
from __future__ import annotations

from collections.abc import Sequence

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
    raise NotImplementedError("Implement Exercise 004 stable BCE-with-logits")
