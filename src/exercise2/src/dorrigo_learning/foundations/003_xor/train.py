"""Exercise 003: Train an MLP on XOR and compare against a linear baseline.

Implement train_full_batch. All logging, CLI parsing, comparison and JSON output
are supplied so you can focus on the training algorithm.

Run from the repository root:
    python experiments/003_xor/train.py --seed 11 --steps 3000 --lr 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from dorrigo_learning.foundations.nn import MLP
from dorrigo_learning.foundations.optim import mse, sgd_step

class Evaluation(TypedDict):
    mse: float
    predictions: list[float]
    classification_accuracy: float
    within_0_2: bool


class HistoryEntry(TypedDict):
    step: int
    mse: float


class Report(TypedDict):
    activation: str
    seed: int
    architecture: list[int]
    steps: int
    learning_rate: float
    before: Evaluation
    after: Evaluation
    history: list[HistoryEntry]


XOR_DATA: tuple[tuple[tuple[float, float], float], ...] = (
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
)


def evaluate(model: MLP) -> Evaluation:
    """Evaluate training examples *without* backpropagating through this graph."""
    inputs = [x for x, _ in XOR_DATA]
    targets = [y for _, y in XOR_DATA]
    predictions = [model(x)[0] for x in inputs]
    loss = mse(predictions, targets)
    floats = [p.data for p in predictions]
    return {
        "mse": loss.data,
        "predictions": floats,
        "classification_accuracy": sum(
            int((value >= 0.5) == (target == 1.0))
            for value, target in zip(floats, targets)
        ) / len(targets),
        "within_0_2": all(
            abs(value - target) < 0.2
            for value, target in zip(floats, targets)
        ),
    }


def train_full_batch(
    model: MLP,
    *,
    steps: int,
    learning_rate: float,
    log_every: int = 500,
) -> list[HistoryEntry]:
    """Return a log of {'step': int, 'mse': float} for the updated model.

    TODO — YOUR IMPLEMENTATION:
      for each step:
        - compute fresh predictions for *all* XOR inputs
        - obtain one scalar Value loss by calling mse(predictions, targets)
        - call loss.backward() once
        - update every model parameter using sgd_step(...)
        - log the updated model's mse periodically, including the last step

    Hint: in every iteration, rebuild the graph. Do not keep a reference to a
    loss from an earlier parameter update. Include a step-0 entry if useful.
    """
    if steps <= 0 or log_every <= 0 or learning_rate <= 0:
        raise ValueError("steps, log_every, and learning_rate must be positive")

    raise NotImplementedError("Implement train_full_batch for Exercise 003")


def run_one(*, activation: str, seed: int, steps: int, lr: float) -> Report:
    model = MLP(
        [2, 4, 1],
        hidden_activation=activation,
        output_activation="linear",
        seed=seed,
    )
    before = evaluate(model)
    history = train_full_batch(
        model, steps=steps, learning_rate=lr, log_every=max(1, steps // 6)
    )
    after = evaluate(model)
    return {
        "activation": activation,
        "seed": seed,
        "architecture": [2, 4, 1],
        "steps": steps,
        "learning_rate": lr,
        "before": before,
        "after": after,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Optional comma-separated list, e.g. 11,42,7",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    if not seeds:
        parser.error("at least one seed is required")

    results_path = Path(__file__).resolve().parent / "results"
    results_path.mkdir(exist_ok=True)

    for seed in seeds:
        for activation in ("relu", "linear"):
            print(f"\nTraining {activation} network (seed={seed})")
            report = run_one(activation=activation, seed=seed, steps=args.steps, lr=args.lr)
            before = report["before"]
            after = report["after"]
            print(f"  Initial MSE: {before['mse']:.6f}")
            print(f"  Final MSE:   {after['mse']:.6f}")
            print(f"  Accuracy:    {after['classification_accuracy']:.0%}")
            for (x, y), prediction in zip(XOR_DATA, after["predictions"]):
                print(f"  x={x}  expected={y:.0f}  predicted={prediction:.4f}")
            report_file = results_path / f"xor_{activation}_seed_{seed}.json"
            report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"  Saved: {report_file}")


if __name__ == "__main__":
    main()
