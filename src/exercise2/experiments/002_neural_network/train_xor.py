"""Exercise 002 Part B: implement XOR training with your MLP."""
from dorrigo_learning.foundations.nn import MLP
from dorrigo_learning.foundations.optim import mse, sgd_step

DATA = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]


def main() -> None:
    model = MLP([2, 4, 1], seed=42)
    # TODO: build the full-batch training loop. Extract output with model(x)[0].
    # Start with 1000 steps and learning rate 0.05; alter these after measuring.
    # Print initial and final loss and a prediction for each input in DATA.
    # Repeat with an all-linear model: why does it fail to fit XOR?
    raise NotImplementedError("Implement your XOR experiment")


if __name__ == "__main__":
    main()
