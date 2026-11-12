"""Exercise 002 Part A: implement the training loop (no frameworks)."""
from dorrigo_learning.foundations.nn import Neuron
from dorrigo_learning.foundations.optim import mse, sgd_step

DATA = [(-2.0, -3.0), (-1.0, -1.0), (0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]


def main() -> None:
    model = Neuron(1, activation="linear")
    # TODO: implement a full-batch training loop.
    # 1. Build fresh predictions for all DATA inputs.
    # 2. loss = mse(predictions, targets)
    # 3. loss.backward() ONCE.
    # 4. sgd_step(model.parameters(), learning_rate=...)
    # 5. print loss periodically; evaluate model on [3.0] and [-3.0].
    raise NotImplementedError("Implement your full-batch linear training loop")


if __name__ == "__main__":
    main()
