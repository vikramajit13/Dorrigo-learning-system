# Dorrigo Learning Systems · Exercise 002
## Build and train a neural network with your own autodiff engine

This is an **add-on**, not a replacement repository. It does not include or change
`src/dorrigo_learning/foundations/autograd.py`; **keep your completed `Value` class**.

### Install in your existing repository

1. Unzip this archive into the **root** of your existing
   `dorrigo-learning-systems-starter` repository. Merge the `src/`, `tests/`, and
   `experiments/` directories. Do not replace your existing files.
2. Activate your existing Python virtual environment.
3. Run the new tests: `python -m pytest -q tests/test_nn.py`.
   They should initially fail because the implementation has TODOs.
4. Work in this order: `Neuron` -> `Layer` -> `MLP` -> `mse` -> `sgd_step`.
5. Then execute `python experiments/002_neural_network/train_linear.py`.
6. Finally execute `python experiments/002_neural_network/train_xor.py`.

### The mathematical foundation

Neuron: `z = sum(w_i * x_i) + b`. A linear neuron returns `z`; a ReLU neuron
returns `max(0,z)`. Every computation must use `Value` operations so the graph
is preserved.

For N predictions, `MSE = (1/N) sum((prediction_i - target_i)^2)`.
Your `Value` engine doesn't yet implement subtraction. Use `prediction + (-1.0 * target)`;
likewise, a mean can be obtained by multiplying by `1.0/N`.

Gradient descent: `parameter.data -= learning_rate * parameter.grad`.
**Important:** Use the sum/mean of every sample's losses, then call `loss.backward()`
**once per training step**. Your current `Value.backward()` implementation zeroes
gradients in its graph; calling backward separately per sample would overwrite
previous per-sample gradients.

### API contracts

- `Neuron(n_inputs, *, activation='relu', rng=None)` stores trainable `w` (list of
  `Value`) and `b` (`Value`), and `__call__(inputs)` returns a scalar `Value`.
- `Layer(n_inputs, n_outputs, *, activation='relu', rng=None)` returns `list[Value]`.
- `MLP(sizes, *, hidden_activation='relu', output_activation='linear', seed=42)`
  returns `list[Value]` even when there is one output. `sizes=[2,4,1]` means
  two features, four hidden neurons, one output neuron.
- All modules implement `.parameters() -> list[Value]` with no duplicates.
- `mse(predictions, targets) -> Value` takes equally long nonempty sequences.
- `sgd_step(parameters, learning_rate) -> None` updates `.data` in place.

### Acceptance gates

- `python -m pytest -q` passes your original autograd tests and the new tests.
- Linear regression predicts `y=2*x+1` on training and held-out inputs.
- Your MLP can fit binary XOR (input [0/1,0/1], target [0/1]).
- You can explain: (1) what parameters mean, (2) why gradients are accumulated,
  (3) why a stack of linear layers remains a linear function, (4) why ReLU allows
  XOR, and (5) why a training result is not evidence of generalization.
- Write an experiment note documenting learning rate, seed, initial/final loss,
  predictions, a failure you saw, and one improvement you'd test next.

**Apple Silicon 16 GB:** this exercise is CPU-only, requires no model downloads
and no extra ML package. Run everything in the local Python environment.
