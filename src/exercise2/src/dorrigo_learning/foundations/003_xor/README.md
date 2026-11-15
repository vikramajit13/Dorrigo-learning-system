# Experiment 003 — Can an MLP learn XOR?

## Research question

Can a small ReLU MLP learn XOR while an all-linear MLP cannot, using the same four training examples and your own scalar autodiff engine?

## XOR dataset (binary inputs, real-valued targets)

| x1 | x2 | y |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single linear decision boundary cannot represent XOR. More strongly, *any composition of affine/linear layers* is still affine: `f(0,0) + f(1,1) = f(0,1) + f(1,0)`. XOR labels violate this identity (0 != 2).

## Architecture and initial experiment

- Input: 2 values.
- Hidden layer: 4 ReLU neurons.
- Output: 1 **linear** neuron (we're regressing toward targets 0 and 1 using MSE).
- Architecture: `MLP([2, 4, 1], hidden_activation="relu", output_activation="linear", seed=11)`.
- Control: the same MLP with `hidden_activation="linear"`.
- Optimizer: your plain SGD, full batch (all four examples per update).
- Initial sweep: 3000 steps, learning rate 0.05. **These are hypotheses, not magic settings.**

## What you implement

Open `train.py`, then complete the `train_full_batch(...)` function:

1. Create **fresh** predictions from the model for all four samples.
2. Call your `mse(...)` to get one scalar `Value` loss.
3. Call `loss.backward()` **once per batch**.
4. Call `sgd_step(model.parameters(), learning_rate)`.
5. At logging intervals, evaluate the **updated** model to report its loss.

Do not call `.data` while constructing the loss. `.data` is for reporting numerical values and performing optimizer updates.

## Running

```bash
python -m pytest -q tests/test_xor.py
python experiments/003_xor/train.py --seed 11 --steps 3000 --lr 0.05
# Optional: compare multiple random initializations
python experiments/003_xor/train.py --seeds 11,42 --steps 3000 --lr 0.05
```

The CLI trains and reports the ReLU model **and** its all-linear control with matching seed, steps and learning rate. JSON summaries are written to `results/` (generated at runtime).

## Evidence of learning

- For a successful ReLU run, aim for training MSE < 0.02, and each of the four predictions within 0.2 of its target.
- The all-linear model should not achieve the same result; with this full XOR dataset and MSE, the best possible affine predictor has MSE = 0.25 (the constant 0.5).
- Results can vary by seed and initialization. An unsuccessful ReLU run is an observation, not proof your autograd is broken. Investigate dead ReLUs and changes in gradient magnitudes.
- The four XOR inputs are *all* the training examples. Fitting them is **not evidence of generalization**. We'll study generalization with separate data in later experiments.

## Explain in your own words

1. Why is the linear model unable to fit XOR, even with multiple linear layers?
2. Why can a ReLU hidden layer overcome this restriction?
3. Why must the loss remain a `Value` (not `Value(numeric_loss)`) for gradients to flow?
4. Why are some training runs sensitive to random initialization?
5. What does the same training dataset in evaluation tell us—and not tell us?
