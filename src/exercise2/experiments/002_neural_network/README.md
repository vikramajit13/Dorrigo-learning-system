# Exercise 002 — Learn from gradients

## Part A — Linear regression

Build `Neuron(1, activation="linear")` and train on
`x = [-2,-1,0,1,2]` with labels `y = 2*x + 1`.

A full-batch training step is:
1. Construct predictions with a **fresh forward pass** for all five samples.
2. Compute a **single** mean squared error over all predictions.
3. Call `.backward()` on that one loss.
4. Update every weight and bias using `sgd_step`.

Evaluate on inputs not in your training set, such as `x=3` and `x=-3`.

## Part B — XOR

Train `MLP([2, 4, 1], hidden_activation="relu", output_activation="linear")`
on four examples:

| x1 | x2 | target |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Start with 1,000 full-batch steps and learning rate 0.05. Try another seed,
width or learning rate when training stalls; not every ReLU initialization
will converge. Treat this as optimization investigation, not a race to zero loss.

Try removing all hidden ReLU activations. What does that do to the decision
function? Why can't stacked affine transformations express XOR perfectly?

## Experiment notes

Record seed, architecture, optimizer, learning rate, training steps, initial
loss, final loss, four predictions and one failed experiment. Explain what
changed and why. A tiny fixed dataset is a **memorization/expressivity test**;
it does not establish generalization.

No ML dependencies and no GPU needed on your 16 GB Mac.
