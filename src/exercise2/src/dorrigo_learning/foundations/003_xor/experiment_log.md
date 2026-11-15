# XOR experiment log

## Hypothesis
What do you expect to happen for a ReLU MLP versus an all-linear MLP? Why?

## Controlled settings
- Architecture:
- Hidden activation:
- Output activation:
- Seed(s):
- Learning rate:
- Updates:
- Loss function:
- Dataset split: All four examples used for training. No held-out set.

## Results
| Run | Initial MSE | Final MSE | Predictions in order 00,01,10,11 | Notes |
|---|---:|---:|---|---|
| ReLU, seed __ | | | | |
| Linear, seed __ | | | | |

## Investigation
1. Did all four outputs cross the right side of the 0.5 classification threshold?
2. Did the ReLU model reach MSE < 0.02? If not, what evidence explains the failure?
3. Why does the linear network fail to represent XOR even if trained for longer?
4. Compare two seeds. Did they converge differently? What does that suggest about initialization?
5. What would count as generalization evidence (which this four-point experiment does NOT provide)?

## Next experiment
State one planned controlled change and the outcome you predict.
