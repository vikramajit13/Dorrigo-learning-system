# Experiment 001 — Build reverse-mode automatic differentiation

## Research/engineering question
How does a neural-network framework compute gradients efficiently when a value influences the output through multiple computation paths?

## Build
Complete `src/dorrigo_learning/foundations/autograd.py`:
- scalar addition and multiplication, including Python-number operands;
- power, ReLU;
- tracking graph dependencies;
- reverse-mode gradient propagation through a topologically sorted graph;
- correct accumulation at shared nodes.

## Prove correctness
Run `python -m pytest -q` and make all tests pass. Then create at least two additional tests of your own, including one that checks gradient accumulation at a reused node.

## Understand
Answer in your own words:
1. What is the chain rule and how does backpropagation use it?
2. Why must gradients *accumulate* at nodes reached by different paths?
3. Why does reverse topological order matter?
4. What does the numerical central-difference test verify, and what are its limitations?
5. What changes when we move from scalars to tensors?

## Exit gate
- All automated tests green.
- Independently derive gradients of `f(x,y)=x²+3xy+y²` at x=1.5 and y=-2.
- Draw the computational graph for `z=x*x+x*y` and explain the gradient on `x`.
- Describe one failure case you found during implementation.

## After you pass
Implement a tiny two-layer MLP and train it on a simple synthetic classification dataset. Only then progress toward a bigram LM and transformer.

## Learning reference
Karpathy's Micrograd lesson and repository are useful to compare against *after* attempting your own implementation: https://karpathy.ai/zero-to-hero.html
