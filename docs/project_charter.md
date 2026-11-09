# Dorrigo Learning Systems — Project Charter

## Mission
Understand, train, evaluate and improve intelligent models. Build robust systems to support experiments and, later, autonomous task environments.

## North-star capability
Form a testable hypothesis about model behavior; implement or train a model; evaluate it fairly; analyze failures; communicate findings reproducibly.

## Principles
1. Depth before breadth. Implement mechanisms yourself before relying on high-level abstractions.
2. Measurements over anecdotes. Use baselines, held-out data and explicit metrics.
3. Understand failures. Preserve negative results and explain uncertainties.
4. Reproducibility. Track data version, split, seed, config, model version, hardware and environment.
5. Stay resource-conscious. Verify correctness on CPU/tiny models before paying for GPU compute.
6. No unnecessary agent framework. Build an agent runtime only to answer a research question.
7. One project, many connected experiments. No artificial timelines; progress by mastery.

## Current focus
Stage 1: reverse-mode automatic differentiation, gradients, optimization, neural-network training.

## Not in scope now
MCP tools, workflows for real insurance claims, production UI, distributed services, multi-agent orchestration, vendor framework comparisons.
