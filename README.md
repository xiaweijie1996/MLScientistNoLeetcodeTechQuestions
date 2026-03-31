# MLScientistNoLeetcodeTechQuestions

Practical, non-LeetCode interview preparation for ML Scientist and Research Engineer roles.

This repo focuses on the kinds of problems that show up in real ML interviews: PyTorch correctness, autograd internals, numerical stability, attention implementations, mixed precision, debugging, and implementation tradeoffs.

## Table of Contents

- [What This Repo Helps You Practice](#what-this-repo-helps-you-practice)
- [Start Here](#start-here)
- [Recommended Study Paths](#recommended-study-paths)
- [Example Index in `src`](#example-index-in-src)
- [How To Use The Examples](#how-to-use-the-examples)
- [Project Goal](#project-goal)
- [Notes](#notes)

## What This Repo Helps You Practice

- PyTorch tensor and gradient mechanics
- Autograd behavior and custom reasoning
- Numerical stability and softmax/log-softmax details
- Data loading and training loop basics
- CNN and transformer implementation patterns
- Grouped-query and multi-query attention concepts
- Mixed precision and practical debugging habits

## Start Here

### Core guides

1. [overall-explain.md](overall-explain.md)  
   Full interview map, category breakdowns, and question bank.

2. [one-week-fast-plan.md](one-week-fast-plan.md)  
   A compact 7-day preparation plan with daily deliverables.

## Recommended Study Paths

### Fast path

1. Read [one-week-fast-plan.md](one-week-fast-plan.md).
2. Pick one example from [`src`](src) each day.
3. Use [overall-explain.md](overall-explain.md) when you want deeper reference material.
4. After each exercise, write down:
   - what failed,
   - why it failed,
   - how you fixed it.

### Deep path

1. Start with [overall-explain.md](overall-explain.md).
2. Choose one topic area such as autograd, numerics, attention, or performance.
3. Pair that topic with one or two examples from [`src`](src).
4. Re-implement a variation yourself and explain the tradeoffs out loud.

## Example Index in `src`

This table covers the example notebooks and scripts currently under [`src`](src).

| Area | Example | Type | Main Link | Focus |
|---|---|---|---|---|
| Autograd | Activation function autograd | Notebook | [src/autograde-activationfunction/autograde.ipynb](src/autograde-activationfunction/autograde.ipynb) | Gradient flow and autograd reasoning around activation functions |
| Autograd | Parameter definition autograd | Notebook | [src/autograde-defineparameters/autograde.ipynb](src/autograde-defineparameters/autograde.ipynb) | How parameters participate in autograd and optimization |
| Gradients | Basic gradient operations | Notebook | [src/basicgradoperation/basicgrad.ipynb](src/basicgradoperation/basicgrad.ipynb) | Intro gradient calculations and inspection |
| Tensor basics | Basic gradient notebook | Notebook | [src/basicoperation/basicgrad.ipynb](src/basicoperation/basicgrad.ipynb) | Foundational tensor and gradient exercises |
| Tensor basics | Basic operations | Notebook | [src/basicoperation/basicoperation.ipynb](src/basicoperation/basicoperation.ipynb) | Core tensor operations and behavior |
| Tensor basics | Demo basics | Notebook | [src/basicoperation/demobasic.ipynb](src/basicoperation/demobasic.ipynb) | Intro walkthrough and sanity-check examples |
| Vision | CNN baseline | Notebook | [src/cnn/cnn.ipynb](src/cnn/cnn.ipynb) | CNN training workflow and debugging |
| Vision | MNIST raw dataset files | Data | [src/cnn/data/MNIST/raw](src/cnn/data/MNIST/raw) | Local dataset files used by the CNN notebook |
| Data pipeline | Dataloader practice | Notebook | [src/dataloader/dataloader.ipynb](src/dataloader/dataloader.ipynb) | Dataset and dataloader patterns |
| Data pipeline | Sample CSV | Data | [src/dataloader/data.csv](src/dataloader/data.csv) | Small dataset for pipeline exercises |
| Attention | Grouped-query attention | Notebook | [src/groupattention/groupqueryattention.ipynb](src/groupattention/groupqueryattention.ipynb) | Grouped-query attention concepts and implementation |
| Attention | Grouped-query attention | Python script | [src/groupattention/groupqueryattention.py](src/groupattention/groupqueryattention.py) | Script version of grouped-query attention logic |
| Attention | Multi-query attention | Notebook | [src/groupattention/multiqueryattention.ipynb](src/groupattention/multiqueryattention.ipynb) | Multi-query attention concepts and implementation |
| Attention | Multi-query attention | Python script | [src/groupattention/multiqueryattention.py](src/groupattention/multiqueryattention.py) | Script version of multi-query attention logic |
| Precision | Mixed precision | Notebook | [src/mixed-precision/mixedprecision.ipynb](src/mixed-precision/mixedprecision.ipynb) | AMP and precision tradeoffs in training |
| Model basics | Simple neural network | Notebook | [src/simplennmodel/simplenn.ipynb](src/simplennmodel/simplenn.ipynb) | Model-building and training loop fundamentals |
| Numerics | Stable log-softmax | Notebook | [src/stablesoftmax/stablelogsoftmax.ipynb](src/stablesoftmax/stablelogsoftmax.ipynb) | Numerical stability for softmax and log-softmax |
| Numerics | Softmax playground | Python script | [src/stablesoftmax/play_softmax.py](src/stablesoftmax/play_softmax.py) | Quick comparisons and experiments for softmax behavior |
| Transformer | Basic transformer | Notebook | [src/transformer-basic/transformer.ipynb](src/transformer-basic/transformer.ipynb) | Attention, masking, and sequence-model implementation details |

## How To Use The Examples

- Start with notebooks in this order if you want a smooth ramp:
  1. `basicoperation`
  2. `basicgradoperation`
  3. `autograde-*`
  4. `dataloader`
  5. `simplennmodel`
  6. `cnn`
  7. `stablesoftmax`
  8. `mixed-precision`
  9. `transformer-basic`
  10. `groupattention`
- Treat each notebook like interview practice, not just code execution.
- For every example, be able to explain:
  - what the code is doing,
  - where it can fail,
  - what tradeoffs the implementation makes,
  - how you would debug it under interview pressure.

## Project Goal

Build interview-ready ML engineering judgment, not just coding speed:

- explain why a training run fails,
- reason about gradients and numerics,
- choose practical PyTorch fixes under constraints,
- communicate implementation tradeoffs clearly.

## Notes

- The repo is practice-focused and notebook-heavy.
- Some datasets are stored locally for convenience, especially under [src/cnn/data/MNIST/raw](src/cnn/data/MNIST/raw).
- If a notebook is long, work through it section by section and keep your own short notes after each section.
