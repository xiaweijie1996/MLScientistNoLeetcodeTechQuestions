# MLScientistNoLeetcodeTechQuestions

Practical, non-LeetCode interview preparation for ML Scientist and Research Engineer roles.

This repo focuses on the questions that show up in real ML interviews: PyTorch correctness, autograd internals, numerical stability, model debugging, and implementation tradeoffs.

## What Is Inside

## Two Guides (Start Here)

1. [overall-explain.md](overall-explain.md)
Purpose: full PyTorch interview map, category breakdowns, and question bank.
Use this when you want broad coverage and deeper reference material.

2. [one-week-fast-plan.md](one-week-fast-plan.md)
Purpose: a compressed 7-day practice plan with daily deliverables.
Use this when you want a fast, execution-focused preparation sprint.

## Real Case Practice Files

### Autograd and gradient mechanics
- [src/autograde/autograde.ipynb](src/autograde/autograde.ipynb)
Focus: custom autograd functions, gradient reasoning, backward checks.

### CNN and vision baseline
- [src/cnn/cnn.ipynb](src/cnn/cnn.ipynb)
Focus: CNN training workflow and model debugging.
- [src/cnn/data/MNIST/raw](src/cnn/data/MNIST/raw)
Local MNIST raw files used by the notebook.

### Data pipeline basics
- [src/dataloader/dataloader.ipynb](src/dataloader/dataloader.ipynb)
Focus: dataset and dataloader patterns.
- [src/dataloader/data.csv](src/dataloader/data.csv)
Sample data file for pipeline exercises.

### Simple neural network implementation
- [src/simplennmodel/simplenn.ipynb](src/simplennmodel/simplenn.ipynb)
Focus: core model-building patterns and training loop fundamentals.

### Numerical stability (softmax)
- [src/stablesoftmax/stablelogsoftmax.ipynb](src/stablesoftmax/stablelogsoftmax.ipynb)
Focus: stable softmax and log-softmax behavior.
- [src/stablesoftmax/play_softmax.py](src/stablesoftmax/play_softmax.py)
Quick script to test and compare softmax behavior.

### Transformer case study
- [src/transformer/transformer.ipynb](src/transformer/transformer.ipynb)
Focus: attention, masking, contiguous vs view semantics, and sequence modeling implementation details.

## How To Use This Repo

## Recommended path (first-time user)

1. Read [one-week-fast-plan.md](one-week-fast-plan.md) and pick your daily schedule.
2. Use [overall-explain.md](overall-explain.md) as your reference when you get stuck.
3. Practice one real case per day from [src](src).
4. For each case, write three things:
What failed, why it failed, and how you fixed it.
5. End each day by answering 3 interview-style questions out loud.

## Alternative path (deeper study)

1. Start with [overall-explain.md](overall-explain.md).
2. Choose one category (autograd, performance, distributed, etc.).
3. Pair the category with one notebook in [src](src).
4. Implement one variation or failure case yourself.

## Suggested practice rhythm

- Day 1 to 2: tensor semantics, autograd, numerical stability.
- Day 3 to 4: model design and debugging workflows.
- Day 5 to 6: performance profiling and memory tradeoffs.
- Day 7: mock interview recap using your own notes.

## Project Goal

Build interview-ready ML engineering judgment, not just coding speed:
- explain why a training run fails,
- reason about gradients and numerics,
- choose practical PyTorch fixes under constraints,
- communicate tradeoffs clearly in interview settings.

## Notes

- This repo is practice-focused and notebook-heavy.
- Some datasets are kept locally in [src/cnn/data/MNIST/raw](src/cnn/data/MNIST/raw) for convenience.
- If a notebook is long, run it section by section and record your findings in your own prep notes.
