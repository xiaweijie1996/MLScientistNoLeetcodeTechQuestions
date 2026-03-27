# One-Week Fast Plan (PyTorch Interview Prep)

## Goal
Build strong interview readiness in 7 days for AI Scientist and Research Engineer roles by focusing on the highest-frequency PyTorch topics:
- correctness (autograd, tensor semantics, numerics)
- debugging and reproducibility
- performance and compiler behavior
- distributed training fundamentals
- memory optimization and deployment basics

## How to use this plan
- Daily budget: 3 to 5 focused hours.
- Structure each day: 60 percent hands-on, 30 percent explanation practice, 10 percent review.
- Output every day: one short write-up and one runnable artifact (notebook or script).
- End of day test: answer 3 questions out loud in 5 to 8 minutes each.

## Day 1 - Tensor semantics and autograd foundations
### Focus
- tensor shape, strides, contiguous vs non-contiguous
- view vs reshape, transpose and memory layout
- requires_grad, grad_fn, leaf vs non-leaf tensors

### Practice
1. Build a notebook showing when view fails and why reshape can copy.
2. Trace a small computation graph and inspect grad_fn nodes.
3. Write a one-page note: "How backward reaches parameter .grad".

### Deliverable
- notebook with 5 minimal examples and expected outputs
- 10 flashcards of common pitfalls

## Day 2 - Numerical stability and debugging failures
### Focus
- NaN and Inf triage workflow
- deterministic settings and reproducibility limits
- in-place operation autograd errors

### Practice
1. Create a controlled NaN case (high LR or unstable normalization) and fix it.
2. Make two runs deterministic and document which settings were required.
3. Reproduce one in-place gradient error and repair it in two different ways.

### Deliverable
- debugging checklist you can reuse in interviews
- before and after training logs

## Day 3 - Profiling and torch.compile
### Focus
- profiler interpretation (CPU bottleneck vs GPU bottleneck)
- graph breaks in torch.compile
- basic benchmarking hygiene

### Practice
1. Profile one training step and identify top 3 hotspots.
2. Enable torch.compile, find at least one graph break, reduce it.
3. Benchmark eager vs compile with warmup and fair timing.

### Deliverable
- short performance report with one chart or table
- 3 optimization hypotheses and test outcomes

## Day 4 - Transformer block fluency
### Focus
- attention math, masking, pre-norm vs post-norm
- regularization choices and gradient clipping
- practical training stability checks

### Practice
1. Explain scaled dot-product attention from memory.
2. Implement causal mask variants and validate outputs.
3. Compare one stable and one unstable configuration and explain why.

### Deliverable
- concise architecture note with equations and failure modes
- one clean reference transformer notebook

## Day 5 - Distributed training core
### Focus
- DDP mechanics, gradient buckets, synchronization timing
- DDP vs FSDP tradeoffs
- gradient accumulation in multi-GPU settings

### Practice
1. Draw the DDP backward plus allreduce timeline by hand.
2. Explain what changes with gradient accumulation and no_sync usage.
3. Write a decision table: when to choose DDP vs FSDP.

### Deliverable
- one-page distributed system cheatsheet
- 6 high-value distributed interview answers

## Day 6 - Memory optimization and custom/autograd internals
### Focus
- activation checkpointing tradeoffs
- AMP precision boundaries and GradScaler reasoning
- saved tensors, hooks, gradcheck, custom Function safety

### Practice
1. Add checkpointing to a model and measure memory and step time.
2. Run AMP, log stability issues, and document safe FP32 regions.
3. Create a tiny custom autograd function and validate with gradcheck.

### Deliverable
- memory optimization playbook (what to try first, second, third)
- custom op correctness checklist

## Day 7 - Deployment plus mock interview day
### Focus
- export paths: TorchScript, ONNX, torch.export
- parity testing after optimization or export
- synthesis and communication under time pressure

### Practice
1. Export one model path and run a parity check against eager.
2. Build a 90-minute mock interview set (theory, debugging, distributed, performance).
3. Record concise spoken answers for 10 must-know questions.

### Deliverable
- final summary: strengths, weak spots, next 2-week plan
- interview-ready answer bank (one paragraph each)

## Key topics to prioritize first
If your time is shorter than one week, prioritize in this order:
1. tensor semantics and autograd correctness
2. debugging and determinism
3. profiler plus torch.compile graph breaks
4. DDP and FSDP fundamentals
5. memory optimization (checkpointing and AMP)

## Daily review template (10 minutes)
- What did I learn today that prevents real training bugs?
- What can I explain clearly without notes?
- What failed, and what is my next hypothesis?
- Which two questions should I re-practice tomorrow?

## Interview answer quality bar
Your answer is strong when it includes:
- precise mechanism (what happens internally)
- practical tradeoff (speed, memory, stability, correctness)
- failure mode and mitigation
- one concrete example from your own practice
