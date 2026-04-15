# LoRA Fine-Tuning From Scratch and QLoRA Initialization

## Table of Contents

- [Short Answer](#short-answer)
- [Ordinary Fine-Tuning](#ordinary-fine-tuning)
- [The Core LoRA Idea](#the-core-lora-idea)
- [Why Low-Rank Helps](#why-low-rank-helps)
- [Forward-Pass View](#forward-pass-view)
- [Why LoRA Is Parameter-Efficient](#why-lora-is-parameter-efficient)
- [Where LoRA Is Usually Applied](#where-lora-is-usually-applied)
- [Scaling](#scaling)
- [Initialization](#initialization)
- [Why LoRA Works So Well](#why-lora-works-so-well)
- [LoRA vs Full Fine-Tuning](#lora-vs-full-fine-tuning)
- [QLoRA Connection](#qlora-connection)
- [Default QLoRA Initialization](#default-qlora-initialization)
- [Why A Is Random and B Is Zero](#why-a-is-random-and-b-is-zero)
- [Why Kaiming-Uniform Is Used](#why-kaiming-uniform-is-used)
- [PEFT Initialization Options](#peft-initialization-options)
- [Practical Recommendation](#practical-recommendation)
- [Notes From Online Sources](#notes-from-online-sources)

## Short answer

LoRA fine-tuning does not update the full pretrained weight matrix. Instead, it learns a small low-rank correction to that matrix.

That is why it is parameter-efficient: instead of training every entry in a large weight matrix, you train only two much smaller matrices whose product approximates the update you need.

In one sentence:

> LoRA is parameter-efficient because it replaces a full weight update with a learned low-rank correction, so you adapt a large pretrained model by training only a very small number of extra parameters.

---

## Ordinary Fine-Tuning

Suppose a pretrained linear layer has weight

$$
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
$$

and computes

$$
h = Wx
$$

In full fine-tuning, you update all entries of $W$.

That is expensive because modern models have billions of parameters, and full fine-tuning means:

- many trainable parameters,
- large optimizer state,
- large gradient storage,
- a separate full model checkpoint for each task.

---

## The Core LoRA Idea

LoRA assumes that for a downstream task, you often do not need a completely free update matrix $\Delta W$.

Instead of learning a full update, LoRA writes it as

$$
\Delta W = BA
$$

where

$$
A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}
$$

and $r$ is a small number called the rank, with

$$
r \ll \min(d_{\text{in}}, d_{\text{out}})
$$

So the adapted layer becomes

$$
h = (W + BA)x
$$

Usually the pretrained weight $W$ is frozen, and only $A$ and $B$ are trained.

This is the central LoRA idea: do not relearn the whole matrix, only learn a compact correction to it.

---

## Why Low-Rank Helps

A full update matrix has

$$
d_{\text{out}} d_{\text{in}}
$$

degrees of freedom.

A rank-$r$ update uses only about

$$
rd_{\text{in}} + d_{\text{out}}r = r(d_{\text{in}} + d_{\text{out}})
$$

parameters.

That can be dramatically smaller.

Example:

$$
d_{\text{in}} = d_{\text{out}} = 4096, \quad r = 8
$$

Then a full update has size

$$
4096 \times 4096 \approx 16.8 \text{M}
$$

while LoRA uses

$$
8 \times 4096 + 4096 \times 8 = 65536
$$

trainable parameters.

That is the main source of the savings.

Intuitively, LoRA bets that the task-specific weight movement lives in a much smaller effective subspace than the full matrix dimension.

---

## Forward-Pass View

Because

$$
(W + BA)x = Wx + B(Ax)
$$

LoRA can be implemented as a small residual branch on top of the frozen layer:

1. Compute the normal pretrained output $Wx$.
2. Compute the adapter path $Ax$.
3. Project back with $B(Ax)$.
4. Add the adapter output to the frozen output.

So operationally, LoRA looks like a compact down-projection followed by an up-projection added to the original layer output.

---

## Why LoRA Is Parameter-Efficient

LoRA is parameter-efficient for several separate reasons.

### Fewer trainable parameters

You train only $A$ and $B$, not all of $W$.

### Less optimizer memory

Optimizers such as Adam usually keep extra tensors per trainable parameter. If trainable parameters are much fewer, optimizer memory drops sharply.

### Less gradient storage

Gradients are needed only for the LoRA parameters, not for the frozen base weights.

### Easier storage and deployment

You can keep one frozen base model and save many small LoRA adapters for different tasks instead of saving a full fine-tuned copy of the entire model each time.

### Easier multi-task adaptation

Different tasks can share the same base model and swap in different adapters.

That is why LoRA is usually discussed as a parameter-efficient fine-tuning method rather than just a low-rank modeling trick.

---

## Where LoRA Is Usually Applied

In transformers, LoRA is often attached to selected linear projections rather than every parameter in the network.

Common choices are:

- query projection $W_q$
- value projection $W_v$

Sometimes people also adapt:

- key projection $W_k$
- output projection $W_o$
- MLP linear layers

The exact target modules depend on the model and training setup. Standard LoRA often focuses on a subset such as attention projections, while QLoRA commonly targets all linear layers.

---

## Scaling

LoRA usually introduces a scaling factor, so the effective layer is written as

$$
W + \frac{\alpha}{r} BA
$$

where $\alpha$ is a hyperparameter.

This controls how strong the adapter update is relative to the frozen base model.

In implementations, this scaling may be written in slightly different but equivalent forms, but the idea is the same: separate the rank choice from the update magnitude.

---

## Initialization

A common LoRA initialization is:

- one factor initialized randomly,
- the other initialized to zero.

So initially

$$
BA \approx 0
$$

and the model starts very close to the original pretrained model.

This is useful because the adapter begins as a no-op instead of immediately perturbing the pretrained network.

---

## Why LoRA Works So Well

LoRA looks restrictive at first, but it often performs surprisingly close to full fine-tuning.

The main reasons are:

- pretrained models already contain a large amount of useful structure,
- many downstream tasks require only modest movement in weight space,
- useful task-specific updates are often effectively low-dimensional.

So the low-rank bottleneck is often much less harmful than it first appears.

Another way to say it is:

> Instead of relearning the whole matrix, LoRA learns a compact directional correction that steers the pretrained model toward the new task.

---

## LoRA vs Full Fine-Tuning

### Full fine-tuning

- updates all weights
- offers the highest flexibility
- is expensive in memory and storage
- usually requires one full model copy per task

### LoRA

- freezes the base model
- trains only small adapters
- is much cheaper in memory and storage
- makes task-specific adaptation easy to swap

The trade-off is straightforward:

- full fine-tuning is more expressive,
- LoRA is usually much more practical.

---

## QLoRA Connection

QLoRA keeps the same adapter idea but changes the base model storage.

For LoRA, the effective weight is

$$
W_{\text{eff}} = W + sBA
$$

For QLoRA, it becomes

$$
W_{\text{eff}} = W_q + sBA
$$

where $W_q$ is the quantized frozen base weight.

So the key difference is:

- LoRA: full-precision frozen base plus trainable adapter
- QLoRA: quantized frozen base plus trainable adapter

The LoRA adapter itself is still small and trainable. Quantization is applied to the base model weights, not to the fact that the adapter is a low-rank update.

---

## Default QLoRA Initialization

For both standard LoRA and QLoRA, the usual default initialization is the same:

- LoRA A: random initialization, typically Kaiming-uniform
- LoRA B: zeros

This makes the initial low-rank update zero, so at step 0 the model behaves like the original pretrained model.

In Hugging Face PEFT, `init_lora_weights=True` is the standard default behavior.

Example:

```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    init_lora_weights=True,
)
```

This is the standard starting point for LoRA and also for most QLoRA training setups.

---

## Why A Is Random and B Is Zero

The asymmetric initialization matters.

If both $A$ and $B$ were initialized to zero, then:

- the adapter output would be zero,
- the gradient with respect to $A$ would be zero,
- the gradient with respect to $B$ would also be zero.

So training would stall immediately.

If $A$ is random and $B$ is zero, then the adapter still starts as a no-op because $BA = 0$, but gradients can immediately update $B$ on the first step. Once $B$ moves away from zero, gradients also flow into $A$.

In principle you could swap the roles, but the standard LoRA convention is:

- $A$ random
- $B$ zero

---

## Why Kaiming-Uniform Is Used

Kaiming initialization was originally designed to keep activation variance stable in deep networks with ReLU-like nonlinearities.

Its classic variance intuition is roughly:

$$
\mathrm{Var}(W) \approx \frac{2}{\text{fan\_in}}
$$

In the LoRA setting, the practical reason is simpler:

- it is a strong, well-tested default for linear weights,
- it keeps the random factor $A$ at a reasonable scale,
- it matches the reference implementation behavior used by PEFT.

So the takeaway is not that LoRA itself contains a ReLU. The takeaway is that Kaiming-uniform is a sensible variance-scaled initialization for the random factor $A$, while zero-initializing $B$ preserves the identity start.

---

## PEFT Initialization Options

PEFT exposes several initialization choices beyond the standard default.

### 1. Standard default: `init_lora_weights=True`

- A: Kaiming-uniform
- B: zeros
- starts as a no-op
- best default for most LoRA and QLoRA setups

### 2. Gaussian: `init_lora_weights="gaussian"`

- A: Gaussian
- B: zeros
- also starts as a no-op
- used in some diffusion-style LoRA setups

```python
config = LoraConfig(init_lora_weights="gaussian", ...)
```

### 3. Non-identity random init: `init_lora_weights=False`

- does not preserve the identity start
- mainly useful for debugging or testing
- usually not recommended for real training

```python
config = LoraConfig(init_lora_weights=False, ...)
```

### 4. PiSSA: `init_lora_weights="pissa"`

- uses singular vectors and singular values of pretrained weights
- often converges faster than plain LoRA
- can reduce quantization error relative to plain QLoRA

```python
config = LoraConfig(init_lora_weights="pissa", ...)
```

Fast-SVD variants are also available, for example:

```python
config = LoraConfig(init_lora_weights="pissa_niter_4", ...)
```

### 5. CorDA: `init_lora_weights="corda"`

- task-aware initialization based on covariance structure
- supports knowledge-preserved and instruction-previewed modes
- requires a preprocessing pass

### 6. OLoRA: `init_lora_weights="olora"`

- QR-based initialization
- mutates base weights before training
- intended to improve stability and convergence

```python
config = LoraConfig(init_lora_weights="olora", ...)
```

### 7. EVA: `init_lora_weights="eva"`

- data-driven initialization based on activation SVD
- can redistribute rank across layers
- works with quantized models as well

```python
from peft import EvaConfig, LoraConfig

config = LoraConfig(
    init_lora_weights="eva",
    eva_config=EvaConfig(rho=2.0),
    ...
)
```

### 8. LoftQ for quantized models

- specifically designed for quantized LoRA and QLoRA setups
- initializes LoRA weights to reduce quantization error
- especially relevant when the base model is 4-bit quantized

In PEFT documentation, LoftQ is treated as a quantization-aware initialization workflow rather than the normal default. A common PEFT entry point is:

```python
from peft import replace_lora_weights_loftq

replace_lora_weights_loftq(peft_model)
```

For LoftQ, PEFT recommends targeting as many linear layers as possible, often with `target_modules="all-linear"`, and using NF4 for 4-bit quantization.

---

## Practical Recommendation

For interviews, study notes, and most implementation discussions, the clean explanation is:

1. LoRA freezes the pretrained matrix and learns a low-rank update instead of updating the whole matrix.
2. It is parameter-efficient because the trainable parameter count drops from $d_{\text{out}}d_{\text{in}}$ to about $r(d_{\text{in}} + d_{\text{out}})$.
3. QLoRA keeps the same adapter idea but quantizes the frozen base model.
4. The default adapter initialization in both LoRA and QLoRA is still the standard LoRA initialization: A random, B zero.
5. LoftQ and related schemes are optional improvements, not the basic default.

If you want a very short answer:

> LoRA is parameter-efficient because it learns a tiny low-rank correction to a frozen pretrained weight matrix instead of relearning the whole matrix, and QLoRA keeps that same idea while quantizing the frozen base model.

---

## Notes From Online Sources

This summary is consistent with:

- Hugging Face PEFT LoRA documentation on `init_lora_weights`, including the default Kaiming-uniform plus zero initialization, Gaussian initialization, PiSSA, CorDA, OLoRA, EVA, and LoftQ.
- Hugging Face PEFT quantization documentation on LoftQ and QLoRA-style training.
- The QLoRA paper, which describes backpropagation through a frozen quantized base model into LoRA adapters.
- PyTorch `torch.nn.init.kaiming_uniform_` documentation describing Kaiming initialization as a variance-preserving initialization originally motivated by rectifier nonlinearities.
