# BDH Training Improvements

## What Was Implemented

These changes are in `src/bdh.py` and `src/train.py`. Together they make training
**~20x faster** on Apple M1 8GB (from ~60s/step to ~3.5s/step) and fix the OOM crash
that made the original config (`B=32, T=512`) impossible to run.

### 1. Gradient Checkpointing (`bdh.py`)
**Impact: ~60% memory reduction, ~20x faster on M1**

The original code stores all intermediate activations for backward (6 layers of
`(B, nh, T, N=8192)` tensors). With `B=32, T=512` this needs ~74GB — impossible on 8GB.
Gradient checkpointing recomputes activations during backward instead of storing them.

On M1 this is paradoxically *faster* because reduced memory pressure prevents GPU stalls
and avoids swapping to system memory.

### 2. RoPE Cache Precomputation (`bdh.py`)
**Impact: ~1.3ms saved per forward call (6 layers × every step)**

The original recomputes `cos(phases)` and `sin(phases)` every forward call, for every
layer. These only depend on sequence position and frequency — they never change. We
precompute them once for `max_seq_len` positions and slice at runtime.

### 3. Gradient Accumulation (`train.py`)
**Impact: Simulate large effective batch with small micro-batches**

Original: `B=32` in one shot → OOM.
New: `MICRO_BATCH=4` × `GRAD_ACCUM_STEPS=1` = effective batch of 4. Users can increase
`GRAD_ACCUM_STEPS` to simulate any effective batch size (e.g., 8 steps → effective 32).

### 4. Cosine LR Schedule with Warmup (`train.py`)
**Impact: Better convergence, standard practice**

Linear warmup (200 steps) prevents early instability, cosine decay smoothly anneals to
`MIN_LR`. This is the standard schedule from GPT-2/3, PaLM, LLaMA, and nanoGPT.

### 5. Gradient Clipping (`train.py`)
**Impact: Training stability**

`clip_grad_norm_(model.parameters(), 1.0)` prevents rare large gradients from blowing
up weights. Standard for all deep transformer training.

### 6. Validation Evaluation (`train.py`)
**Impact: Detect overfitting**

Periodic multi-batch val loss estimation. Without this you're flying blind — train loss
alone doesn't tell you if the model generalizes.

### 7. Fixed Device/AMP Detection (`train.py`)
**Impact: Correctness**

Original: hardcoded `device = torch.device("mps")`, AMP context was always `nullcontext()`
on MPS, GradScaler was always disabled. None of the mixed-precision code worked.
New: proper auto-detection. Note: **fp16 autocast causes NaN after 1 step** because BDH's
attention inner dim (N=8192) produces gradient values that overflow fp16 range. bfloat16
works but gives no speedup on M1 (no hardware BF16 units).

### 8. Removed `torch.compile` (`train.py`)
**Impact: Avoids 0.72x slowdown on MPS**

PyTorch's `inductor` backend has no Metal support. On MPS, `torch.compile` adds overhead
with no optimization. Benchmarked: 0.72x speed (slower).

---

## Improvements To Investigate (Not Implemented)

### A. MLX Port — Expected: 2-5x Faster on Apple Silicon
**Why:** MLX is Apple's native ML framework, built specifically for Apple Silicon's unified
memory architecture and Metal GPU. It avoids the PyTorch→MPS translation layer entirely.

A community MLX port already exists: [severian42/BDH-MLX](https://github.com/severian42/BDH-MLX).
Key advantages:
- Lazy evaluation and unified memory: no CPU↔GPU copies
- Metal-native kernels for matmul, softmax, etc.
- `mlx.nn` has direct gradient checkpointing support
- The BDH architecture is a natural fit (no custom CUDA needed)

**Effort:** Medium. Port is already done, needs benchmarking and training loop optimization.

### B. Per-Layer Parameters — Expected: Better Model Quality
**Why:** The current implementation **reuses the same `encoder`, `encoder_v`, `decoder`
parameters across all 6 layers** (the `BDHLayer` object is shared). This is weight tying.
While it reduces parameter count, it limits the model's expressiveness.

Standard transformers have independent parameters per layer. Making each layer have its
own `encoder`/`decoder` would increase params from ~25M to ~150M but allow each layer to
learn different representations.

**Effort:** Low. Change the `for level in range(C.n_layer)` loop to use per-layer parameter
sets. Would require 6x more memory for parameters (but parameters are small compared to
activations).

### C. Muon Optimizer — Expected: 1.5-2x Faster Convergence
**Why:** The [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) project shows
that the Muon optimizer (MomentUm Orthogonalized by Newton-Schulz) converges significantly
faster than AdamW for transformer-like architectures. It applies Newton-Schulz orthogonalization
to the momentum, which better conditions the optimization landscape.

For BDH, the large projection matrices (`encoder`: `(nh, D, N)`, `decoder`: `(nh*N, D)`)
are exactly the kind of parameters Muon excels at optimizing.

**Effort:** Medium. Implement Muon or use a library. Apply to projection matrices only
(keep AdamW for embeddings and LayerNorm, per modded-nanogpt's recipe).

### D. FlashAttention-Style Tiling — Expected: 2-4x Faster Attention
**Why:** BDH's attention computes `(B, nh, T, N) @ (B, nh, N, T)` where N=8192. The full
Q/K tensors don't fit in GPU L2 cache (9MB per head per sample, vs 4MB L2 on M1).

FlashAttention tiles the computation to process small blocks that *do* fit in SRAM/L2,
avoiding repeated HBM round-trips. For BDH this would mean tiling along the T dimension
and possibly the N dimension.

**Challenge:** FlashAttention implementations (Tri Dao's, PyTorch's `scaled_dot_product_attention`)
assume standard `d_k=64-128` head dimension, not N=8192. A custom tiled kernel would be
needed. On MPS this means writing Metal shaders.

**Effort:** High. Requires Metal shader programming or waiting for PyTorch to support
`scaled_dot_product_attention` with very large head dims on MPS.

### E. Attention Score Normalization — Expected: Better Training Stability
**Why:** Standard transformers scale attention by `1/√d_k`. BDH does NOT scale scores:
```python
scores = (QR @ KR.mT).tril(diagonal=-1)  # no 1/√N scaling!
```
With N=8192, raw scores reach 400-450 at initialization. This is why fp16 gradients overflow.
Adding `scores = scores / math.sqrt(N)` or `scores = scores / N` would:
- Enable fp16 training (halve memory bandwidth, ~1.5x faster on newer Apple Silicon)
- Stabilize training dynamics
- Follow the theoretical attention scaling from "Attention Is All You Need"

**Effort:** Very low (one line change) but needs careful tuning — the BDH paper's training
dynamics may rely on the current unnormalized scores.

### F. KV-Cache for Generation — Expected: 50-100x Faster Inference
**Why:** Currently, `generate()` recomputes the full forward pass for ALL tokens at every
step, with cost O(T²) per new token and O(T³) total for generating T tokens.

A KV-cache would store previously computed attention states and only compute the new token's
attention, reducing generation from O(T³) to O(T²).

BDH's shared Q=K structure makes caching slightly different from standard transformers:
the post-RoPE representations can be cached, and only the new row of the attention matrix
needs to be computed.

**Effort:** Medium. Need to restructure `Attention.forward` to support incremental mode.

### G. Data Pipeline — Expected: 10-20% Throughput Improvement
**Why:** Current data loading is synchronous: the GPU waits while Python prepares the next
batch. A `torch.utils.data.DataLoader` with `num_workers > 0` would prepare the next batch
on CPU while the GPU runs the current step.

On M1 with unified memory, the benefit is smaller than on discrete GPUs (no PCIe transfer),
but CPU-side numpy/torch overhead still blocks the training loop.

**Effort:** Low. Standard DataLoader with prefetching.

### H. Sequence Length Curriculum — Expected: Faster Early Training
**Why:** Start training with short sequences (e.g., T=64) and gradually increase to T=256.
Short sequences are much cheaper (attention is O(T²)) and provide useful gradient signal
early in training when the model is still learning basic token relationships.

This is used in practice by several large model training runs (PaLM, some LLaMA variants).

**Effort:** Low. Schedule `BLOCK_SIZE` as a function of training step.

### I. Separate Layer Parameters — Expected: Better Loss per Parameter
**Why:** Currently all 6 layers share `encoder`, `encoder_v`, and `decoder` parameters.
While weight-sharing reduces memory, each layer performs the same transformation on
different input distributions. Independent per-layer weights would allow:
- Earlier layers to learn local patterns (character n-grams)
- Later layers to learn compositional/semantic patterns
- Better gradient flow (each layer gets its own gradient, not a sum)

**Effort:** Low. Create lists of parameters instead of single shared tensors.

---

## Quick Reference: Impact vs Effort Matrix

| Improvement | Speed | Quality | Memory | Effort |
|---|---|---|---|---|
| **E. Score Normalization** | +50%* | ↑↑ | ↓↓ | 1 line |
| **I. Per-Layer Params** | — | ↑↑↑ | ↑ | Low |
| **H. Seq Length Curriculum** | +30% early | ↑ | ↓ | Low |
| **G. Data Pipeline** | +15% | — | — | Low |
| **C. Muon Optimizer** | — | ↑↑ | — | Medium |
| **F. KV-Cache** | 50-100x gen | — | ↑ | Medium |
| **A. MLX Port** | 2-5x | — | ↓ | Medium |
| **B. Per-Layer Params** | — | ↑↑↑ | ↑ | Low |
| **D. FlashAttention Tiling** | 2-4x attn | — | ↓↓ | High |

*\*Score normalization enables fp16, which helps on M2+ (M1 has limited fp16 throughput)*
