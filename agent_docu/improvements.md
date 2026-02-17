# BDH Improvements

## Implemented

### 1. Gradient Checkpointing — OOM → Trainable
Recompute layer activations during backward instead of storing 6 layers of
(B, nh, T, N=8192) tensors. Saves ~60% peak memory. Paradoxically faster on M1
because reduced memory pressure prevents GPU stalls and swapping.

### 2. RoPE Cache Precomputation — -1.3ms/forward
Precompute cos/sin tables for all positions up to max_seq_len once at init.
Original code recomputed trig functions every forward call, every layer.

### 3. Attention Score Normalization (1/√N) — Enables fp16
Standard Vaswani et al. scaling. Without it, raw scores reach 400+ with N=8192,
causing fp16 gradient overflow. With scaling, scores have unit variance at init.

### 4. fp16 Autocast — +14% Speed
Enabled by score normalization. Halves memory bandwidth for activations. Also
doubles max batch size (B=8 → B=20 fits in 8GB).

### 5. Per-Layer Parameters — Val 3.39 → 2.60 (Biggest Quality Win)
Each of the 2 layers gets its own encoder/decoder/encoder_v instead of sharing.
This lets early layers learn local patterns and later layers learn compositional
patterns. 2 layers optimal — more layers add params without improving loss.

### 6. Recurrent Synaptic State + TBPTT — Continuous Learning
The paper's core mechanism: synaptic state ρ accumulates across time via Hebbian
learning. Attention carries state between chunks. Training uses sequential text
with truncated BPTT. Also enables incremental generation (O(T) per token).
See paper Eq. 8 and Appendix B.3.

### 7. Sequence Length Curriculum — 1.76x Faster Warmup
Ramp T from 64→256 during warmup. Attention is O(T²) so short sequences are much
cheaper. T=64 is 2.65x faster than T=256 per step. Early training only needs to
learn basic character patterns, not long-range dependencies.

### 8. Muon Optimizer (Optional) — +0.01 Val Loss
Newton-Schulz orthogonalized momentum for projection matrices. Marginal quality
gain on M1 (18% slower due to NS compute). Better on larger GPUs. Removed from
codebase (not worth the complexity for M1 training).

### 9. BDH-HRM Hybrid Architecture
Combined BDH sparse attention with HRM hierarchical reasoning. The no-grad trick
gives 2x speed. Best config: 1H×2L. Finding: the quality gain came from per-layer
params, not the hierarchy. Removed from codebase (per-layer params adopted directly
in the main BDH model instead).

### 10. Training Infrastructure
Cosine LR schedule with warmup, gradient clipping, validation evaluation,
proper device detection, gradient accumulation, sequential data streaming.

### 11. Data-Dependent Forgetting Gate — Corrected Understanding
GLA-style input-dependent gate for the recurrent synaptic state ρ. Projects
the chunk's mean embedding through `Linear(D, n_head)` → sigmoid → per-head
forget scalar. **Correction (Entry 11):** Controlled ablation showed the gate's
improvement (val 2.64→2.24) came from neutralizing harmful state accumulation,
NOT from improving memory quality. Stateless training achieves equal quality
(val=2.24) without any gate. The gate is a safety net for when state is used,
not a quality enhancer. Gate remains useful for large-scale training where
cross-chunk state may provide genuine benefit. Configurable via
`BDHConfig.forget_mode`: `"none"` (default), `"scalar"`, `"data"`.

### 12. N=2048 Sparse Dimension — Val 2.24→2.20, 3.7x Faster
Reduced mlp_internal_dim_multiplier from 128 to 32, giving N=2048 per head
instead of N=8192. This is simultaneously better quality (fewer params =
more efficient training on small data) and 3.7x faster (all matmuls scale
linearly with N). Validated at 1000 steps: val=1.74 with readable text output.

### 13. Complex RoPE — Forward -41%
BDH's RoPE uses quantize(q=2) so dimension pairs share frequency — this IS
standard rotary position embedding. Rewritten using complex multiplication:
`view_as_complex(v) * freq_complex`. Single fused operation replaces the
stack+view+multiply+add pipeline. Microbench: 11.83ms → 3.7ms (3.2x faster).
Also eliminates fp32 upcast (cos/sin buffers were fp32, causing expensive
mixed-precision multiply). Max error ~2e-3, within training noise.

### 14. Stateless State Skip — 10% Forward Savings
When `state=None` (stateless training), the state update `QR.T @ V` was still
computed and returned. Now returns `None` directly. Generation explicitly
initializes zero state to preserve O(T) incremental inference.

### 15. Flat Encode + Einsum — +45% Throughput
Profiling showed broadcast matmul `(B,1,T,D) @ (nh,D,N)` had ~10% GPU
utilization due to batched kernel overhead. Restructured as single flat
matmul: `(B*T,D) @ (D,nh*N)` → 2x faster for encode. For encode_v, replaced
batched `@` with `einsum('bhtd,hdn->bhtn')` → 40% faster. Combined with
complex RoPE: forward 78ms→46ms, backward 141ms→110ms.

## Summary

| Improvement | Speed Impact | Quality Impact |
|---|---|---|
| Gradient checkpointing | OOM → 3.5s/step | Training possible |
| RoPE cache | -1.3ms/forward | — |
| Score normalization | Enables fp16 | Training stability |
| fp16 autocast | +14% | — |
| Per-layer parameters | — | **val 3.39 → 2.60** |
| Recurrent state (TBPTT) | — | Useful for large-scale only |
| Seq length curriculum | **1.76x warmup** | — |
| Muon optimizer | -18% (M1) | +0.01 val loss |
| Data-dependent forget gate | 10% overhead | Safety net (not quality win) |
| **N=2048 (reduced sparse dim)** | **3.7x faster** | **val 2.24→2.20** |
| **Complex RoPE** | **fwd -41%** | — |
| **Flat encode + einsum enc_v** | **+45% tok/s** | — |
| **Differential Attention** | **+4% overhead** | **val Δ=-0.019** |
| **Local Attention Window (w=64)** | **no overhead** | **val Δ=-0.032** |
| **Batch Size B=8** | **2x slower/step** | **val Δ=-0.284** |

**Total: OOM/~60s per step → ~370 ms/step, val ~1.68 at 500 steps (all improvements)**

## Future Work

### A. MLX Port — Expected 2-5x Faster
Apple's native ML framework. Avoids PyTorch→MPS translation layer. Community port
exists: severian42/BDH-MLX. Would require full rewrite but gives native Metal kernels.

### B. FlashAttention Tiling — Expected 2-4x Faster Attention
BDH's N=2048 per-head dim is more manageable now. Tiling for attention scores
(QR @ QR.T) could further improve cache utilization.

### C. Adaptive Computation Time (ACT)
From HRM: learn to halt early when the answer is "easy". Use Q-learning on top of
BDH-HRM's iteration count. Would save compute on predictable tokens while spending
more on complex reasoning.

### D. Larger-Scale Validation
Train on Europarl or similar multi-language corpus as in the paper. Test whether
monosemantic synapses emerge. Validate scaling laws against paper's Figure 5.
At larger scale, the stateful TBPTT + forget gate mechanism should become useful.

### 16. Differential Attention — Val Δ=-0.019 at 500 steps
Inspired by Microsoft's DIFF Transformer. Pairs adjacent attention heads and computes
the difference of their attention patterns: `output = (attn1 - λ*attn2) @ V`. This
cancels noise in attention scores and amplifies the true signal. λ is a learned scalar
per head pair, initialized at 0.5 via sigmoid. With 4 heads → 2 differential groups.
Only 2 extra parameters. Adds ~4% overhead from the double score computation, but
gives consistent quality improvement across all training durations tested (200, 500 steps).
Default: `diff_attn=True` in BDHConfig. See logbook Entry 15 for full sweep of 9
novel approaches and combination testing.

### Novel Approaches Tested but NOT Adopted (Entry 15)
- **SwiGLU encoding**: Δ=+0.194, 36% slower. Gated activation hurts BDH's sparse encoding.
- **Top-K sparsification**: Δ=+0.033, 60% slower. scatter/gather expensive on MPS.
- **per_head_v**: Δ=+0.027. Shared V across heads is a feature, not a bug.
- **RMSNorm**: Δ=-0.005 alone but hurts in combinations. LayerNorm is fine.
- **Gated residual**: Δ=-0.028 alone but hurts when combined with diff attention.
- **Learned temperature**: Δ=-0.009. Marginal and doesn't stack with diff attention.

### 17. Local Attention Window (w=64) — Val Δ=-0.032 at 500 steps
Restrict attention to the 64 most recent tokens instead of full causal attention.
Acts as regularization for character-level LM: prevents overfitting to spurious
long-range correlations in small datasets. Biologically plausible — cortical neurons
primarily interact locally. Zero extra parameters, no speed overhead.
Configurable via `BDHConfig.attn_window` (0 = full causal).

### Novel Approaches Tested but NOT Adopted (Entry 16)
- **Causal Conv1D preprocessing**: Δ=+0.112. Disrupts character embeddings.
- **Weight tying (embed ↔ lm_head)**: Δ=+0.002. Neutral.
- **Output MLP**: Δ=+0.149. Too many params in output, undertrained.
- **Embed scaling (√D)**: Δ=+0.026. Slight hurt.
- **3 layers (separate decoders)**: Δ=+0.005, 50% slower. Undertrained.
- **QK-Norm**: Δ=+0.438. Destroys ReLU sparsity patterns.
- **Wider N=4096**: Δ=+0.025. Undertrained at this data scale.
