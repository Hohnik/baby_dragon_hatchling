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
gain on M1 (18% slower due to NS compute). Better on larger GPUs. Available in
`src/muon.py` as drop-in replacement.

### 9. BDH-HRM Hybrid Architecture
Combined BDH sparse attention with HRM hierarchical reasoning. The no-grad trick
gives 2x speed. Best config: 1H×2L. Finding: the quality gain came from per-layer
params, not the hierarchy. Preserved in `src/bdh_hrm.py` for reasoning task
experiments where the hierarchical structure is expected to help.

### 10. Training Infrastructure
Cosine LR schedule with warmup, gradient clipping, validation evaluation,
proper device detection, gradient accumulation, sequential data streaming.

## Summary

| Improvement | Speed Impact | Quality Impact |
|---|---|---|
| Gradient checkpointing | OOM → 3.5s/step | Training possible |
| RoPE cache | -1.3ms/forward | — |
| Score normalization | Enables fp16 | Training stability |
| fp16 autocast | +14% | — |
| Per-layer parameters | — | **val 3.39 → 2.60** |
| Recurrent state (TBPTT) | — | Continuous learning |
| Seq length curriculum | **1.76x warmup** | — |
| Muon optimizer | -18% (M1) | +0.01 val loss |

**Total: OOM/~60s per step → 1.1s/step, val 3.39 → 2.54**

## Future Work

### A. MLX Port — Expected 2-5x Faster
Apple's native ML framework. Avoids PyTorch→MPS translation layer. Community port
exists: severian42/BDH-MLX. Would require full rewrite but gives native Metal kernels.

### B. FlashAttention Tiling — Expected 2-4x Faster Attention
BDH's N=8192 per-head dim causes cache thrashing (9MB working set vs 4MB M1 L2).
Tiling would process blocks that fit in SRAM. Requires Metal shader programming.

### C. Adaptive Computation Time (ACT)
From HRM: learn to halt early when the answer is "easy". Use Q-learning on top of
BDH-HRM's iteration count. Would save compute on predictable tokens while spending
more on complex reasoning.

### D. Larger-Scale Validation
Train on Europarl or similar multi-language corpus as in the paper. Test whether
monosemantic synapses emerge. Validate scaling laws against paper's Figure 5.
