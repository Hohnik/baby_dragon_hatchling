# TODO — BDH Improvement Tracker

## Active

### 1. Recurrent State + TBPTT (Continuous Learning) ← FUNDAMENTAL
The paper's core claim: BDH is a state-space model with Hebbian synaptic state ρ that
accumulates across time. Our implementation resets attention each forward pass, losing
the continuous learning property entirely.

**What to implement:**
- [ ] Recurrent `ρ` state in Attention that persists across chunks
- [ ] Truncated BPTT training: sequential chunks with state carryover + detach
- [ ] KV-cache for inference falls out naturally from this

**Paper reference:** Eq. (8): `ρ_{t,l} := ρ_{t-1,l} + LN(E·y) · x^T · U`
Appendix B.3: "Truncated Backpropagation Through time, carrying over the recurrent
state of attention and training on sequences of length 2048 characters at a time."

### 2. Sequence Length Curriculum
Start training at T=64, ramp to T=256. Attention is O(T²), so short sequences are
much cheaper and still provide useful gradient signal early on.
- [x] Schedule T as a function of training step
- [ ] Benchmark early training speed improvement

### 3. Muon Optimizer
Muon (MomentUm Orthogonalized by Newton-Schulz) converges 1.5-2x faster than AdamW
for transformer-like projection matrices. BDH's large encoder/decoder params are a
natural fit.
- [ ] Implement Muon for projection matrices (encoder, decoder, encoder_v)
- [ ] Keep AdamW for embeddings and LayerNorm (per modded-nanogpt recipe)
- [ ] Benchmark convergence speed vs AdamW

### 4. Data Pipeline
Async data loading to overlap CPU batch preparation with GPU compute.
- [ ] DataLoader with prefetching

## Done
- [x] Gradient checkpointing (OOM → trainable)
- [x] RoPE cache precomputation (-1.3ms/forward)
- [x] Attention score normalization 1/√N (enables fp16)
- [x] fp16 autocast (+14% speed)
- [x] Per-layer parameters (val 3.39 → 2.60)
- [x] 2-layer optimal config
- [x] Cosine LR schedule + gradient clipping
- [x] BDH-HRM hybrid architecture (preserved in bdh_hrm.py)

## Deferred
- MLX port: 2-5x speed but complete rewrite. Community port exists (severian42/BDH-MLX).
- FlashAttention tiling: needs Metal shader programming, high effort.
