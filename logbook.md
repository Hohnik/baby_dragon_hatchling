# Research & Development Logbook

## Entry 1 — Baseline Optimization (2026-02-16 20:45)

**Goal:** Make BDH trainable on M1 8GB Mac.

**Findings:**
- Original config (B=32, T=512) needs ~74GB for activations → instant OOM on 8GB
- Gradient checkpointing reduces peak memory ~60%
- RoPE cos/sin precomputation saves ~1.3ms/forward (recomputed 6x per step before)
- `torch.compile` is 0.72x slower on MPS (no Metal backend)
- fp16 autocast causes NaN after 1 training step (attention scores with N=8192 inner dim
  produce gradient values overflowing fp16 range max=65504). bfloat16 works but no speedup on M1.
- Net result: ~20x faster (60s→3.5s per step) with B=4 T=256

**Baseline performance:** B=4, T=256, ~3.5s/step, ~288 tok/s, loss 5.64→3.22 in 30 steps

---

## Entry 2 — Attention Score Normalization (2026-02-16 22:50)

**Goal:** Add 1/√N scaling to attention scores. This is the lowest-effort, highest-potential
improvement from improvements.md. Without scaling, raw scores reach 400-450 at init (N=8192).
Standard transformers scale by 1/√d_k (Vaswani et al. 2017).

**Hypothesis:** Normalization will:
1. Enable fp16 training (scores/grads won't overflow)
2. Improve training stability and convergence
3. Follow theoretical attention scaling

**Results:** [to be filled after implementation]

---

## Entry 3 — HRM + BDH Hybrid Architecture (2026-02-16 23:00)

**Goal:** Combine BDH's biologically-inspired sparse attention with HRM's hierarchical
multi-timescale reasoning.

**HRM Key Ideas (from sapientinc/HRM):**
- Two recurrent modules: H-level (slow, abstract planning) and L-level (fast, detailed)
- Iterative refinement: H_cycles × L_cycles forward passes
- "No-grad trick": all iterations except the last in torch.no_grad() — only final step
  has gradients. This is critical for memory efficiency with many iterations.
- Input injection: L receives (H + input_embeddings) at each step
- Optional adaptive halting via Q-learning (ACT)

**Why combine:**
- BDH's sparse ReLU activations give interpretable, biologically-grounded representations
- BDH's shared-weight layers are natural for iterative refinement (same computation repeated)
- HRM's hierarchical structure adds multi-timescale reasoning to BDH
- The "no-grad trick" enables many iterations without memory explosion
- BDH's Hebbian-style multiplicative gating (x_sparse * y_sparse) could act as a
  plasticity mechanism for the refinement loop

**Design:**
- H-level: BDH layer operating on slow/abstract state
- L-level: BDH layer operating on fast/detailed state
- L receives (z_H + input_embeddings) as input injection
- H receives z_L as input injection
- H_cycles=3, L_cycles=2 inner iterations (configurable)
- No-grad trick: only final iteration has gradients
- Ponder loss to regularize computation depth

**Results:** [to be filled after implementation]
