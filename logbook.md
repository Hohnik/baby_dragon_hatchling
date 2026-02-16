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

**Results:**
- 1/√N scaling brings init scores from ~87-160 down to ~0.96 (unit variance)
- **fp16 now works** — 10 training steps with zero NaN
- 14% speedup from fp16: 3515ms → 3095ms per step
- fp16 also doubles max batch: B=8 → B=20 fits in 8GB M1
- Loss curves match f32 closely (5.59→4.90 vs 5.59→4.91 in 10 steps)

---

## Entry 3 — BDH-HRM Hybrid Architecture (2026-02-16 23:00)

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
- H-level: BDH block (separate encoder/decoder params)
- L-level: BDH block (separate encoder/decoder params)
- Shared attention (RoPE is position-based)
- L receives (z_H + input_embeddings) as input injection
- H receives z_L as input injection
- h_cycles=3, l_cycles=2 (total 6 iterations = same as original 6 layers)
- No-grad trick: only final L+H iteration has gradients

**Results (50 training steps, B=4, T=256):**

| Model | Params | Val Loss | ms/step | tok/s | Total Time |
|---|---|---|---|---|---|
| BDH (6 shared layers) | 25.3M | **3.77** | 3126 | 328 | 156s |
| BDH-HRM (3H×2L) | 50.5M | 3.89 | **1600** | **640** | **80s** |

**Key findings:**
1. **BDH-HRM is 2x faster per step** thanks to the no-grad trick (only 1 iteration has
   gradients instead of 6). This is the biggest win.
2. **BDH-HRM trains loss faster**: at step 25, BDH-HRM=3.12 vs BDH=3.34 (train loss)
3. **Val loss slightly worse** after 50 steps (3.89 vs 3.77) — likely needs more steps for
   the hierarchical structure to learn H vs L roles, and has 2x more params to fit
4. **2x more parameters** (50M vs 25M) because H and L have separate encoder/decoder
5. In **wall-clock time**, BDH-HRM trains 50 steps in 80s vs BDH needs 156s — clearly
   more efficient
