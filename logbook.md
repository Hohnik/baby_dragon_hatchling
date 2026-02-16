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

**Cycle configuration sweep (30 steps each):**

| Config | Val Loss | ms/step | tok/s | Notes |
|---|---|---|---|---|
| BDH (6 shared layers) | 3.39 | 3126 | 328 | baseline |
| HRM 3H×2L | 3.09 | 1599 | 640 | default HRM config |
| HRM 2H×3L | 3.07 | 1478 | 693 | slightly better |
| HRM 6H×1L | 3.13 | 1976 | 518 | all abstract, slower |
| **HRM 1H×6L** | **2.74** | **1425** | **718** | **best by far!** |
| HRM 4H×3L (deep) | 3.14 | 2483 | 412 | more iterations but slower |

**Key insight: 1H×6L dominates.** Only 1 abstract H-cycle with 6 detailed L-cycles
gives both the best val loss AND the highest throughput. This aligns with the ARC Prize
team's finding: "HRM's gains came largely from its outer-loop refinement process" — in
our case, the L-level's iterative refinement of details is doing the heavy lifting, not
the H-level abstraction.

This makes biological sense: BDH's Hebbian gating (x_sparse * y_sparse) acts as a
plasticity mechanism that refines representations iteratively. More L-cycles = more
refinement passes through this gating mechanism.

---

## Entry 4 — Optimizing L-cycle Count (2026-02-16 23:30)

**Goal:** Now that 1H×?L dominates, find the optimal L-cycle count.

**L-cycle scaling sweep (30 steps each, B=4, T=256):**

| Config | Val Loss | ms/step | tok/s |
|---|---|---|---|
| BDH baseline | 3.39 | 3125 | 328 |
| **1H×1L (1 iter)** | 2.76 | **868** | **1180** |
| **1H×2L (2 iters)** | **2.68** | 920 | 1113 |
| 1H×3L (3 iters) | 2.70 | 1047 | 978 |
| 1H×4L (4 iters) | 2.71 | 1171 | 875 |
| 1H×6L (6 iters) | 2.74 | 1424 | 719 |

**Key insight: diminishing returns beyond 2 no-grad L iterations.** With the no-grad trick,
gradients only flow through the final L+H step. Additional no-grad iterations "warm up" the
hidden state but don't improve the gradient signal. For this character-level task, 1-2
warmup iterations are optimal.

**Best config: 1H×2L** — best val loss (2.68) at 3.4x faster than BDH baseline.

The 1H×1L config (no warmup) is also excellent: essentially just two separate BDH blocks
(H and L) with a single gradient pass. This is 3.6x faster than baseline and still far
better val loss (2.76 vs 3.39).

**What's really happening:** The gain isn't from more iterations. It's from having
**separate H and L parameters** instead of shared layers. The BDH baseline uses the same
encoder/decoder for all 6 layers. BDH-HRM gives each level its own parameters (50M vs 25M).
This supports improvement item I from improvements.md (per-layer parameters).

Created `src/train_hrm.py` with the optimal 1H×2L configuration.

---

## Entry 5 — Isolating the Source of Improvement (2026-02-17 00:00)

**Goal:** Is the gain from HRM hierarchy or from separate per-layer parameters?

**Controlled comparison (20 steps, B=4, T=256, all ~50M params):**

| Config | Val Loss | ms/step | tok/s |
|---|---|---|---|
| BDH 2-layer per-param | **2.75** | 863 | **1187** |
| BDH-HRM 1H×2L | 2.79 | 919 | 1115 |
| BDH-HRM 1H×1L | 2.91 | 864 | 1186 |

**Conclusion: The gain is from separate per-layer parameters, not from the HRM hierarchy.**

BDH 2-layer with per-layer params (val=2.75) slightly outperforms BDH-HRM 1H×2L (val=2.79)
at the same speed. The HRM cross-level injection (H+input→L, L→H) doesn't help for
character-level language modeling.

**However:** For complex reasoning tasks (ARC, Sudoku, mazes), the HRM hierarchy would
likely matter much more. The iterative refinement and multi-timescale processing are
designed for tasks that require planning and backtracking, not next-character prediction.

**What this means for the codebase:**
1. Adding per-layer params to the standard BDH is the simplest, highest-impact change
2. The BDH-HRM architecture is ready for future reasoning task experiments
3. The no-grad trick is valuable for memory efficiency regardless
