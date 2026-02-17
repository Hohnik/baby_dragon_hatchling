# TODO — BDH Implementation Plan

This file is the single source of truth for what to implement next.
Read this first in any new chat. Then read papers/README.md for paper context,
agent_docu/logbook.md for history, and src/bdh.py for current code.

---

## Current State

**Architecture:** BDH (Baby Dragon Hatchling) — a biologically-inspired language model
using sparse ReLU encoding into N=2048 dimensions, Hebbian gating, and RoPE.

**Code:** `src/bdh.py` (~340 lines), `src/train.py` (~350 lines), plus
`src/generate.py`, `src/benchmark.py`, `src/profile_ops.py`, `src/info.py`.
Run via `just <recipe>` (see `justfile`).

**What works:**
- 12.7M params, 2 layers with per-layer encoder/decoder/encoder_v
- N=2048 sparse dimension (reduced from 8192, better quality + 3.7x faster)
- Complex RoPE (3.2x faster than original stack+view)
- Flat encode matmul + einsum encode_v (+45% throughput)
- Stateless state skip (state=None → skip QR.T@V entirely)
- Stateless within-chunk attention (no cross-chunk state on small datasets)
- Sequence length curriculum (T=64→256 during warmup)
- fp16 autocast, gradient checkpointing, 1/√N scaling
- **122 ms/step, 8373 tok/s** on M1 8GB, val loss ~2.25 after 200 steps
- 1000-step run: val=1.74, 295ms/step (with checkpointing), readable text

---

## Task 1: Full Training Run with Optimized Config  ← IN PROGRESS

**Priority: HIGHEST.** Running 3000-step training with all improvements:
diff_attn=True, attn_window=64, B=8, N=2048, fp16.

Progress so far:
- [x] Step 500: val=1.7697 (vs old 1.9415, Δ=-0.172)
- [x] Step 1000: val=1.6213 (vs old 1.7827, Δ=-0.161)
- [x] Step 1500: val=1.5443 (vs old 1.6624, Δ=-0.118)
- [x] Step 2000: val=1.5381 (vs old 1.6027, Δ=-0.065)
- [ ] Step 2500
- [ ] Step 3000 — final eval + text samples

---

## Task 2: Longer Sequences (T=512)

**Priority: HIGH.** Currently limited to T=256. The paper uses T=2048.

### Why it should help
- More context = better predictions, especially for word-level patterns
- Attention is O(T²), so T=512 is 4x more compute per step but 2x more context
- Net effect: ~2x slower per step but higher quality
- Need to extend RoPE cache to max_seq_len=512+

### Implementation
- [ ] Change BDHConfig.max_seq_len = 512
- [ ] Update BLOCK_SIZE = 512 in train.py
- [ ] May need B=2 instead of B=4 for memory (T=512 doubles activation size)
- [ ] Benchmark: 200 steps at T=512 B=2 vs T=256 B=4

---

## Task 3: Batch Size Optimization

**Priority: HIGH.** Currently B=4 with fp16. Memory budget allows more.

### Why it should help
- Larger batch = more stable gradients, especially for character-level LM
- With stateless training (no state tensors), memory is mostly activations
- B=8 might fit with T=256 fp16 (or B=4 T=512)
- Could also try gradient accumulation: accum=2 with B=4 = effective B=8

### Implementation
- [ ] Test B=8 T=256 — does it fit in 8GB?
- [ ] Test B=4 T=256 accum=2 — same effective batch, lower memory
- [ ] Compare quality at same total tokens seen

---

## Task 4: Attention Pruning Analysis

**Priority: MEDIUM.** Depends on Task 1 (per-layer attention).

After Task 1, analyze whether different layers develop different attention patterns.
With per-layer attention, some heads might become redundant within certain layers.

The Forgetting Transformer (2504.06949) and Gu et al. (2602.11374) suggest only
2% of attention heads are retrieval-critical. With 4 heads × 2 layers = 8 heads
total, we could potentially skip 4-6 heads.

---

## Task 5: Titans-Style Memory (Large-Scale Only)

**Priority: LOW for current dataset.** Revisit if/when training on larger corpus.

The cross-chunk state mechanism needs billions of tokens to be useful.
Current tiny Shakespeare findings: state hurts (Entry 11). Keep the
stateful + forget gate code for future large-scale experiments.

---

## Done (Previous Sessions)
- [x] **Profiling-driven optimization** — 5.4x total speedup (Entry 14):
  - Complex RoPE: forward -41% (11.83→3.7ms per layer)
  - Flat encode matmul: 2x faster encode (4.38→2.15ms)
  - Einsum encode_v: 40% faster (4.02→2.41ms)
  - Stateless state skip: eliminate QR.T@V when unused
- [x] **N=2048 sparse dimension** — 3.7x faster + better quality (Entry 13)
- [x] **State mechanism correction** — stateless ≈ gated stateful on small data (Entry 11)
- [x] **Novel architecture sweep** — 9 approaches tested (Entry 15):
  - Differential Attention (DIFF Transformer): Δ=-0.019, now default
  - SwiGLU, Top-K, per_head_v, RMSNorm: tested and rejected
  - Gated residual, learned temp: marginal, don't stack
- [x] **Local attention window** — w=64, Δ=-0.032 (Entry 16)
- [x] **Batch size B=8** — fits in 8GB, Δ=-0.284 at 500 steps (Entry 17)
- [x] **Comprehensive novel approach testing** — 30+ experiments across 7 rounds:
  - Depth/width: 3-4 layers, N=4096, asymmetric, shared decoder
  - Structural: conv1d, weight tying, output MLP, embed scaling
  - Creative: layer recycling, token drop, Hebbian variants, multi-token pred
  - Regularization: label smoothing, dropout sweep, LR sweep
  - Normalization: pre-norm, pre+post, QK-norm, Xavier/scaled init
- [x] Data-dependent forgetting gate — safety net for stateful mode (Entry 10)
- [x] Scalar forgetting gate — limited, no head specialization (Entry 10)
- [x] Gradient check: forget_proj gets TBPTT gradients ✓
- [x] 500-step training run — val=1.90, checkpointing works (Entry 11)
- [x] Recurrent synaptic state + TBPTT (Entry 7)
- [x] Sequence length curriculum — 1.76x faster warmup (Entry 8)
- [x] Muon optimizer — optional, +0.01 val, 18% slower on M1 (Entry 9)
- [x] Per-layer parameters — val 3.39→2.60, biggest quality win (Entry 5)
- [x] Score normalization 1/√N — enables fp16 (Entry 2)
- [x] fp16 autocast — +14% speed (Entry 2)
- [x] Gradient checkpointing — OOM→trainable (Entry 1)
- [x] RoPE cache precomputation — -1.3ms/forward (Entry 1)
- [x] Cosine LR + gradient clipping (Entry 6)
- [x] BDH-HRM hybrid — preserved in bdh_hrm.py (Entry 3-5)
- [x] Data pipeline: SequentialStreamer for TBPTT (Entry 7)
- [x] Paper survey: 15 papers with relevance analysis (papers/README.md)

## Deferred
- MLX port: 2-5x speed, complete rewrite.
- FlashLinearAttention (GLA, 2312.06635): hardware-efficient state update.
- TrasMuon (2602.13498): trust-region Muon upgrade.
- RAM-Net addressing (2602.11958): principled sparse vector construction.
- CRAM consolidation (2602.12204): route around attention when state suffices.
- MiTA compression (2602.01219): landmark compression of N=8192 dim.
- Blending strategies (2506.00744): sequential vs parallel memory integration.
- Per-dimension forget gate (Step 3): likely overkill, Step 2 gave diminishing returns.

---

## File Map
```
src/bdh.py              — Main model (~320 lines). Attention + BDHLayer + BDH.
                          Config: BDHConfig.forget_mode = "none" | "scalar" | "data"
src/train.py            — Training with TBPTT + curriculum (~330 lines).
src/muon.py             — Optional Muon optimizer (168 lines).
src/bdh_hrm.py          — BDH-HRM hybrid for reasoning tasks (287 lines).
src/train_hrm.py        — HRM training script (200 lines).
src/benchmark.py        — Benchmark: BDH vs BDH-HRM (142 lines).
src/bench_forget_abc.py — A/B/C forget gate benchmark (all 3 modes).
src/bench_ab.py         — Legacy A/B forget gate benchmark.
src/bench_forget.py     — Quick 55-step forget gate test.
src/check_grad.py       — Verify gradients flow through forget gate in TBPTT.
src/checkpoints/        — Saved model checkpoints.
agent_docu/             — improvements.md, logbook.md, this TODO.md
papers/                 — 15 PDFs + README.md with relevance analysis
```
