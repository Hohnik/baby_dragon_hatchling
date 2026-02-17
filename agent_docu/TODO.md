# TODO — BDH Implementation Plan

This file is the single source of truth for what to implement next.
Read this first in any new chat. Then read papers/README.md for paper context,
agent_docu/logbook.md for history, and src/bdh.py for current code.

---

## Current State

**Architecture:** BDH (Baby Dragon Hatchling) — a biologically-inspired language model
using sparse ReLU encoding into N=8192 dimensions, Hebbian gating, and RoPE.

**Code:** `src/bdh.py` (312 lines), `src/train.py` (315 lines), plus optional
`src/muon.py`, `src/bdh_hrm.py`, `src/train_hrm.py`, `src/benchmark.py`.

**What works:**
- 50M params, 2 layers with per-layer encoder/decoder/encoder_v
- Recurrent synaptic state ρ carrying over between chunks (paper's Eq. 8)
- TBPTT training with SequentialStreamer
- Sequence length curriculum (T=64→256 during warmup, 1.76x faster)
- fp16 autocast, gradient checkpointing, RoPE cache, 1/√N scaling
- ~1160 ms/step, ~880 tok/s on M1 8GB, val loss ~2.54 after 50 steps

**Critical limitation:** State ρ accumulates monotonically (`ρ += QR^T @ V * scale`)
with NO forgetting mechanism. Over long sequences, state saturates and loses the
ability to distinguish recent from ancient information. This is the #1 thing to fix.

---

## Task 1: Forgetting Gate for State ρ  ← START HERE

**Priority: HIGHEST.** Four independent papers (2024–2026) converge on this exact fix.

### The Problem
In `src/bdh.py`, Attention.forward() line ~130:
```python
new_state = state + chunk_state  # monotonic accumulation, never forgets
```
This means ρ grows without bound. After many chunks, the accumulated state drowns
out the current chunk's contribution. The model can't distinguish "I saw this word
2 chunks ago" from "I saw this word 200 chunks ago."

### The Solution
Add a learned, data-dependent forgetting gate:
```python
# Instead of: new_state = state + chunk_state
# Do:         new_state = forget_gate * state + input_gate * chunk_state
```
where `forget_gate` and optionally `input_gate` are learned per-head scalars or
vectors, possibly input-dependent.

### Paper Guidance

**GLA (2312.06635)** — the foundational reference:
- State update: `S_t = G_t ⊙ S_{t-1} + v_t k_t^T`
- G_t is a data-dependent gate from a linear layer + sigmoid
- Has FlashLinearAttention algorithm for hardware-efficient training
- Key: the gate is per-head AND per-state-dimension

**Titans (2501.00663):**
- Uses "surprise" (loss on current input) to control forgetting
- High surprise → forget more (new info doesn't match memory)
- Memory update: `M_t = M_{t-1} - θ_t · ∇_M L_t` (gradient-based, not Hebbian)
- Forget: `M_t = (1 - α · σ(surprise)) · M_t + update`
- Three variants: MAC (memory as context), MAG (memory as gate), MAL (memory as layer)

**Miras (2504.13173):**
- Provides a menu of retention gates:
  - Exponential decay: `gate = exp(-α)` (constant, learned per head)
  - Sigmoid: `gate = σ(w · x + b)` (data-dependent)
  - Low-rank: `gate = σ(Wx)` where W is low-rank (cheaper than full)
- Novel models (Moneta, Yaad, Memora) combine different gate types

**Palimpsa (2602.09075):**
- Bayesian metaplasticity: each state entry has "importance" Ω
- High Ω → low plasticity (important memories are hard to overwrite)
- `Ω_new = Ω + |new_gradient|` — importance grows with use
- Shows Mamba2 is a special case where forgetting dominates

**Forgetting Transformer (2504.06949):**
- Forget gate in softmax attention: per-head, learned, data-dependent
- Key finding: many heads learn to forget quickly → only need local context
- Enables dynamic attention pruning (skip computation for decayed entries)
- 70% FLOP reduction, 50-70% speedup, zero quality loss

### Implementation Plan

**Step 1 — Simplest version: scalar learned gate per head**
Add a single learnable parameter per head that controls decay:
```python
# In Attention.__init__:
self.forget_bias = nn.Parameter(torch.zeros(1, n_head, 1, 1))  # init at 0 → gate=0.5

# In Attention.forward:
forget_gate = torch.sigmoid(self.forget_bias)  # (1, nh, 1, 1)
new_state = forget_gate * state + chunk_state
```
This is ~5 lines of code. Benchmark immediately.

**Step 2 — Data-dependent gate (GLA-style)**
Make the gate depend on the input:
```python
# In Attention.__init__:
self.forget_proj = nn.Linear(D, n_head)  # project input to per-head gate

# In Attention.forward:
# x_mean: mean of input embeddings over time → (B, D)
x_mean = V.squeeze(1).mean(dim=1)  # (B, D)
forget_gate = torch.sigmoid(self.forget_proj(x_mean))  # (B, nh)
forget_gate = forget_gate.unsqueeze(-1).unsqueeze(-1)  # (B, nh, 1, 1)
new_state = forget_gate * state + chunk_state
```

**Step 3 — Per-dimension gate (full GLA)**
Gate each dimension of the state independently:
```python
forget_gate = torch.sigmoid(self.forget_proj(x_mean))  # (B, nh*N)
forget_gate = forget_gate.view(B, nh, N, 1)  # (B, nh, N, 1)
new_state = forget_gate * state + chunk_state  # (B, nh, N, D)
```
This is the most expressive but adds nh*N parameters per layer.

### Benchmarking
For each variant, measure:
1. Val loss after 50 steps (quick) and 200 steps (convergence check)
2. ms/step overhead (should be negligible for scalar gate)
3. Inspect learned gate values: do they differ across heads?
4. Long-sequence test: generate 2000+ tokens, check coherence vs no-gate

### Success Criteria
- Val loss improvement (any amount — currently 2.54)
- Different heads learn different forget rates (specialization)
- Longer coherent generation than without gate

---

## Task 2: Per-Head Attention Pruning

**Priority: HIGH.** Depends on Task 1 (needs forget gates to identify pruneable heads).

### The Idea
After training with forget gates, some heads will learn to forget quickly (rely on
recent context / state only) while others forget slowly (true retrieval heads).
For fast-forgetting heads, the full QK^T attention computation is wasted — the
state ρ already captures everything they need.

**Sources:**
- Forgetting Transformer (2504.06949): 70% FLOP reduction from pruning decayed attention
- Gu et al. (2602.11374): only 2% of heads are retrieval-critical
- Pavlovian Conditioning (2508.08289): capacity O(√N) ≈ 90 associations per head

### Implementation
After Task 1 is trained, analyze forget gate values:
```python
for layer in model.layers:
    gate = torch.sigmoid(layer.attn.forget_bias)  # or forget_proj output
    print(f"Head forget gates: {gate.squeeze()}")
    # Heads with gate ≈ 0 (forget everything) → skip QK^T, use only state
    # Heads with gate ≈ 1 (remember everything) → need full attention
```

If heads specialize, add a threshold: heads with `forget_gate < threshold` skip
the expensive `QR @ QR.mT` computation and only use `QR @ state`.

### Expected Outcome
- 1.5-2x attention speedup if 2 of 4 heads are state-only
- Minimal quality loss (those heads weren't using attention anyway)

---

## Task 3: Titans-Style Memory Comparison

**Priority: MEDIUM.** Longer research task after Tasks 1-2.

Titans (2501.00663) is architecturally the closest existing work to BDH:
- Both: long-term memory (state/neural memory) + short-term (attention)
- BDH: Hebbian state update (`ρ += QR^T @ V`)
- Titans: gradient-based memory update (`M -= θ · ∇_M L_t`)

### Questions to answer:
1. Does gradient-based memory update outperform Hebbian for BDH's task?
2. Which Titans variant (MAC/MAG/MAL) maps best to BDH's architecture?
3. Can we use Titans' "surprise" metric to gate BDH's state updates?

### Implementation
- [ ] Read Titans paper thoroughly (especially Section 3: Memory Module)
- [ ] Implement surprise-gated version: measure prediction error, use it to
      modulate forget gate (high surprise → forget more, learn new pattern)
- [ ] Compare Hebbian vs gradient-based memory update on Shakespeare

---

## Task 4: Full Training Run

**Priority: MEDIUM.** Validate everything end-to-end after Tasks 1-2.

- [ ] Full 3000-iteration training with best configuration
- [ ] Checkpoint saving every 500 steps
- [ ] Compare generated text quality: with vs without forget gate
- [ ] Measure if TBPTT + forget gate outperforms stateless random sampling
      (TBPTT was slightly worse before — forget gate may fix this)

---

## Done (Previous Sessions)
- [x] Recurrent synaptic state + TBPTT (continuous learning per paper Eq. 8)
- [x] Sequence length curriculum (1.76x faster warmup, T=64→256)
- [x] Muon optimizer (optional, src/muon.py, +0.01 val, 18% slower on M1)
- [x] Per-layer parameters (val 3.39→2.60, biggest quality win)
- [x] Score normalization 1/√N (enables fp16, training stability)
- [x] fp16 autocast (+14% speed)
- [x] Gradient checkpointing (OOM → trainable)
- [x] RoPE cache precomputation (-1.3ms/forward)
- [x] Cosine LR + gradient clipping (standard best practices)
- [x] BDH-HRM hybrid (preserved in bdh_hrm.py for reasoning tasks)
- [x] Data pipeline (SequentialStreamer for TBPTT)
- [x] Paper survey: 15 papers downloaded with relevance analysis (papers/README.md)

## Deferred
- MLX port: 2-5x speed, complete rewrite. See vllm-mlx (2601.19139).
- FlashLinearAttention (GLA, 2312.06635): hardware-efficient state update algorithm.
- TrasMuon (2602.13498): trust-region upgrade for Muon optimizer.
- RAM-Net addressing (2602.11958): principled sparse vector construction.
- CRAM consolidation (2602.12204): route around attention when state suffices.
- MiTA compression (2602.01219): landmark compression of N=8192 dim.
- Blending strategies (2506.00744): sequential vs parallel memory integration.

---

## File Map
```
src/bdh.py          — Main model (312 lines). Attention + BDHLayer + BDH classes.
                      State update is in Attention.forward(), lines ~120-135.
src/train.py        — Training with TBPTT + curriculum (315 lines).
src/muon.py         — Optional Muon optimizer (168 lines).
src/bdh_hrm.py      — BDH-HRM hybrid for reasoning tasks (287 lines).
src/train_hrm.py    — HRM training script (200 lines).
src/benchmark.py    — Benchmark utilities (142 lines).
agent_docu/         — improvements.md, logbook.md, this TODO.md
papers/             — 15 PDFs + README.md with relevance analysis
papers/README.md    — Full paper index with priority ranking
```
