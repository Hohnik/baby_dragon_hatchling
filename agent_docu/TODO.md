# TODO — BDH Improvement Tracker

## Active

### 1. Larger-Scale Training Run
Run the full 3000-iteration training with all improvements to validate end-to-end
quality. Should take ~56 minutes on M1 with curriculum + TBPTT.
- [ ] Full training run with checkpoint saving
- [ ] Compare generated text quality vs original BDH

### 2. Metaplastic State Update (from Palimpsa, 2602.09075)
BDH's state ρ accumulates monotonically — no forgetting. Add per-entry importance
weights that control plasticity: `ρ_new = gate * ρ_old + (1-gate) * QR^T @ V`.
- [ ] Implement learned forgetting gate on state update
- [ ] Benchmark: does gated state improve long-sequence quality?
- [ ] Compare to Palimpsa's formulation (they show Mamba2 = forgetting-dominated case)

### 3. Retrieval-Critical Head Analysis (from Gu et al., 2602.11374)
Only 2% of attention heads are retrieval-critical. If 1-2 of BDH's 4 heads handle
retrieval, the rest could skip QK^T attention and use only the state ρ.
- [ ] Analyze per-head attention patterns in trained model
- [ ] Test: convert non-retrieval heads to pure state-based (no attention, just ρ)
- [ ] Potential: 2x attention speedup at minimal quality cost

## Done
- [x] **Recurrent state + TBPTT** — continuous learning per paper Eq. 8
- [x] **Sequence length curriculum** — 1.76x faster warmup (T=64→256)
- [x] **Muon optimizer** (optional, in src/muon.py) — +0.01 val, 18% slower on M1
- [x] **Per-layer parameters** — val 3.39 → 2.60 (biggest quality win)
- [x] **Score normalization 1/√N** — enables fp16, training stability
- [x] **fp16 autocast** — +14% speed
- [x] **Gradient checkpointing** — OOM → trainable
- [x] **RoPE cache** — -1.3ms/forward
- [x] **Cosine LR + gradient clipping** — standard best practices
- [x] **BDH-HRM hybrid** — preserved in bdh_hrm.py for reasoning tasks
- [x] **Data pipeline** — SequentialStreamer for TBPTT replaces random sampling

## Deferred
- **MLX port**: 2-5x speed, complete rewrite. Community port exists (severian42/BDH-MLX).
- **FlashAttention tiling**: needs Metal shaders. High effort.
- **Adaptive halting (ACT)**: Q-learned iteration count for BDH-HRM. Needs reasoning task.
- **RAM-Net addressing** (2602.11958): compare sparse vector construction to BDH's encoder.
- **CRAM consolidation** (2602.12204): route around attention when state is sufficient.
- **MiTA compression** (2602.01219): landmark-based compression of BDH's N=8192 dim.
