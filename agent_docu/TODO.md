# TODO — BDH Improvement Tracker

## Active — Paper-Informed Improvements

### 1. Forgetting Gate for State ρ ← HIGHEST IMPACT
BDH's state accumulates monotonically (ρ += QR^T @ V) with no ability to forget.
Three papers converge on the same solution: add a learned forgetting/retention gate.

**Sources:**
- **Titans** (2501.00663): surprise-gated forgetting in neural long-term memory
- **Palimpsa** (2602.09075): Bayesian metaplasticity with per-entry importance
- **Miras** (2504.13173): menu of retention gates beyond simple exponential decay
- **Forgetting Transformer** (2504.06949): per-head forget gates, 70% FLOP reduction

**Implementation:** `ρ_new = forget_gate * ρ_old + input_gate * QR^T @ V`
where gates are learned, data-dependent, per-head. Small code change in bdh.py.
- [ ] Implement data-dependent forget gate on state update
- [ ] Try sigmoid gate (GLA-style) vs exponential decay (Mamba-style)
- [ ] Benchmark long-sequence quality: does forgetting help on Shakespeare?
- [ ] Test per-head gate values: do some heads forget fast (pattern) vs slow (retrieval)?

### 2. Per-Head Specialization / Attention Pruning
Some heads may not need full attention if the state ρ already captures their patterns.

**Sources:**
- **Gu et al.** (2602.11374): only 2% of heads are retrieval-critical
- **Forgetting Transformer** (2504.06949): fast-forgetting heads → prune long-range attention
- **Pavlovian Conditioning** (2508.08289): capacity O(√N) ≈ 90 associations per head

- [ ] Analyze per-head attention patterns in trained model
- [ ] Test: skip QK^T attention for heads with fast forget gates (use only state ρ)
- [ ] Potential: 2x attention speedup at minimal quality cost

### 3. Titans-Style Memory Integration
Titans (2501.00663) is the closest existing work to BDH's architecture.
Compare their memory mechanism to improve BDH's.

- [ ] Read Titans carefully: compare MAC (Memory as Context), MAG (Memory as Gate),
      MAL (Memory as Layer) to BDH's approach
- [ ] Titans uses gradient-based memory update (not Hebbian). Test if this helps BDH.
- [ ] Titans' "surprise" metric for gating — can we measure surprise in BDH?

### 4. Larger-Scale Training Run
Run the full 3000-iteration training with all improvements to validate end-to-end
quality. Should take ~56 minutes on M1 with curriculum + TBPTT.
- [ ] Full training run with checkpoint saving
- [ ] Compare generated text quality vs original BDH

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
- **MLX port**: 2-5x speed, complete rewrite. See vllm-mlx (2601.19139) for Apple
  Silicon native inference patterns.
- **FlashLinearAttention** (from GLA, 2312.06635): hardware-efficient algorithm for
  BDH's linear state updates. Faster than FlashAttention-2.
- **TrasMuon** (2602.13498): trust-region + RMS calibration upgrade for Muon.
- **Adaptive halting (ACT)**: Q-learned iteration count for BDH-HRM.
- **RAM-Net addressing** (2602.11958): principled sparse vector construction.
- **CRAM consolidation** (2602.12204): route around attention when state suffices.
- **MiTA compression** (2602.01219): landmark-based compression of N=8192 dim.
- **Blending strategies** (2506.00744): compare sequential vs parallel memory
  integration in BDH (currently sequential).

## Paper References
See `papers/README.md` for full index with relevance analysis.
Key papers by priority:
1. Titans (2501.00663) — same concept as BDH, mature memory mechanism
2. Miras (2504.13173) — retention gate design space
3. Palimpsa (2602.09075) — principled forgetting for monotonic state
4. Forgetting Transformer (2504.06949) — per-head forget gates + pruning
5. FWP Primer (2508.08435) — theoretical grounding for BDH
