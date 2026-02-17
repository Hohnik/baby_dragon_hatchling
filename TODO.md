# TODO — BDH Improvement Tracker

## Active

### 1. Larger-Scale Training Run
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
- **MLX port**: 2-5x speed, complete rewrite. Community port exists (severian42/BDH-MLX).
- **FlashAttention tiling**: needs Metal shaders. High effort.
- **Adaptive halting (ACT)**: Q-learned iteration count for BDH-HRM. Needs reasoning task.
