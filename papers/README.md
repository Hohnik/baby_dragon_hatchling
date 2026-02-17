# Papers — Relevance to BDH

Downloaded 2026-02-17. Searched arXiv for recent papers (2025–2026) on state-space
models, sparse attention, linear attention, Hebbian learning, fast weights, memory
consolidation, and efficient optimizers.

---

## Tier 1 — Directly applicable to BDH

### Titans: Learning to Memorize at Test Time
- **File:** `titans_learning_to_memorize_at_test_time.pdf`
- **arXiv:** 2501.00663 (Dec 2024)
- **Core idea:** Neural long-term memory module that learns to memorize historical
  context. Attention = short-term memory (accurate but limited context), neural
  memory = long-term persistent memory. Three architecture variants for combining them.
  Scales to 2M+ context with fast parallelizable training.
- **BDH connection:** This is conceptually identical to BDH's design — BDH's synaptic
  state ρ IS the long-term memory, and BDH's within-chunk attention IS the short-term
  memory. Titans provides a more mature framework with multiple integration strategies.
  The "Memory as Gate" (MaG) variant is closest to BDH's approach.
- **Actionable:** Compare BDH's state update rule to Titans' memory update. Titans uses
  gradient-based memory updates with surprise-gated forgetting — more principled than
  BDH's raw Hebbian accumulation. Could directly improve BDH's state quality.
- **Priority: HIGH** — This is the most important paper for understanding BDH's place
  in the landscape and improving its memory mechanism.

### Palimpsa: Learning to Remember, Learn, and Forget
- **File:** `palimpsa_metaplasticity_continual_learning.pdf`
- **arXiv:** 2602.09075v2 (Feb 2026)
- **Core idea:** Bayesian metaplasticity for attention state. Each state entry has
  learned importance controlling plasticity. Shows Mamba2 = special case (forgetting
  dominates). Transforms any non-metaplastic model into metaplastic one.
- **BDH connection:** BDH's state ρ accumulates monotonically with no forgetting.
  Palimpsa adds per-entry importance that controls write/forget balance. This is the
  missing piece for BDH's long-sequence quality.
- **Actionable:** Add learned forgetting gate: `ρ_new = gate * ρ + update`.
  Palimpsa provides the principled framework for choosing gate initialization.
- **Priority: HIGH** — Small code change, large potential impact.

### RAM-Net: Expressive Linear Attention with Selectively Addressable Memory
- **File:** `ram_net_sparse_memory_attention.pdf`
- **arXiv:** 2602.11958 (Feb 2026)
- **Core idea:** Maps inputs to high-dimensional sparse vectors as explicit memory
  addresses. Exponential state size with O(sparsity) compute per update.
- **BDH connection:** BDH's sparse encoder does the same thing — maps tokens to
  N=8192 sparse vectors used as memory addresses into state ρ. RAM-Net provides
  theoretical justification: sparsity mitigates signal interference.
- **Actionable:** Compare addressing schemes. RAM-Net may suggest better sparse
  vector construction or show optimal sparsity levels.
- **Priority: MEDIUM**

### MiTA: Efficient Fast-Weight Scaling via Mixture of Top-k Activations
- **File:** `mita_fast_weight_top_k_attention.pdf`
- **arXiv:** 2602.01219v2 (Feb 2026)
- **Core idea:** Attention = 2-layer fast-weight MLP. Compress with landmark queries,
  then top-k routing for deformable experts.
- **BDH connection:** BDH IS a fast-weight architecture. MiTA's framework explains
  why it works. Landmark compression could reduce N=8192 while preserving capacity.
- **Actionable:** Evaluate compress-then-route as alternative to BDH's direct top-k.
- **Priority: MEDIUM**

### Miras: It's All Connected
- **File:** `miras_its_all_connected_retention_gates.pdf`
- **arXiv:** 2504.13173 (Apr 2025)
- **Core idea:** Unifying framework: all sequence models = associative memory with
  (i) architecture, (ii) attentional bias, (iii) retention gate, (iv) learning
  algorithm. Shows novel retention gates beyond simple exponential decay.
- **BDH connection:** BDH uses dot-product similarity (attentional bias) with no
  retention gate (infinite memory). Miras provides a menu of retention gates that
  could be plugged into BDH's state update. The Moneta/Yaad/Memora variants show
  what's possible beyond standard forgetting.
- **Actionable:** Try Miras' retention gate variants on BDH's state update.
- **Priority: HIGH** — Direct recipe for the forgetting mechanism BDH lacks.

---

## Tier 2 — Strong indirect relevance

### Forgetting Transformer + Adaptive Computation Pruning
- **File:** `forgetting_transformer_adaptive_pruning.pdf`
- **arXiv:** 2504.06949v2 (Apr 2025)
- **Core idea:** Add forget gate to softmax attention. Many heads forget quickly →
  only need local context. Dynamically prune long-range attention that's already
  decayed. 70% FLOP reduction in attention, 50-70% attention speedup, no quality loss.
- **BDH connection:** BDH could add forget gates per attention head. Heads that
  forget quickly = rely on state ρ instead. This naturally separates retrieval heads
  (slow forget) from pattern heads (fast forget = state-based).
- **Actionable:** Add per-head forget gate to BDH attention. Prune attention for
  fast-forgetting heads (use only state ρ). Could halve attention cost.
- **Priority: MEDIUM-HIGH**

### Fast Weight Programming Primer
- **File:** `fast_weight_programming_primer.pdf`
- **arXiv:** 2508.08435v4 (Aug 2025, updated Jan 2026)
- **Core idea:** Comprehensive survey of Fast Weight Programmers (FWPs), connecting
  them to transformers, SSMs, and neurobiology. Reviews computational characteristics
  and brain connections.
- **BDH connection:** BDH is explicitly a FWP — the paper's theoretical foundations
  apply directly. Good reference for understanding where BDH sits in the landscape
  and what's known about FWP training dynamics.
- **Priority: REFERENCE** — Read for theoretical grounding, not direct code changes.

### Test-Time Regression: A Unifying Framework
- **File:** `test_time_regression_unifying_framework.pdf`
- **arXiv:** 2501.12352v3 (Jan 2025)
- **Core idea:** All sequence models = memorization (regression) + retrieval.
  Softmax attention, linear attention, SSMs, FWPs all arise from 3 design choices.
  Derives novel higher-order generalizations.
- **BDH connection:** Places BDH in a principled design space. BDH's sparse Hebbian
  update = a specific regression weight + function class + optimization algorithm.
  The framework shows what other choices exist in the same space.
- **Priority: REFERENCE** — Theoretical understanding, design space exploration.

### Pavlovian Conditioning View of Transformers
- **File:** `pavlovian_conditioning_transformers_hebbian.pdf`
- **arXiv:** 2508.08289 (Aug 2025)
- **Core idea:** Attention = Pavlovian conditioning. Q/K/V map to test/conditional/
  unconditional stimuli. Each attention op constructs transient associative memory
  via Hebbian rule. Capacity theorem: O(√d_k) associations per head.
- **BDH connection:** BDH's Hebbian gating IS this — the paper provides mathematical
  capacity bounds. With N=8192, BDH should store O(√8192) ≈ 90 associations per head.
  This predicts when BDH's memory saturates and needs forgetting.
- **Actionable:** Use capacity theorem to size BDH's N parameter optimally.
- **Priority: MEDIUM** — Principled N sizing instead of arbitrary 8192.

### Hebbian + Gradient Plasticity in Transformers
- **File:** `hebbian_gradient_plasticity_transformers.pdf`
- **arXiv:** 2510.21908v2 (Oct 2025)
- **Core idea:** Augment transformers with fast-weight modules using Hebbian or
  gradient-based plasticity. Hebbian = sharply gated around salient events,
  gradient-based = persistent updates. Hebbian better for short associations,
  gradient for long-horizon credit assignment.
- **BDH connection:** Validates BDH's Hebbian approach for character-level modeling
  (short associations). For longer tasks, gradient-based updates might complement
  the Hebbian state. Could add both plasticity types to different BDH layers.
- **Priority: MEDIUM**

### Gated Linear Attention (GLA)
- **File:** `gated_linear_attention_hardware_efficient.pdf`
- **arXiv:** 2312.06635v6 (Dec 2023, updated Aug 2024)
- **Core idea:** Linear attention with data-dependent gates. Hardware-efficient
  algorithm (FlashLinearAttention) faster than FlashAttention-2. Excellent length
  generalization (2K→20K).
- **BDH connection:** GLA's gating mechanism is closely related to BDH's Hebbian
  gating. GLA's hardware-efficient training algorithm could accelerate BDH's
  state update computation. The length generalization result is encouraging for
  BDH's TBPTT approach.
- **Priority: MEDIUM** — Algorithm could speed up BDH's state computation.

### Blending Complementary Memory Systems
- **File:** `blending_complementary_memory_systems.pdf`
- **arXiv:** 2506.00744v2 (Jun 2025)
- **Core idea:** Hybrid of KV-memory (softmax attention) + FW-memory (fast weights).
  Three blending methods: sequential, parallel, and interleaved. 340M and 1.3B
  models trained from scratch.
- **BDH connection:** BDH already blends within-chunk attention (KV-memory) with
  cross-chunk state (FW-memory). This paper systematically compares blending
  strategies — could optimize how BDH combines its two memory systems.
- **Priority: MEDIUM** — Optimization of existing architecture.

### TrasMuon: Trust Region Adaptive Scaling for Muon
- **File:** `trasmuon_trust_region_muon.pdf`
- **arXiv:** 2602.13498 (Feb 2026)
- **Core idea:** Muon discards magnitude info after Newton-Schulz. TrasMuon adds
  global RMS calibration + energy-based trust-region clipping. Converges faster,
  no warmup needed.
- **BDH connection:** Direct upgrade to our Muon optimizer in `src/muon.py`. Could
  fix the instability we observed and improve convergence without warmup.
- **Actionable:** Add RMS calibration and trust-region clipping to Muon.step().
- **Priority: LOW** (Muon is optional for us, AdamW is default)

### CRAM: Memory Consolidation for Adaptive Compute Reduction
- **File:** `cram_memory_consolidation_attention.pdf`
- **arXiv:** 2602.12204 (Feb 2026)
- **Core idea:** 88% of attention retrieves info already in hidden state. Learns to
  route around attention when redundant. 37.8x reduction, phase transition at 3K steps.
- **BDH connection:** BDH's state ρ should absorb recurring patterns over training.
  CRAM's routing could bypass attention when state already has the answer.
- **Priority: LOW-MEDIUM** — Interesting optimization, complex implementation.

### Retrieval-Aware Distillation for SSM Hybrids (Albert Gu)
- **File:** `retrieval_aware_distillation_ssm_hybrid.pdf`
- **arXiv:** 2602.11374 (Feb 2026)
- **Core idea:** Only 2% of attention heads are retrieval-critical. Preserve those,
  distill rest into SSM. 8x state reduction, 5-6x memory efficiency.
- **BDH connection:** If 1-2 of BDH's 4 heads handle retrieval, rest could skip
  attention and use only state ρ. Potential 2x attention speedup.
- **Priority: MEDIUM** — Requires head analysis on trained model.

---

## Summary: Recommended reading order

1. **Titans** — Most important. Same concept as BDH, more mature memory mechanism.
2. **Miras** — Menu of retention gates to plug into BDH's state update.
3. **Palimpsa** — Principled forgetting for BDH's monotonic state accumulation.
4. **Forgetting Transformer** — Per-head forget gates with dynamic attention pruning.
5. **FWP Primer** — Theoretical grounding for BDH as a fast weight programmer.
6. **Test-time regression** — Design space for BDH's position among sequence models.
7. **Pavlovian Conditioning** — Capacity bounds for BDH's Hebbian memory.
8. **RAM-Net** — Sparse addressing theory.
9. **GLA** — Hardware-efficient linear attention training.
10. **Blending Memory** — How to combine BDH's two memory systems optimally.

The single highest-impact change would be adding a **forgetting gate** to BDH's state
update, informed by Titans' surprise-gated mechanism or Miras' retention gates.
This addresses BDH's fundamental limitation: monotonic state accumulation with no
ability to forget outdated information.
