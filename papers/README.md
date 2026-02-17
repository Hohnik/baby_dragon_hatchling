# Papers — Relevance to BDH

Downloaded 2026-02-17. Searched arXiv for recent papers (Feb 2026) on state-space
models, sparse attention, linear attention, Hebbian learning, and fast weights.

---

## Tier 1 — Directly applicable

### RAM-Net: Expressive Linear Attention with Selectively Addressable Memory
- **File:** `ram_net_sparse_memory_attention.pdf`
- **arXiv:** 2602.11958 (Feb 12, 2026)
- **Core idea:** Maps inputs to high-dimensional *sparse vectors* as explicit memory
  addresses, enabling exponential state size with O(sparsity) compute per update.
- **BDH connection:** BDH does exactly this — sparse encoding maps tokens into N=8192
  dimensional vectors, which serve as addresses into the synaptic state ρ. RAM-Net
  provides a principled theoretical framework for why this works and how to optimize
  the sparsity/capacity tradeoff. Key insight: sparsity mitigates signal interference
  in the memory state, which is exactly what BDH's top-k activation achieves.
- **Actionable:** Compare RAM-Net's addressing scheme to BDH's encoder. RAM-Net may
  suggest better sparse vector construction or state update rules.

### Palimpsa: Learning to Remember, Learn, and Forget
- **File:** `palimpsa_metaplasticity_continual_learning.pdf`
- **arXiv:** 2602.09075v2 (Feb 11, 2026)
- **Core idea:** Adds Bayesian metaplasticity to attention state — each state entry
  has a learned "importance" that controls its plasticity. Shows Mamba2 is a special
  case (where forgetting dominates).
- **BDH connection:** BDH's synaptic state ρ accumulates monotonically (ρ += QR^T@V),
  with no forgetting mechanism. This causes state saturation over long sequences.
  Palimpsa's metaplasticity adds per-entry importance weights that would let BDH
  selectively forget outdated associations while preserving important ones.
- **Actionable:** Add a learned forgetting gate to BDH's state update:
  `ρ_new = gate * ρ_old + (1-gate) * QR^T @ V` where gate depends on importance.
  This is a small code change with potentially large impact on long-sequence quality.

### MiTA: Efficient Fast-Weight Scaling via Mixture of Top-k Activations
- **File:** `mita_fast_weight_top_k_attention.pdf`
- **arXiv:** 2602.01219v2 (Feb 3, 2026)
- **Core idea:** Interprets attention as a 2-layer fast-weight MLP. Proposes
  compress-and-route: compress with landmark queries, then top-k routing to
  construct deformable experts.
- **BDH connection:** BDH IS a fast-weight architecture — the sparse encoder is the
  first layer, the synaptic state is the fast weight, and the decoder reads it.
  MiTA's unifying framework explains why BDH works and suggests optimization:
  the top-k sparsity in BDH acts as the routing mechanism. MiTA's landmark-based
  compression could reduce BDH's N=8192 effective dimension while maintaining
  capacity.
- **Actionable:** Evaluate whether MiTA's compress-then-route is cheaper than BDH's
  direct top-k. Could reduce encoder size while preserving sparsity benefits.

---

## Tier 2 — Strong indirect relevance

### CRAM: Consolidation-based Routing for Adaptive Memory
- **File:** `cram_memory_consolidation_attention.pdf`
- **arXiv:** 2602.12204 (Feb 12, 2026)
- **Core idea:** 88% of attention retrieves info already predictable from hidden state.
  CRAM gradually distills episodic (attention) into semantic (parametric) memory,
  achieving 37.8x attention reduction via a phase transition at ~3K steps.
- **BDH connection:** BDH's synaptic state ρ is exactly the "semantic memory" that
  should absorb recurring attention patterns. CRAM's routing could skip attention
  entirely when ρ already contains the answer. The phase transition matches BDH's
  design intent: early training builds ρ, later training relies on it.
- **Actionable:** Add a learned routing gate that bypasses full attention when state
  retrieval is sufficient. Measure attention redundancy in trained BDH models.

### Retrieval-Aware Distillation for Transformer-SSM Hybrids (Albert Gu)
- **File:** `retrieval_aware_distillation_ssm_hybrid.pdf`
- **arXiv:** 2602.11374 (Feb 11, 2026)
- **Core idea:** Only 2% of attention heads (10 in a 1B model) are retrieval-critical.
  Preserve those, distill the rest into SSM. 8x state reduction possible once
  retrieval is handled by dedicated heads.
- **BDH connection:** BDH-HRM uses 4 attention heads. If only 1-2 are retrieval-
  critical, the rest could be simplified to pure recurrent (no attention computation).
  This could halve BDH's attention cost with minimal quality loss.
- **Actionable:** Analyze which BDH heads do retrieval vs pattern matching. Convert
  non-retrieval heads to pure state-based (skip the QK^T attention, use only ρ).

---

## Selection rationale

These 5 papers were selected from ~40 candidates. Key filters:
1. Must relate to BDH's specific mechanisms (sparse encoding, Hebbian state, fast weights)
2. Must be actionable (not just theoretical analysis)
3. Published Feb 2026 (within the last 2 weeks)

Rejected candidates (interesting but not directly actionable):
- Mamba training dynamics (2602.12499): theoretical, no new mechanism
- Hebbian with Global Direction (2601.21367): needs non-differentiable training
- Kalman Linear Attention (2602.10743): different architecture class
- WildCat (2602.10056): coreset sampling, doesn't fit BDH's sparse structure
- SLA2 (2602.12675): for diffusion models specifically
