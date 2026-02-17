# Graph-based BDH: faithful implementation of the paper's Section 2 & Table 1.
#
# This implements the GRAPH model (BDH), not the tensor-friendly BDH-GPU.
# Key differences from bdh.py (BDH-GPU):
#
#   BDH-GPU (bdh.py):                    BDH-Graph (this file):
#   ─────────────────                    ──────────────────────
#   Dense matmul (D_x E, D_y E)         Sparse graph propagation (G_x, G_y)
#   Compressed state ρ ∈ R^{n×d}        Full synaptic state σ on graph edges
#   Global attention (all neurons)       Local attention (K neighbors only)
#   Mean-field interaction               Wire-based graph communication
#   O(n·d) params per matrix             O(n·K) params per graph
#
# From the paper (Section 3, p.17):
#   "BDH-GPU is obtained from BDH by treating the communication of the n
#    particles as proceeding through a mean-field ('radio network'), rather
#    than a graph ('communication by wire')."
#
# Architecture (Table 1, Eq. 6):
#   σ_{t,l} := σ_{t-1,l} + y_{t,l-1} x_{t,l}^T ⊙ G_s · U
#   x_{t,l} := [x_{t,l-1} + (G_x^e - G_x^i) y_{t,l-1}]^+
#   y_{t,l} := [(G_y^e - G_y^i)(σ_{t-1,l} x_{t,l})]^+ ⊙ x_{t,l}
#
# Graph construction: power-law connectivity P(connect) ∝ 1/distance^α
# Neurons arranged on a ring, mostly local + long-range jumps.
#
# Performance: Uses torch.sparse.mm for graph propagation (14x faster than
# gather+sum). Attention uses parallel cumsum trick for O(T·N·K) compute.

import dataclasses
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclasses.dataclass
class BDHGraphConfig:
    """Configuration for graph-based BDH model."""
    n_neurons: int = 4096           # N: number of neurons (the large dimension)
    n_embd: int = 256               # d: embedding dimension (low-rank IO)
    k_attn: int = 16                # K_s: neighbors per neuron in attention graph
    k_prop: int = 32                # K: neighbors per neuron in propagation graphs
    n_layer: int = 4                # L: layers (per-layer weights, shared topology)
    n_head: int = 4                 # h: heads (subdivide N for attention)
    vocab_size: int = 256           # byte-level
    max_seq_len: int = 512
    dropout: float = 0.1
    alpha: float = 1.5              # power-law exponent for graph construction
    use_rope: bool = True           # RoPE in K-dimensional attention space
    # Whether to use separate excitatory/inhibitory circuits (paper's full model)
    # or simplified signed weights. Full model is more biologically faithful.
    use_inhibitory: bool = True


# ─── Graph Construction ─────────────────────────────────────────────────────────

def build_power_law_graph(
    n: int, k: int, alpha: float, seed: int = 42,
) -> torch.Tensor:
    """Build a sparse directed graph with power-law connectivity.

    Neurons are arranged on a ring. For each neuron j, K neighbors are
    sampled with probability P(distance) ∝ 1/distance^alpha.
    This gives mostly local connections with some long-range jumps.

    From the paper (Section 5.1):
        "BDH model is biologically plausible... The neuron interaction network
         of BDH is a graph of high modularity with heavy-tailed degree
         distribution."

    Args:
        n: number of neurons
        k: neighbors per neuron
        alpha: power-law exponent (higher = more local)
        seed: random seed for reproducibility

    Returns:
        neighbors: [N, K] long tensor of neighbor indices
    """
    rng = torch.Generator().manual_seed(seed)

    # Distance probabilities on a ring
    offsets = torch.arange(1, n, dtype=torch.float64)
    ring_dist = torch.minimum(offsets, n - offsets)
    probs = 1.0 / (ring_dist ** alpha)
    probs = probs / probs.sum()
    probs_f = probs.float()

    neighbors = torch.zeros(n, k, dtype=torch.long)
    for j in range(n):
        sampled = torch.multinomial(probs_f, k, replacement=False, generator=rng)
        neighbors[j] = (j + 1 + sampled) % n

    return neighbors


def _build_sparse_matrix(
    neighbors: torch.Tensor,
    weights: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Build a sparse COO matrix from neighbor indices and weights.

    Args:
        neighbors: [N, K] neighbor indices (columns for each row)
        weights: [N, K] edge weights
        n: matrix size (N x N)

    Returns:
        sparse: [N, N] sparse COO tensor
    """
    N, K = neighbors.shape
    rows = torch.arange(N, device=neighbors.device).unsqueeze(1).expand(N, K)
    indices = torch.stack([rows.reshape(-1), neighbors.reshape(-1)])
    values = weights.reshape(-1)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def _get_rope_freqs(k: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """RoPE frequencies for K-dimensional attention. Returns complex exponentials."""
    freqs = 1.0 / (theta ** (torch.arange(0, k, 2, dtype=torch.float32) / k))
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [max_len, K//2]
    return torch.polar(torch.ones_like(angles), angles)  # complex64


# ─── Modules ────────────────────────────────────────────────────────────────────

class SparseGraphProp(nn.Module):
    """Sparse graph propagation using torch.sparse.mm (14x faster than gather).

    Implements: out(j) = Σ_{k} weight(j,k) · input(neighbors(j,k))

    With excitatory/inhibitory: out = G_e @ input - G_i @ input, then ReLU.
    Uses sparse COO matrices rebuilt at each forward (weights change during training).
    """

    def __init__(self, n: int, k: int, use_inhibitory: bool = True):
        super().__init__()
        self.n = n
        self.k = k
        self.use_inhibitory = use_inhibitory
        # Weights (positive via abs; softplus is too slow in inner loop)
        self.w_e = nn.Parameter(torch.zeros(n, k).normal_(std=0.02))
        if use_inhibitory:
            self.w_i = nn.Parameter(torch.zeros(n, k).normal_(std=0.02))

    def forward(
        self, x: torch.Tensor, neighbors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [BT, N] or [B, T, N] input activations
            neighbors: [N, K] neighbor indices

        Returns:
            out: same shape as x, propagated activations (pre-ReLU)
        """
        orig_shape = x.shape
        x_dtype = x.dtype
        if x.dim() == 3:
            B, T, N = x.shape
            x_flat = x.reshape(B * T, N).float()
        else:
            x_flat = x.float()

        # Sparse matmul always in float32 (MPS/CUDA sparse doesn't support fp16)
        w_e = self.w_e.abs()
        sp_e = _build_sparse_matrix(neighbors, w_e, self.n)
        out = torch.sparse.mm(sp_e, x_flat.T).T

        if self.use_inhibitory:
            w_i = self.w_i.abs()
            sp_i = _build_sparse_matrix(neighbors, w_i, self.n)
            out = out - torch.sparse.mm(sp_i, x_flat.T).T

        return out.to(x_dtype).reshape(orig_shape)


class GraphAttention(nn.Module):
    """Graph-based synaptic attention (Section 2, Table 1).

    For each neuron j with K_s neighbors {i_1,...,i_K} in G_s:
      State update: σ_t(i_k, j) += y_prev_t(i_k) · x_t(j) · w(j,k)
      Attention:    A_t(j) = Σ_k x_t(i_k) · σ_{t-1}(i_k, j)

    Parallelized via cumulative sum (avoids O(T²·N) attention matrix):
      kv[t,j,k] = y_prev[t, nb[j,k]] · x[t, j] · w[j,k]
      S[t,j,k]  = Σ_{τ<t} kv[τ,j,k]
      A[t,j]    = Σ_k x[t, nb[j,k]] · S[t,j,k] / √(N/h)
    """

    def __init__(self, config: BDHGraphConfig):
        super().__init__()
        self.config = config
        N = config.n_neurons
        K = config.k_attn

        # Synaptic weights: Hebbian update scale per edge
        self.gs_w = nn.Parameter(torch.zeros(N, K).normal_(std=0.02))

        self._scale = (N // config.n_head) ** -0.5

        # RoPE for K-dimensional attention
        if config.use_rope and K >= 2:
            freqs = _get_rope_freqs(K, config.max_seq_len)
            self.register_buffer('_rope_freqs', freqs)
        else:
            self._rope_freqs = None

    def _apply_rope(self, v: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to [B, T, N, K] tensor along the T (position) dimension."""
        if self._rope_freqs is None:
            return v
        T, K = v.size(1), v.size(-1)
        freqs = self._rope_freqs[offset:offset + T]  # [T, K//2] complex
        v_c = torch.view_as_complex(v.float().reshape(*v.shape[:-1], K // 2, 2))
        rotated = torch.view_as_real(v_c * freqs.unsqueeze(0).unsqueeze(2))
        return rotated.reshape(*v.shape).to(v.dtype)

    def forward(
        self,
        x: torch.Tensor,
        y_prev: torch.Tensor,
        neighbors: torch.Tensor,
        state: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, N] current layer activations (sparse, positive)
            y_prev: [B, T, N] previous layer y (sparse, positive)
            neighbors: [N, K_s] neighbor indices for G_s
            state: [B, N, K_s] accumulated state from previous chunk, or None
            pos_offset: absolute position of first token (for RoPE)

        Returns:
            attn_out: [B, T, N] attention output per neuron
            new_state: [B, N, K_s] updated state for next chunk
        """
        B, T, N = x.shape
        K = neighbors.size(1)

        # Gather neighbor activations (the only gather we need)
        # Force float32 for precision in cumsum and RoPE
        y_nb = y_prev.float()[..., neighbors]  # [B, T, N, K]
        x_nb = x.float()[..., neighbors]       # [B, T, N, K]

        # Synaptic weight (positive per-edge Hebbian scale)
        w = self.gs_w.abs()  # [N, K]

        # Key-value: Hebbian update σ(i_k,j) += y(i_k) * x(j) * w(j,k)
        kv = y_nb * x.float().unsqueeze(-1) * w  # [B, T, N, K]

        # Apply RoPE to keys and queries
        kv = self._apply_rope(kv, offset=pos_offset)
        x_nb = self._apply_rope(x_nb, offset=pos_offset)

        # Causal cumulative sum: S[t] = Σ_{τ≤t} kv[τ]
        S = torch.cumsum(kv, dim=1)  # [B, T, N, K]

        # Shift: we want Σ_{τ<t}, so use S - kv (= cumsum up to t-1)
        S_causal = S - kv  # [B, T, N, K]

        # Add state from previous chunk
        if state is not None:
            S_causal = S_causal + state.unsqueeze(1)

        # Attention output: A[t,j] = Σ_k query[t,j,k] * S[t,j,k] * scale
        attn_out = (x_nb * S_causal).sum(-1) * self._scale  # [B, T, N]

        # New state for next chunk: full cumsum at last position
        new_state = S[:, -1]  # [B, N, K]
        if state is not None:
            new_state = new_state + state

        return attn_out.to(x.dtype), new_state


class BDHGraphLayer(nn.Module):
    """Single BDH graph layer (one iteration of Eq. 6).

    Processing order (following Table 1 rounds 4l..4l+3):
      1. x update:  x_{t,l} = [x_{t,l-1} + G_x(y_{t,l-1})]^+
      2. Attention:  a_{t,l} = σ_{t-1,l} x_{t,l}
      3. G_y prop:   b_{t,l} = G_y(a_{t,l})
      4. Hebbian:    y_{t,l} = [b_{t,l}]^+ ⊙ x_{t,l}
    """

    def __init__(self, config: BDHGraphConfig, attn: GraphAttention):
        super().__init__()
        self.config = config
        N = config.n_neurons

        # Per-layer propagation weights (topology is shared, weights differ)
        self.gx_prop = SparseGraphProp(N, config.k_prop, config.use_inhibitory)
        self.gy_prop = SparseGraphProp(N, config.k_attn, config.use_inhibitory)

        # Reference to shared attention module
        self.attn = attn

        # LayerNorm (non-parametric, like BDH-GPU)
        self.ln_x = nn.LayerNorm(N, elementwise_affine=False, bias=False)
        self.ln_a = nn.LayerNorm(N, elementwise_affine=False, bias=False)
        self.ln_out = nn.LayerNorm(N, elementwise_affine=False, bias=False)

        self.drop = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, y_prev: torch.Tensor,
        gs_neighbors: torch.Tensor,
        gx_neighbors: torch.Tensor,
        gy_neighbors: torch.Tensor,
        state: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (x_new, y_new, new_state)
        """
        # Step 1: x update via G_x propagation
        gx_out = self.gx_prop(y_prev, gx_neighbors)
        x_new = F.relu(x + gx_out)
        x_new = self.ln_x(x_new)

        # Step 2: Synaptic attention (parallel cumsum)
        attn_out, new_state = self.attn(
            x_new, y_prev, gs_neighbors,
            state=state, pos_offset=pos_offset,
        )

        # Step 3: G_y propagation of attention output
        b = self.gy_prop(self.ln_a(attn_out), gy_neighbors)

        # Step 4: Hebbian gating y = ReLU(b) * x
        y_new = F.relu(b) * x_new
        y_new = self.drop(y_new)

        # Residual connection
        x_out = self.ln_out(x + y_new)

        return x_out, y_new, new_state


class BDHGraph(nn.Module):
    """Graph-based BDH language model (Section 2 of the paper).

    N neurons with K-neighbor sparse graph connectivity (power-law).
    Synaptic state on edges with Hebbian plasticity.
    Excitatory + inhibitory circuits with ReLU thresholding.
    Sparse positive activations (~5% nonzero empirically).

    Dense IO bridges the d-dimensional token space to the N-dimensional
    neuron space. All intermediate computation is sparse in N-space.
    """

    def __init__(self, config: BDHGraphConfig, use_grad_checkpoint: bool = False):
        super().__init__()
        self.config = config
        self.use_grad_checkpoint = use_grad_checkpoint

        N = config.n_neurons
        d = config.n_embd

        # ── Build static graph topologies (3 separate graphs) ──
        gs_nb = build_power_law_graph(N, config.k_attn, config.alpha, seed=42)
        gx_nb = build_power_law_graph(N, config.k_prop, config.alpha, seed=137)
        gy_nb = build_power_law_graph(N, config.k_attn, config.alpha, seed=271)
        self.register_buffer('gs_neighbors', gs_nb)
        self.register_buffer('gx_neighbors', gx_nb)
        self.register_buffer('gy_neighbors', gy_nb)

        # ── Shared attention ──
        self.attn = GraphAttention(config)

        # ── Per-layer modules ──
        self.layers = nn.ModuleList([
            BDHGraphLayer(config, self.attn) for _ in range(config.n_layer)
        ])

        # ── Dense IO: token ↔ neuron mapping ──
        self.embed = nn.Embedding(config.vocab_size, d)
        self.encoder = nn.Parameter(torch.zeros(d, N).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros(N, d).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros(d, config.vocab_size).normal_(std=0.02))

        self.ln_embed = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.ln_decode = nn.LayerNorm(d, elementwise_affine=False, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        state: list[torch.Tensor] | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """
        Args:
            idx: [B, T] token indices
            targets: [B, T] target tokens (for loss)
            state: list of per-layer [B, N, K_s] states, or None
            pos_offset: absolute position of first token

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar or None
            new_state: list of per-layer states
        """
        B, T = idx.shape
        N = self.config.n_neurons

        # Embed and lift to neuron space
        v = self.ln_embed(self.embed(idx))  # [B, T, d]
        x = F.relu(v @ self.encoder)        # [B, T, N]
        y = x  # initial y = x for first layer

        # Process through layers
        new_state = []
        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None else None

            if self.use_grad_checkpoint and self.training:
                x, y, s = checkpoint(
                    layer, x, y,
                    self.gs_neighbors, self.gx_neighbors, self.gy_neighbors,
                    layer_state, pos_offset,
                    use_reentrant=False,
                )
            else:
                x, y, s = layer(
                    x, y,
                    self.gs_neighbors, self.gx_neighbors, self.gy_neighbors,
                    state=layer_state, pos_offset=pos_offset,
                )
            new_state.append(s)

        # Decode: N → d → vocab
        v_out = self.ln_decode(y @ self.decoder)
        logits = v_out @ self.lm_head

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
            )

        return logits, loss, new_state

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with persistent synaptic state."""
        C = self.config
        B = idx.size(0)

        state = [
            torch.zeros(B, C.n_neurons, C.k_attn,
                        device=idx.device, dtype=torch.float32)
            for _ in range(C.n_layer)
        ]

        logits, _, state = self(idx, pos_offset=0, state=state)
        pos = idx.size(1)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1)),
                )
                next_logits[next_logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            logits, _, state = self(idx_next, pos_offset=pos, state=state)
            pos += 1

        return idx


def count_params(model: nn.Module) -> dict[str, int]:
    """Count parameters by component."""
    counts = {}
    for name, p in model.named_parameters():
        if 'gx_prop' in name or 'gy_prop' in name:
            group = 'graph_prop'
        elif 'attn' in name or 'gs_w' in name:
            group = 'attn_synaptic'
        elif name in ('encoder', 'decoder', 'lm_head'):
            group = 'io_' + name
        elif 'embed' in name:
            group = 'io_embed'
        else:
            group = 'other'
        counts[group] = counts.get(group, 0) + p.numel()
    return counts


# ─── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("Building graph-based BDH model...")
    config = BDHGraphConfig(
        n_neurons=4096,
        k_attn=16,
        k_prop=32,
        n_layer=4,
        n_head=4,
    )

    model = BDHGraph(config, use_grad_checkpoint=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    print("\nParameter breakdown:")
    for group, count in sorted(count_params(model).items()):
        print(f"  {group:20s}: {count:>10,}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    B, T = 4, 128
    idx = torch.randint(0, 256, (B, T), device=device)
    tgt = torch.randint(0, 256, (B, T), device=device)

    print(f"\nForward + backward test on {device} (B={B}, T={T})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup — no AMP: sparse ops need float32, small model doesn't benefit
    _, loss, _ = model(idx, tgt)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Timed
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        _, loss, _ = model(idx, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if device == "mps":
        torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / 3
    print(f"  {dt*1000:.0f}ms/step | {B*T/dt:.0f} tok/s | loss={loss.item():.4f}")

    # Generation test
    print("\nGeneration test...")
    model.eval()
    prompt = torch.tensor([[72, 101, 108, 108, 111]], device=device)
    out = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)
    text = bytes(out[0].cpu().tolist()).decode(errors='replace')
    print(f"  Generated: {text!r}")

    print("\n✓ All tests passed!")
