# Copyright 2025 Pathway Technology, Inc.
# BDH with recurrent synaptic state (continuous learning) as described in the paper.
#
# Key insight from the paper (Eq. 8, Appendix B.3):
#   BDH is a state-space model. The synaptic state ρ accumulates across time via
#   Hebbian learning: ρ_{t,l} := ρ_{t-1,l} + v*_{t,l-1} · x_{t,l}^T · U
#
#   In the GPU-parallel implementation, ρ is implicitly computed by the attention
#   mechanism within a single sequence. But for continuous learning across chunks,
#   ρ must persist: the model should carry over its synaptic state from one chunk
#   of text to the next, trained with Truncated BPTT.
#
# Implementation:
#   - Attention.forward() accepts optional recurrent state from previous chunk
#   - Returns updated state for the next chunk
#   - Within a chunk: standard parallel attention (fast)
#   - Across chunks: state captures accumulated QR^T @ V (the paper's ρ)
#   - Training uses sequential chunks with state carryover + gradient detach

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 2
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    max_seq_len: int = 512


def _get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    """BDH attention with optional recurrent synaptic state.

    Supports two modes:
      1. Stateless (state=None): standard parallel attention, no cross-chunk memory.
         Equivalent to the original implementation.
      2. Stateful (state=tensor): adds contribution from previous chunks via the
         accumulated synaptic state ρ = Σ_{τ<t} QR[τ]^T @ V[τ] * scale.

    The state has shape (B, n_head, N, D) — the outer product of keys and values
    accumulated over all previous tokens. This IS the paper's synaptic state ρ.
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        freqs = _get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        self.freqs = nn.Buffer(freqs)

        # Precompute RoPE cos/sin for all positions up to max_seq_len.
        r_phases = (
            torch.arange(0, config.max_seq_len, dtype=torch.float32).view(1, 1, -1, 1)
            * freqs
        )
        angles = (r_phases % 1) * (2 * math.pi)
        self._cos_cached = nn.Buffer(torch.cos(angles))
        self._sin_cached = nn.Buffer(torch.sin(angles))

        self._scale = N**-0.5

    def rope(self, v: torch.Tensor, T: int, offset: int = 0) -> torch.Tensor:
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos = self._cos_cached[:, :, offset : offset + T, :]
        sin = self._sin_cached[:, :, offset : offset + T, :]
        return (v * cos).to(v.dtype) + (v_rot * sin).to(v.dtype)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        state: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional recurrent state.

        Args:
            Q, K: (B, n_head, T, N) — sparse activations after ReLU
            V: (B, 1, T, D) — input embeddings (broadcast across heads)
            state: (B, n_head, N, D) or None — accumulated synaptic state ρ
            pos_offset: absolute position of the first token in this chunk.
                        Required for correct RoPE when using recurrent state.

        Returns:
            output: (B, n_head, T, D) — attention output
            new_state: (B, n_head, N, D) — updated state for next chunk
        """
        assert K is Q  # BDH uses Q=K (self-attention)
        _, _, T, N = Q.size()

        QR = self.rope(Q, T, offset=pos_offset)

        # Within-chunk attention (parallel, causal)
        scores = (QR @ QR.mT * self._scale).tril(diagonal=-1)
        output = scores @ V  # (B, nh, T, D)

        # Cross-chunk contribution from accumulated state
        if state is not None:
            # state: (B, nh, N, D) — accumulated QR^T @ V from all previous chunks
            # QR @ state gives each position's attention to all previous chunks
            output = output + QR @ state  # (B, nh, T, N) @ (B, nh, N, D)

        # Update state: accumulate this chunk's QR^T @ V
        # QR^T @ V: (B, nh, N, T) @ (B, 1, T, D) → (B, nh, N, D)
        chunk_state = QR.transpose(-2, -1) @ V * self._scale
        if state is not None:
            new_state = state + chunk_state
        else:
            new_state = chunk_state

        return output, new_state


class BDHLayer(nn.Module):
    """Single BDH computation layer."""

    def __init__(
        self, attn: Attention, ln: nn.LayerNorm, drop: nn.Dropout,
        encoder: nn.Parameter, encoder_v: nn.Parameter, decoder: nn.Parameter,
        config: BDHConfig,
    ):
        super().__init__()
        self.attn = attn
        self.ln = ln
        self.drop = drop
        self.encoder = encoder
        self.encoder_v = encoder_v
        self.decoder = decoder
        self.config = config

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C = self.config
        B, _, T, D = x.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x_latent = x @ self.encoder
        x_sparse = F.relu(x_latent)

        yKV, new_state = self.attn(
            Q=x_sparse, K=x_sparse, V=x, state=state, pos_offset=pos_offset,
        )
        yKV = self.ln(yKV)

        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse  # Hebbian gating

        xy_sparse = self.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)
        return self.ln(x + y), new_state


class BDH(nn.Module):
    """BDH language model with recurrent synaptic state.

    The model maintains a per-layer synaptic state ρ that accumulates across
    chunks of text, implementing the paper's continuous Hebbian learning.

    Training modes:
      1. Stateless (default): each forward pass is independent. Fast, simple,
         equivalent to the original implementation.
      2. Stateful (pass state): state carries over between chunks for
         continuous learning via truncated BPTT.
    """

    def __init__(self, config: BDHConfig, use_grad_checkpoint: bool = False):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        self.use_grad_checkpoint = use_grad_checkpoint

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.layers = nn.ModuleList([
            BDHLayer(
                self.attn, self.ln, self.drop,
                encoder=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
                encoder_v=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
                decoder=nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02)),
                config=config,
            )
            for _ in range(config.n_layer)
        ])

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
        """Forward pass with optional recurrent state.

        Args:
            idx: (B, T) token indices
            targets: (B, T) target token indices for loss computation
            state: list of per-layer states, or None for stateless mode
            pos_offset: absolute position of the first token in this chunk

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
            new_state: list of per-layer states for next chunk
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        new_state = []
        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None else None
            if self.use_grad_checkpoint and self.training:
                x, s = checkpoint(
                    layer, x, layer_state, pos_offset, use_reentrant=False,
                )
            else:
                x, s = layer(x, layer_state, pos_offset)
            new_state.append(s)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_state

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with persistent synaptic state.

        On the first call, processes the full prompt and builds the state.
        On subsequent tokens, feeds only the new token + state, making
        generation O(T) per token instead of O(T²).
        """
        # First pass: encode full prompt, build state
        logits, _, state = self(idx, pos_offset=0)
        pos = idx.size(1)  # next absolute position

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Feed only the new token + state (incremental)
            logits, _, state = self(idx_next, pos_offset=pos, state=state)
            pos += 1

        return idx
