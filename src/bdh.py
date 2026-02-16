# Copyright 2025 Pathway Technology, Inc.
# Optimized for Apple M1 (8GB) by caching RoPE and reducing intermediate allocations.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    max_seq_len: int = 512  # for RoPE cache precomputation


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        freqs = get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        self.freqs = torch.nn.Buffer(freqs)

        # Precompute cos/sin for all positions up to max_seq_len.
        # This avoids recomputing trig functions every forward call (6 layers x every step).
        # Cost: 2 * max_seq_len * N * 4 bytes = 2 * 512 * 8192 * 4 = 32 MB
        r_phases = (
            torch.arange(0, config.max_seq_len, dtype=torch.float32).view(1, 1, -1, 1)
            * freqs
        )
        angles = (r_phases % 1) * (2 * math.pi)
        self._cos_cached = torch.nn.Buffer(torch.cos(angles))
        self._sin_cached = torch.nn.Buffer(torch.sin(angles))

    def rope(self, v, T):
        """Apply RoPE using precomputed cos/sin tables."""
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos = self._cos_cached[:, :, :T, :]
        sin = self._sin_cached[:, :, :T, :]
        return (v * cos).to(v.dtype) + (v_rot * sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert K is Q
        _, _, T, _ = Q.size()

        QR = self.rope(Q, T)
        # K is Q, so KR is QR — no redundant RoPE computation
        scores = (QR @ QR.mT).tril(diagonal=-1)
        return scores @ V


class BDHLayer(nn.Module):
    """Single BDH layer extracted for gradient checkpointing."""

    def __init__(self, attn, ln, drop, encoder, encoder_v, decoder, config):
        super().__init__()
        self.attn = attn
        self.ln = ln
        self.drop = drop
        self.encoder = encoder
        self.encoder_v = encoder_v
        self.decoder = decoder
        self.config = config

    def forward(self, x):
        C = self.config
        B = x.size(0)
        T = x.size(2)
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x_latent = x @ self.encoder
        x_sparse = F.relu(x_latent)

        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = self.ln(yKV)

        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse

        xy_sparse = self.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)
        return self.ln(x + y)


class BDH(nn.Module):
    def __init__(self, config: BDHConfig, use_grad_checkpoint: bool = False):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        self.use_grad_checkpoint = use_grad_checkpoint

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        # Wrap each layer for gradient checkpointing
        self.layer = BDHLayer(
            self.attn,
            self.ln,
            self.drop,
            self.encoder,
            self.encoder_v,
            self.decoder,
            config,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for level in range(C.n_layer):
            if self.use_grad_checkpoint and self.training:
                # Gradient checkpointing: trade compute for memory.
                # Instead of storing all intermediates for backward (6 layers of
                # (B,nh,T,N=8192) tensors), recompute them during backward.
                # Saves ~60% peak memory at cost of ~30% more compute.
                x = checkpoint(self.layer, x, use_reentrant=False)
            else:
                x = self.layer(x)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len (RoPE cache limit)
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
