# Copyright 2025 Pathway Technology, Inc.
# BDH-HRM Hybrid: Baby Dragon Hatchling with Hierarchical Reasoning Model structure.
#
# Combines:
#   - BDH: sparse ReLU activations, RoPE attention, biologically-inspired gating
#   - HRM: two-level hierarchical reasoning (H-level slow/abstract, L-level fast/detail),
#     iterative refinement cycles, no-grad trick for memory efficiency
#
# Key insight from HRM paper: the outer-loop refinement (iterating H and L levels
# multiple times) is what drives reasoning gains. BDH's weight-shared layers and
# Hebbian-style multiplicative gating (x_sparse * y_sparse) make it a natural fit
# for this iterative refinement — the same "circuit" is applied repeatedly, refining
# the representation each time.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclasses.dataclass
class BDHHRMConfig:
    # BDH parameters
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    max_seq_len: int = 512

    # HRM parameters
    h_cycles: int = 3  # outer (high-level) reasoning cycles
    l_cycles: int = 2  # inner (low-level) reasoning cycles per h_cycle
    # Total iterations = h_cycles * l_cycles = 6 (same compute as original 6 layers)

    # Ponder loss: regularize total computation depth
    ponder_loss_weight: float = 0.01


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

        r_phases = (
            torch.arange(0, config.max_seq_len, dtype=torch.float32).view(1, 1, -1, 1)
            * freqs
        )
        angles = (r_phases % 1) * (2 * math.pi)
        self._cos_cached = torch.nn.Buffer(torch.cos(angles))
        self._sin_cached = torch.nn.Buffer(torch.sin(angles))

    def rope(self, v, T):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos = self._cos_cached[:, :, :T, :]
        sin = self._sin_cached[:, :, :T, :]
        return (v * cos).to(v.dtype) + (v_rot * sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert K is Q
        _, _, T, N = Q.size()
        QR = self.rope(Q, T)
        scores = (QR @ QR.mT * (N ** -0.5)).tril(diagonal=-1)
        return scores @ V


class BDHBlock(nn.Module):
    """A single BDH processing block.

    This is BDH's core computation unit: encode → sparse attention → gated MLP decode.
    Used as the building block for both H-level and L-level reasoning modules.
    """

    def __init__(self, config, attn):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))

        self.attn = attn
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, injection):
        """Process x with input injection from the other level.

        Following HRM: the input injection adds information from the other reasoning
        level (H→L or L→H) before processing. This cross-level communication is what
        enables hierarchical reasoning.

        Args:
            x: current level's state (B, 1, T, D)
            injection: input from other level (B, 1, T, D)
        Returns:
            updated state (B, 1, T, D)
        """
        C = self.config
        B = x.size(0)
        T = x.size(2)
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Input injection (HRM-style: additive)
        x_injected = self.ln(x + injection)

        # BDH encode → sparse
        x_latent = x_injected @ self.encoder
        x_sparse = F.relu(x_latent)  # B, nh, T, N

        # BDH attention (Q=K=sparse representation, V=injected input)
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x_injected)
        yKV = self.ln(yKV)

        # BDH gated decode (Hebbian-style multiplicative interaction)
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse  # Hebbian gating
        xy_sparse = self.drop(xy_sparse)

        # Decode back to embedding space
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)

        # Residual connection
        return self.ln(x + y)


class BDHHRM(nn.Module):
    """BDH-HRM Hybrid: Hierarchical Reasoning with BDH's biologically-inspired blocks.

    Architecture:
        - H-level (high): slow, abstract reasoning (updated once per H-cycle)
        - L-level (low): fast, detailed computation (updated L-cycles times per H-cycle)
        - Cross-level injection: L receives (H + input), H receives L

    The iterative structure mirrors HRM, but uses BDH's sparse attention and Hebbian
    gating instead of standard transformer blocks. The key memory trick from HRM: all
    iterations except the last run in torch.no_grad(), so only 1 iteration's activations
    are stored for backward (instead of h_cycles * l_cycles iterations).
    """

    def __init__(self, config: BDHHRMConfig, use_grad_checkpoint: bool = False):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        self.use_grad_checkpoint = use_grad_checkpoint

        D = config.n_embd

        # Shared attention (RoPE is position-based, shared across levels)
        self.attn = Attention(config)

        # Two BDH blocks: one for H-level, one for L-level
        # Separate parameters = each level learns different representations
        self.h_block = BDHBlock(config, self.attn)
        self.l_block = BDHBlock(config, self.attn)

        # I/O
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        # Initial states for H and L (learned, following HRM)
        self.h_init = nn.Parameter(torch.zeros(D).normal_(std=0.02))
        self.l_init = nn.Parameter(torch.zeros(D).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_l_step(self, z_L, z_H, input_emb):
        """L-level: fast, detailed processing. Receives H + input as injection."""
        return self.l_block(z_L, z_H + input_emb)

    def _run_h_step(self, z_H, z_L):
        """H-level: slow, abstract planning. Receives L as injection."""
        return self.h_block(z_H, z_L)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd

        # Input embeddings
        input_emb = self.embed(idx).unsqueeze(1)  # B, 1, T, D
        input_emb = self.ln(input_emb)

        # Initialize H and L states (broadcast learned init to batch)
        z_H = self.h_init.view(1, 1, 1, D).expand(B, 1, T, D).clone()
        z_L = self.l_init.view(1, 1, 1, D).expand(B, 1, T, D).clone()

        # ─── HRM-style iterative refinement ───
        # All iterations except the last run in no_grad (HRM's key memory trick).
        # This means we can do h_cycles * l_cycles = many iterations while only
        # storing activations for the final H and L steps.
        total_iters = C.h_cycles * C.l_cycles

        with torch.no_grad():
            for h_step in range(C.h_cycles):
                for l_step in range(C.l_cycles):
                    is_last = (h_step == C.h_cycles - 1) and (l_step == C.l_cycles - 1)
                    if not is_last:
                        z_L = self._run_l_step(z_L, z_H, input_emb)

                if h_step < C.h_cycles - 1:
                    z_H = self._run_h_step(z_H, z_L)

        # Ensure z_H and z_L don't carry gradients from no_grad block
        assert not z_H.requires_grad and not z_L.requires_grad

        # ─── Final iteration WITH gradients ───
        # No gradient checkpointing needed here: the no-grad trick above means we only
        # store activations for these 2 steps (L + H), not all h*l iterations.
        z_L = self._run_l_step(z_L, z_H, input_emb)
        z_H = self._run_h_step(z_H, z_L)

        # Output from H-level (abstract representation → logits)
        logits = z_H.view(B, T, D) @ self.lm_head

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            # Ponder loss: regularize computation depth (following HRM)
            # For fixed iteration count, this is a constant but keeps the interface
            # ready for adaptive halting
            ponder_loss = torch.tensor(
                float(total_iters), device=logits.device, dtype=logits.dtype
            )
            loss = lm_loss + C.ponder_loss_weight * ponder_loss

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
