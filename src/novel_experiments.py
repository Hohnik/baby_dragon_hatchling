"""Novel Architecture Experiments for BDH.

Tests multiple novel approaches against the baseline, each isolated.
All experiments use identical seed, data, and training hyperparams.

Novel approaches tested:
  A. per_head_v     — Per-head value projection (W_V), already in config
  B. swiglu         — SwiGLU encoding instead of ReLU
  C. topk           — Top-K sparsification instead of ReLU thresholding
  D. rmsnorm        — RMSNorm instead of LayerNorm
  E. diff_attn      — Differential Attention (noise cancellation)
  F. learned_temp   — Learnable per-head attention temperature
  G. gated_residual — Gated residual connection (x + gate * y)
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import bdh


# ─── Novel Components ────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """RMSNorm: normalize by root mean square, no mean subtraction.
    Used in LLaMA, Mistral, Gemma, etc. Simpler and faster than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class BDHLayerSwiGLU(bdh.BDHLayer):
    """BDH Layer with SwiGLU encoding instead of ReLU.

    SwiGLU (Shazeer 2020): out = (x @ W_gate) * σ(x @ W_up) then project down.
    This replaces the ReLU(x @ encoder) step. The gate provides smoother gradients
    and better training dynamics than hard ReLU thresholding.

    To keep params comparable, we split the encoder into gate and up projections,
    each of size N/2, so total encoder size stays the same.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        # Split encoder into gate + up projections (each N//2)
        # Use existing encoder as gate, add new up projection
        self.encoder_up = nn.Parameter(
            torch.zeros((nh, D, N)).normal_(std=0.02)
        )

    def forward(self, x, state=None, pos_offset=0):
        C = self.config
        B, _, T, D = x.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # SwiGLU encode: gate * silu(up)
        enc_gate_flat = self.encoder.permute(1, 0, 2).reshape(D, nh * N)
        enc_up_flat = self.encoder_up.permute(1, 0, 2).reshape(D, nh * N)
        x_flat = x.squeeze(1).reshape(B * T, D)

        gate = (x_flat @ enc_gate_flat).view(B, T, nh, N).permute(0, 2, 1, 3)
        up = (x_flat @ enc_up_flat).view(B, T, nh, N).permute(0, 2, 1, 3)
        x_sparse = gate * F.silu(up)  # SwiGLU activation

        V = x if self.value_proj is None else x @ self.value_proj

        yKV, new_state = self.attn(
            Q=x_sparse, K=x_sparse, V=V, state=state, pos_offset=pos_offset,
        )
        yKV = self.ln(yKV)

        y_latent = torch.einsum("bhtd,hdn->bhtn", yKV, self.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse

        xy_sparse = self.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)
        return self.ln(x + y), new_state


class BDHLayerTopK(bdh.BDHLayer):
    """BDH Layer with Top-K sparsification instead of ReLU.

    Inspired by RAM-Net (2602.11958): explicit sparse addressing via top-k
    selection gives exact control over sparsity level and reduces signal
    interference. Instead of ReLU's variable sparsity (~50% on average),
    top-k selects exactly K active dimensions per token.

    Default K = N // 4 (75% sparsity).
    """

    def __init__(self, *args, top_k_ratio: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_ratio = top_k_ratio

    def _topk_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Keep only top-K values, zero the rest."""
        k = max(1, int(x.size(-1) * self.top_k_ratio))
        vals, idx = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, idx, vals)
        return out

    def forward(self, x, state=None, pos_offset=0):
        C = self.config
        B, _, T, D = x.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        enc_flat = self.encoder.permute(1, 0, 2).reshape(D, nh * N)
        x_flat = x.squeeze(1).reshape(B * T, D)
        x_latent = (x_flat @ enc_flat).view(B, T, nh, N).permute(0, 2, 1, 3)
        x_sparse = self._topk_sparse(x_latent)  # Top-K instead of ReLU

        V = x if self.value_proj is None else x @ self.value_proj

        yKV, new_state = self.attn(
            Q=x_sparse, K=x_sparse, V=V, state=state, pos_offset=pos_offset,
        )
        yKV = self.ln(yKV)

        y_latent = torch.einsum("bhtd,hdn->bhtn", yKV, self.encoder_v)
        y_sparse = self._topk_sparse(y_latent)  # Top-K for encoder_v too
        xy_sparse = x_sparse * y_sparse

        xy_sparse = self.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)
        return self.ln(x + y), new_state


class DifferentialAttention(bdh.Attention):
    """Differential Attention for BDH (inspired by DIFF Transformer, Microsoft).

    Pairs of heads compute attention, and the output is the DIFFERENCE between
    the two attention patterns: attn_out = (attn1 - λ * attn2) @ V.
    This cancels noise and amplifies signal.

    With n_head=4, we get 2 differential groups (4/2).
    λ is a learned per-group scalar initialized near 0.5.
    """

    def __init__(self, config: bdh.BDHConfig):
        super().__init__(config)
        n_groups = config.n_head // 2
        # λ initialized so sigmoid(0) = 0.5
        self.lambda_param = nn.Parameter(torch.zeros(1, n_groups, 1, 1))

    def forward(self, Q, K, V, state=None, pos_offset=0):
        assert K is Q
        B, nh, T, N = Q.size()
        n_groups = nh // 2

        QR = self.rope(Q, T, offset=pos_offset)

        # Split into pairs
        QR1 = QR[:, 0::2, :, :]  # (B, n_groups, T, N) - even heads
        QR2 = QR[:, 1::2, :, :]  # (B, n_groups, T, N) - odd heads

        scores1 = (QR1 @ QR1.mT * self._scale).tril(diagonal=-1)
        scores2 = (QR2 @ QR2.mT * self._scale).tril(diagonal=-1)

        # Differential: cancel noise
        lam = torch.sigmoid(self.lambda_param)
        diff_scores = scores1 - lam * scores2  # (B, n_groups, T, T)

        # V is (B, 1, T, D) — expand to groups
        output_groups = diff_scores @ V.expand(B, n_groups, T, -1)

        # Interleave back to full head dimension for compatibility
        output = torch.zeros(B, nh, T, V.size(-1), device=Q.device, dtype=Q.dtype)
        output[:, 0::2] = output_groups
        output[:, 1::2] = output_groups  # Both sub-heads get same output

        if state is not None:
            # For state, use the combined approach
            if hasattr(self, '_forget_mode') and self._forget_mode == "data":
                x_mean = V.squeeze(1).mean(dim=1)
                forget_gate = torch.sigmoid(self.forget_proj(x_mean))
                forget_gate = forget_gate.unsqueeze(-1).unsqueeze(-1)
            elif hasattr(self, '_forget_mode') and self._forget_mode == "scalar":
                forget_gate = torch.sigmoid(self.forget_bias)
            else:
                forget_gate = None

            if forget_gate is not None:
                gated_state = forget_gate * state
            else:
                gated_state = state
            output = output + QR @ gated_state
            chunk_state = QR.transpose(-2, -1) @ V * self._scale
            new_state = gated_state + chunk_state
        else:
            new_state = None

        return output, new_state


class BDHLayerGatedResidual(bdh.BDHLayer):
    """BDH Layer with gated residual connection.

    Instead of simple x + y, uses x + gate * y where gate is a learned
    per-dimension scalar. This lets the model control how much each layer
    contributes, potentially enabling deeper models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        D = self.config.n_embd
        # Initialize gate near 1.0 (so initial behavior ≈ standard residual)
        self.res_gate = nn.Parameter(torch.ones(D) * 0.5)

    def forward(self, x, state=None, pos_offset=0):
        C = self.config
        B, _, T, D = x.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        enc_flat = self.encoder.permute(1, 0, 2).reshape(D, nh * N)
        x_flat = x.squeeze(1).reshape(B * T, D)
        x_latent = (x_flat @ enc_flat).view(B, T, nh, N).permute(0, 2, 1, 3)
        x_sparse = F.relu(x_latent)

        V = x if self.value_proj is None else x @ self.value_proj

        yKV, new_state = self.attn(
            Q=x_sparse, K=x_sparse, V=V, state=state, pos_offset=pos_offset,
        )
        yKV = self.ln(yKV)

        y_latent = torch.einsum("bhtd,hdn->bhtn", yKV, self.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse

        xy_sparse = self.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)

        # Gated residual instead of simple addition
        gate = torch.sigmoid(self.res_gate)
        return self.ln(x + gate * y), new_state


class BDHLayerLearnedTemp(bdh.BDHLayer):
    """BDH Layer with learned per-head attention temperature.

    Instead of fixed 1/√N scaling, each head learns its own temperature.
    Initialized to 1/√N to match baseline behavior.
    """

    def forward(self, x, state=None, pos_offset=0):
        # This is handled at the attention level - pass through
        return super().forward(x, state, pos_offset)


class LearnedTempAttention(bdh.Attention):
    """Attention with learned per-head temperature scaling."""

    def __init__(self, config):
        super().__init__(config)
        N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
        # Initialize to log(1/√N) so exp(param) = 1/√N at init
        init_val = -0.5 * math.log(N)
        self.log_temp = nn.Parameter(
            torch.full((1, config.n_head, 1, 1), init_val)
        )

    def forward(self, Q, K, V, state=None, pos_offset=0):
        assert K is Q
        _, _, T, N = Q.size()

        QR = self.rope(Q, T, offset=pos_offset)

        # Learned temperature instead of fixed 1/√N
        scale = torch.exp(self.log_temp)  # per-head learned scale
        scores = (QR @ QR.mT * scale).tril(diagonal=-1)
        output = scores @ V

        if state is not None:
            # Forget gate logic
            if self._forget_mode == "data":
                x_mean = V.squeeze(1).mean(dim=1)
                forget_gate = torch.sigmoid(self.forget_proj(x_mean))
                forget_gate = forget_gate.unsqueeze(-1).unsqueeze(-1)
            elif self._forget_mode == "scalar":
                forget_gate = torch.sigmoid(self.forget_bias)
            else:
                forget_gate = None

            if forget_gate is not None:
                gated_state = forget_gate * state
            else:
                gated_state = state
            output = output + QR @ gated_state
            chunk_state = QR.transpose(-2, -1) @ V * scale
            new_state = gated_state + chunk_state
        else:
            new_state = None

        return output, new_state


# ─── Model Builders ──────────────────────────────────────────────────────────────


def build_baseline(config: bdh.BDHConfig) -> bdh.BDH:
    """Standard BDH model (baseline)."""
    return bdh.BDH(config, use_grad_checkpoint=True)


def build_per_head_v(config: bdh.BDHConfig) -> bdh.BDH:
    """BDH with per-head value projections."""
    cfg = copy.copy(config)
    cfg.per_head_v = True
    return bdh.BDH(cfg, use_grad_checkpoint=True)


def build_swiglu(config: bdh.BDHConfig) -> bdh.BDH:
    """BDH with SwiGLU encoding."""
    model = bdh.BDH(config, use_grad_checkpoint=True)
    nh = config.n_head
    D = config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh

    new_layers = nn.ModuleList()
    for i in range(config.n_layer):
        layer = BDHLayerSwiGLU(
            model.attn, model.ln, model.drop,
            encoder=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
            encoder_v=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
            decoder=nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02)),
            config=config,
        )
        new_layers.append(layer)
    model.layers = new_layers
    return model


def build_topk(config: bdh.BDHConfig, ratio: float = 0.25) -> bdh.BDH:
    """BDH with Top-K sparsification."""
    model = bdh.BDH(config, use_grad_checkpoint=True)
    nh = config.n_head
    D = config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh

    new_layers = nn.ModuleList()
    for i in range(config.n_layer):
        layer = BDHLayerTopK(
            model.attn, model.ln, model.drop,
            encoder=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
            encoder_v=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
            decoder=nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02)),
            config=config,
            top_k_ratio=ratio,
        )
        new_layers.append(layer)
    model.layers = new_layers
    return model


def build_rmsnorm(config: bdh.BDHConfig) -> bdh.BDH:
    """BDH with RMSNorm instead of LayerNorm."""
    model = bdh.BDH(config, use_grad_checkpoint=True)
    model.ln = RMSNorm(config.n_embd)
    for layer in model.layers:
        layer.ln = model.ln
    return model


def build_diff_attn(config: bdh.BDHConfig) -> bdh.BDH:
    """BDH with Differential Attention."""
    model = bdh.BDH(config, use_grad_checkpoint=True)
    diff_attn = DifferentialAttention(config).to(next(model.parameters()).device)
    model.attn = diff_attn
    for layer in model.layers:
        layer.attn = diff_attn
    return model


def build_learned_temp(config: bdh.BDHConfig) -> bdh.BDH:
    """BDH with learned per-head attention temperature."""
    model = bdh.BDH(config, use_grad_checkpoint=True)
    lt_attn = LearnedTempAttention(config).to(next(model.parameters()).device)
    model.attn = lt_attn
    for layer in model.layers:
        layer.attn = lt_attn
    return model


def build_gated_residual(config: bdh.BDHConfig) -> bdh.BDH:
    """BDH with gated residual connections."""
    model = bdh.BDH(config, use_grad_checkpoint=True)
    nh = config.n_head
    D = config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh

    new_layers = nn.ModuleList()
    for i in range(config.n_layer):
        layer = BDHLayerGatedResidual(
            model.attn, model.ln, model.drop,
            encoder=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
            encoder_v=nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)),
            decoder=nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02)),
            config=config,
        )
        new_layers.append(layer)
    model.layers = new_layers
    return model


# ─── Training Harness ────────────────────────────────────────────────────────────


def train_and_eval(
    model: nn.Module,
    name: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
    steps: int = 200,
    batch_size: int = 4,
    block_size: int = 256,
    lr: float = 1e-3,
    warmup: int = 50,
) -> dict:
    """Train a model and return metrics."""
    torch.manual_seed(1337)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())

    class Streamer:
        def __init__(self, data, bs):
            self.data, self.bs = data, bs
            sl = len(data) // bs
            self.cursors = [i * sl for i in range(bs)]

        def next_batch(self, bsz):
            xs, ys = [], []
            for i in range(self.bs):
                st = self.cursors[i]
                e = st + bsz + 1
                if e > len(self.data):
                    st, e = 0, bsz + 1
                ch = self.data[st:e]
                xs.append(ch[:-1])
                ys.append(ch[1:])
                self.cursors[i] = st + bsz
            return torch.stack(xs).to(device), torch.stack(ys).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    streamer = Streamer(train_data, batch_size)

    # Sequence length curriculum
    def get_bs(step):
        if step >= warmup:
            return block_size
        progress = step / warmup
        t = int(64 + (block_size - 64) * progress)
        return max(64, (t // 64) * 64)

    t0 = time.perf_counter()
    final_loss = 0.0

    for step in range(steps):
        cur_lr = lr * min(1.0, (step + 1) / warmup)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        bs = get_bs(step)
        opt.zero_grad()
        xb, yb = streamer.next_batch(bs)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, loss, _ = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = loss.item()

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    # Eval
    model.eval()
    with torch.no_grad():
        total_l = 0.0
        for _ in range(20):
            ix = torch.randint(len(val_data) - block_size, (batch_size,))
            xv = torch.stack([val_data[i : i + block_size] for i in ix]).to(device)
            yv = torch.stack([val_data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, l, _ = model(xv, yv)
            total_l += l.item()
    val_loss = total_l / 20

    ms_step = elapsed / steps * 1000
    tok_s = steps * batch_size * block_size / elapsed

    return {
        "name": name,
        "params": n_params,
        "val_loss": val_loss,
        "train_loss": final_loss,
        "ms_step": ms_step,
        "tok_s": tok_s,
        "elapsed": elapsed,
    }


# ─── Main ────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "A_baseline":       build_baseline,
    "B_per_head_v":     build_per_head_v,
    "C_swiglu":         build_swiglu,
    "D_topk_25":        lambda c: build_topk(c, ratio=0.25),
    "E_topk_10":        lambda c: build_topk(c, ratio=0.10),
    "F_rmsnorm":        build_rmsnorm,
    "G_diff_attn":      build_diff_attn,
    "H_learned_temp":   build_learned_temp,
    "I_gated_residual": build_gated_residual,
}


def main():
    import sys

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    raw = np.fromfile("input.txt", dtype=np.uint8)
    td = torch.from_numpy(raw[: int(0.9 * len(raw))].astype(np.int64))
    vd = torch.from_numpy(raw[int(0.9 * len(raw)) :].astype(np.int64))

    STEPS = 200
    config = bdh.BDHConfig(max_seq_len=256)

    # Parse which experiments to run (default: all)
    if len(sys.argv) > 1:
        to_run = {k: v for k, v in EXPERIMENTS.items() if k in sys.argv[1:]}
    else:
        to_run = EXPERIMENTS

    results = []
    print(f"Running {len(to_run)} experiments × {STEPS} steps on {device}\n")

    for exp_name, builder in to_run.items():
        print(f"{'='*60}")
        print(f"Experiment: {exp_name}")
        try:
            model = builder(config)
            r = train_and_eval(model, exp_name, td, vd, device, steps=STEPS)
            results.append(r)
            print(f"  val={r['val_loss']:.4f} | {r['ms_step']:.0f}ms/step | "
                  f"{r['tok_s']:.0f}tok/s | {r['params']:,} params | "
                  f"{r['elapsed']:.1f}s total")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": exp_name, "params": 0, "val_loss": float("inf"),
                "train_loss": float("inf"), "ms_step": 0, "tok_s": 0, "elapsed": 0,
            })

        # Free memory
        del model
        if device.type == "mps":
            torch.mps.synchronize()
        import gc
        gc.collect()

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<20} {'Params':>10} {'Val Loss':>10} {'Δ vs Base':>10} "
          f"{'ms/step':>8} {'tok/s':>8}")
    print(f"{'-'*80}")

    baseline_val = next((r["val_loss"] for r in results if r["name"] == "A_baseline"), None)
    for r in sorted(results, key=lambda x: x["val_loss"]):
        delta = f"{r['val_loss'] - baseline_val:+.4f}" if baseline_val else "—"
        print(f"{r['name']:<20} {r['params']:>10,} {r['val_loss']:>10.4f} "
              f"{delta:>10} {r['ms_step']:>8.0f} {r['tok_s']:>8.0f}")


if __name__ == "__main__":
    main()
