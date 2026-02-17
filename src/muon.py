"""Muon optimizer: MomentUm Orthogonalized by Newton-schulz.

Adapted from modded-nanogpt (KellerJordan/modded-nanogpt) for BDH's
projection matrices. Applies Newton-Schulz orthogonalization to the
momentum, which better conditions the optimization landscape for
matrix-valued parameters.

Usage:
    # Split params: Muon for projections, AdamW for everything else
    muon_params, adam_params = split_params_for_muon(model)
    optimizer = CombinedOptimizer([
        Muon(muon_params, lr=0.02, momentum=0.95),
        torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.1),
    ])
"""

import torch
from torch.optim import Optimizer


def _newton_schulz(M: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    """Orthogonalize M via Newton-Schulz iteration.

    Converges to the orthogonal polar factor of M in ~5 iterations.
    Coefficients from Björck & Bowie (1971), optimized by Keller Jordan.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M / (M.norm() + 1e-7)
    for _ in range(n_iter):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    return X


class Muon(Optimizer):
    """Muon optimizer for matrix-valued parameters.

    For each parameter:
    1. Compute momentum (exponential moving average of gradients)
    2. Reshape to 2D (if needed)
    3. Orthogonalize via Newton-Schulz iteration
    4. Scale by learning rate

    This gives better convergence than Adam for projection matrices
    because it normalizes the update direction along all singular
    directions equally.

    Args:
        params: iterable of parameters (should be ≥2D tensors)
        lr: learning rate (typical: 0.02, higher than Adam)
        momentum: momentum coefficient (typical: 0.95)
        nesterov: use Nesterov momentum
        ns_iter: number of Newton-Schulz iterations
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_iter: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_iter=ns_iter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_iter = group["ns_iter"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                # Use Nesterov-style lookahead
                update = g.add(buf, alpha=mu) if nesterov else buf

                # Reshape to 2D for Newton-Schulz
                orig_shape = update.shape
                if update.ndim > 2:
                    update = update.reshape(update.shape[0], -1)
                elif update.ndim < 2:
                    # Skip 1D params (bias, etc.)
                    p.add_(update, alpha=-lr)
                    continue

                # Orthogonalize: this is the key Muon step.
                # Makes the update uniformly scaled across all singular directions.
                if update.shape[0] <= update.shape[1]:
                    update = _newton_schulz(update, ns_iter)
                else:
                    update = _newton_schulz(update.T, ns_iter).T

                # Reshape back and apply
                update = update.reshape(orig_shape)
                p.add_(update, alpha=-lr)


def split_params_for_muon(
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Split model parameters into Muon-eligible and AdamW-eligible groups.

    Muon works best on ≥2D projection matrices. Embeddings, biases, and
    LayerNorm params should use AdamW (per modded-nanogpt recipe).

    Returns:
        muon_params: list of parameters for Muon
        adam_params: list of param dicts for AdamW (with weight_decay=0 for 1D)
    """
    muon_params = []
    adam_decay = []
    adam_no_decay = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Embeddings, lm_head, and any 1D params → AdamW
        if "embed" in name or "lm_head" in name or p.ndim <= 1:
            if p.ndim >= 2:
                adam_decay.append(p)
            else:
                adam_no_decay.append(p)
        else:
            # Encoder, encoder_v, decoder → Muon
            muon_params.append(p)

    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay, "weight_decay": 0.1})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})

    return muon_params, adam_groups


class CombinedOptimizer:
    """Wraps multiple optimizers into one with a unified interface."""

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    @property
    def param_groups(self) -> list[dict]:
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups
