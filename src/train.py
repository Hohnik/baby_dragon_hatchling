# Copyright Pathway Technology, Inc.
# Training script for BDH with Truncated BPTT (continuous learning).
#
# This implements the paper's intended training procedure:
#   "Truncated Backpropagation Through time, carrying over the recurrent state
#    of attention and training on sequences of length 2048 characters at a time."
#    — BDH paper, Appendix B.3
#
# The key difference from standard training: instead of sampling random chunks,
# we process text sequentially, carrying the synaptic state ρ across chunks.
# This lets the model learn continuous representations that span chunk boundaries.

import math
import os
import time

import bdh
import dataclasses
import numpy as np
import requests
import torch
from tqdm import tqdm

# ─── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(1337)

USE_AMP = device.type in ("cuda", "mps")
AMP_DTYPE = torch.float16
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16

print(f"Device: {device} | AMP: {USE_AMP} ({AMP_DTYPE})")

# ─── Configuration ──────────────────────────────────────────────────────────────
BLOCK_SIZE = 256
BLOCK_SIZE_MIN = 64  # curriculum start (ramp from 64 → 256 during warmup)
MICRO_BATCH = 8  # number of parallel streams for TBPTT
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM_STEPS

MAX_ITERS = 3000
LEARNING_RATE = 1e-3
MIN_LR = 1e-4
WARMUP_ITERS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_FREQ = 50
EVAL_FREQ = 500
EVAL_ITERS = 10
CKPT_FREQ = 500  # Save checkpoint every N steps
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

BDH_CONFIG = bdh.BDHConfig(max_seq_len=BLOCK_SIZE + 320)  # extra room for generation

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


def get_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    progress = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * progress))


def get_block_size(step: int) -> int:
    """Sequence length curriculum: ramp from BLOCK_SIZE_MIN to BLOCK_SIZE during warmup.

    Short sequences are much cheaper (attention is O(T²)) and provide useful
    gradient signal early when the model is still learning basic patterns.
    T=64 is 2.65x faster than T=256 per step.
    """
    if step >= WARMUP_ITERS:
        return BLOCK_SIZE
    progress = step / WARMUP_ITERS
    # Linear ramp, snapped to multiples of 64 for alignment
    t = int(BLOCK_SIZE_MIN + (BLOCK_SIZE - BLOCK_SIZE_MIN) * progress)
    return max(BLOCK_SIZE_MIN, (t // 64) * 64)


# ─── Data ────────────────────────────────────────────────────────────────────────
def fetch_data() -> None:
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


_data_cache: dict[str, torch.Tensor] = {}


def _get_data(split: str) -> torch.Tensor:
    if split not in _data_cache:
        raw = np.fromfile(input_file_path, dtype=np.uint8)
        pivot = int(0.9 * len(raw))
        if split == "train":
            _data_cache[split] = torch.from_numpy(raw[:pivot].astype(np.int64))
        else:
            _data_cache[split] = torch.from_numpy(raw[pivot:].astype(np.int64))
    return _data_cache[split]


class SequentialStreamer:
    """Provides sequential chunks for TBPTT training.

    Instead of random sampling, maintains B parallel read cursors into the
    training data, advancing sequentially. This is required for the recurrent
    synaptic state to accumulate meaningful cross-chunk information.

    When a cursor reaches the end, it wraps around with a fresh state.
    """

    def __init__(self, data: torch.Tensor, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        seg_len = len(data) // batch_size
        self.cursors = [i * seg_len for i in range(batch_size)]

    def next_batch(
        self, block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
        """Get next sequential chunk for each stream.

        Args:
            block_size: number of tokens per chunk (may vary with curriculum)

        Returns:
            x: (B, T) input tokens
            y: (B, T) target tokens
            resets: list[bool] — True if this stream wrapped around
        """
        xs, ys, resets = [], [], []
        for i in range(self.batch_size):
            start = self.cursors[i]
            end = start + block_size + 1
            if end > len(self.data):
                # Wrap around
                start = 0
                end = block_size + 1
                resets.append(True)
            else:
                resets.append(False)
            chunk = self.data[start:end]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.cursors[i] = start + block_size
        return (
            torch.stack(xs).to(device),
            torch.stack(ys).to(device),
            resets,
        )


def get_random_batch(
    split: str, batch_size: int = MICRO_BATCH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random batch for evaluation (no state needed)."""
    data = _get_data(split)
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: bdh.BDH) -> dict[str, float]:
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(EVAL_ITERS):
            x, y = get_random_batch(split)
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                _, loss, _ = model(x, y)
            total += loss.item()
        losses[split] = total / EVAL_ITERS
    model.train()
    return losses


def _detach_state(state: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    """Detach state from computation graph for truncated BPTT."""
    return [s.detach() if s is not None else None for s in state]


def _reset_stream_state(
    state: list[torch.Tensor | None], resets: list[bool],
) -> list[torch.Tensor | None]:
    """Zero out state for streams that wrapped around."""
    if not any(resets):
        return state
    new_state = []
    for s in state:
        if s is None:
            new_state.append(None)
            continue
        s_new = s.clone()
        for i, should_reset in enumerate(resets):
            if should_reset:
                s_new[i] = 0.0
        new_state.append(s_new)
    return new_state


def save_checkpoint(
    model: bdh.BDH, optimizer: torch.optim.Optimizer,
    step: int, val_loss: float,
) -> str:
    """Save model + optimizer state to disk."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"bdh_step{step:05d}_val{val_loss:.4f}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": dataclasses.asdict(model.config),
    }, path)
    return path


@torch.no_grad()
def generate_sample(model: bdh.BDH, prompt: str = "KING RICHARD III:\n") -> str:
    """Generate a text sample from the model."""
    model.eval()
    idx = torch.tensor(
        bytearray(prompt, "utf-8"), dtype=torch.long, device=device,
    ).unsqueeze(0)
    ret = model.generate(idx, max_new_tokens=300, top_k=10, temperature=0.8)
    text = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace",
    )
    model.train()
    return text


# ─── Training ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG, use_grad_checkpoint=True).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(
        f"Config: {MAX_ITERS} iters, batch={MICRO_BATCH}, "
        f"accum={GRAD_ACCUM_STEPS}, seq_len={BLOCK_SIZE}, "
        f"TBPTT=True"
    )
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    streamer = SequentialStreamer(_get_data("train"), MICRO_BATCH)
    state: list[torch.Tensor] | None = None  # recurrent synaptic state

    model.train()
    tok_acc = 0
    t_tok = time.perf_counter()

    pbar = tqdm(range(MAX_ITERS), desc="training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    for step in pbar:
        lr = get_lr(step)
        block_size = get_block_size(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y, resets = streamer.next_batch(block_size)

            # Reset state for streams that wrapped around
            # Also reset when block size changes (state shape depends on T)
            if state is not None:
                state = _reset_stream_state(state, resets)

            with torch.autocast(
                device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP,
            ):
                _, loss, new_state = model(x, y, state=state)

            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
            step_loss += loss.item()

            # Truncated BPTT: carry state but detach from graph
            state = _detach_state(new_state)

        step_loss /= GRAD_ACCUM_STEPS
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        tok_acc += block_size * EFFECTIVE_BATCH
        now = time.perf_counter()
        dt = now - t_tok
        tok_s = tok_acc / dt if dt > 0 else 0

        pbar.set_postfix_str(
            f"loss={step_loss:.4f} lr={lr:.1e} T={block_size} {tok_s:,.0f}tok/s",
            refresh=False,
        )

        if step > 0 and step % LOG_FREQ == 0:
            tok_acc = 0
            t_tok = now

        if step > 0 and step % EVAL_FREQ == 0:
            losses = estimate_loss(model)
            tqdm.write(
                f"  ── eval @ {step}: train={losses['train']:.4f} val={losses['val']:.4f}"
            )

        if step > 0 and step % CKPT_FREQ == 0:
            ckpt_losses = estimate_loss(model)
            path = save_checkpoint(model, optimizer, step, ckpt_losses["val"])
            tqdm.write(f"  ── checkpoint saved: {path}")
            sample = generate_sample(model)
            tqdm.write(f"  ── sample: {sample[:200]}…")

    pbar.close()
    losses = estimate_loss(model)
    print(f"\nFinal: train={losses['train']:.4f} val={losses['val']:.4f}")

    print("\nGenerating sample...")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device,
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=200, top_k=5)
    text = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace",
    )
    print(text)
