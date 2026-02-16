# Copyright Pathway Technology, Inc.
# Training script for BDH-HRM hybrid model.
# Best config found: 1H×2L — fastest + best loss combination.

import math
import os
import time

import bdh_hrm
import numpy as np
import requests
import torch

# ─── Device setup ───────────────────────────────────────────────────────────────
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
MICRO_BATCH = 4
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

# Best config from sweep: 1H×2L gives best val loss + speed tradeoff
MODEL_CONFIG = bdh_hrm.BDHHRMConfig(
    max_seq_len=BLOCK_SIZE,
    h_cycles=1,
    l_cycles=2,
)

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


def get_lr(step):
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    progress = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * progress))


def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


_data_cache = {}


def _get_data(split):
    if split not in _data_cache:
        raw = np.fromfile(input_file_path, dtype=np.uint8)
        pivot = int(0.9 * len(raw))
        if split == "train":
            _data_cache[split] = torch.from_numpy(raw[:pivot].astype(np.int64))
        else:
            _data_cache[split] = torch.from_numpy(raw[pivot:].astype(np.int64))
    return _data_cache[split]


def get_batch(split, batch_size=MICRO_BATCH):
    data = _get_data(split)
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / EVAL_ITERS
    model.train()
    return losses


if __name__ == "__main__":
    fetch_data()

    model = bdh_hrm.BDHHRM(MODEL_CONFIG).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(
        f"Config: {MAX_ITERS} iters, micro_batch={MICRO_BATCH}, "
        f"h_cycles={MODEL_CONFIG.h_cycles}, l_cycles={MODEL_CONFIG.l_cycles}, "
        f"seq_len={BLOCK_SIZE}"
    )
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    model.train()
    loss_acc = 0.0
    loss_steps = 0
    t_start = time.perf_counter()
    t_log = t_start

    for step in range(MAX_ITERS):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0
        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = get_batch("train")
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                _, loss = model(x, y)
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
            step_loss += loss.item()
        step_loss /= GRAD_ACCUM_STEPS

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        loss_acc += step_loss
        loss_steps += 1

        if step % LOG_FREQ == 0:
            if device.type == "mps":
                torch.mps.synchronize()
            now = time.perf_counter()
            dt = now - t_log
            t_log = now
            tokens_per_sec = (
                LOG_FREQ * EFFECTIVE_BATCH * BLOCK_SIZE / dt if step > 0 else 0
            )
            avg_loss = loss_acc / loss_steps
            elapsed = now - t_start
            eta = (MAX_ITERS - step) / max(step, 1) * elapsed
            print(
                f"step {step:>5}/{MAX_ITERS} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{tokens_per_sec:,.0f} tok/s | "
                f"elapsed {elapsed/60:.1f}m | "
                f"eta {eta/60:.1f}m"
            )
            loss_acc = 0.0
            loss_steps = 0

        if step > 0 and step % EVAL_FREQ == 0:
            losses = estimate_loss(model)
            print(
                f"  ── eval: train={losses['train']:.4f} val={losses['val']:.4f}"
            )

    losses = estimate_loss(model)
    print(f"\nFinal: train={losses['train']:.4f} val={losses['val']:.4f}")

    print("\nGenerating sample...")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
