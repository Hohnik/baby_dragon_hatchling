# Copyright Pathway Technology, Inc.
# Optimized training script for Apple M1 (8GB).
#
# Changes from original:
#   1. Gradient checkpointing: recompute activations to save ~60% peak memory
#   2. Gradient accumulation: simulate larger effective batch from small micro-batches
#   3. Cosine LR schedule with warmup (standard from GPT-2/nanoGPT research)
#   4. Gradient clipping: stabilize training with deep 6-layer network
#   5. Preloaded data: avoid per-batch memmap/numpy overhead
#   6. Periodic validation: track overfitting
#   7. No torch.compile: slower on MPS (no inductor Metal backend)
#   8. No fp16 autocast: BDH attention scores (N=8192 inner dim) cause fp16 gradient
#      overflow after 1 step. bfloat16 works but gives no speedup on M1 (no hw support).
#   9. Proper device detection: auto-select MPS/CUDA/CPU

import math
import os
import time

import bdh
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
print(f"Device: {device}")

# ─── Configuration ──────────────────────────────────────────────────────────────
# Tuned for 8GB M1: B=4 T=256 fits in memory with gradient checkpointing.
# Original (B=32 T=512) requires ~74GB for activations — impossible on 8GB.
#
# To increase effective batch size without more memory, raise GRAD_ACCUM_STEPS.
# Each doubling of GRAD_ACCUM_STEPS doubles step time but also effective batch.
BLOCK_SIZE = 256  # sequence length (original: 512)
MICRO_BATCH = 4  # micro-batch per accumulation step (original: 32)
GRAD_ACCUM_STEPS = 1  # gradient accumulation steps
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM_STEPS  # total samples per optimizer step

MAX_ITERS = 3000
LEARNING_RATE = 1e-3
MIN_LR = 1e-4  # cosine schedule floor
WARMUP_ITERS = 200  # linear warmup iterations
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0  # max gradient L2 norm
LOG_FREQ = 50
EVAL_FREQ = 500  # run validation every N steps
EVAL_ITERS = 10  # batches to average for stable val loss estimate

BDH_CONFIG = bdh.BDHConfig(max_seq_len=BLOCK_SIZE)

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


# ─── LR schedule: linear warmup → cosine decay ─────────────────────────────────
def get_lr(step):
    """Cosine LR schedule with linear warmup.

    Standard from GPT-2/3 training (Brown et al. 2020) and nanoGPT.
    - Warmup: prevents early instability when gradients are noisy at init.
    - Cosine decay: smooth annealing gives better final loss than step decay.
    """
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    progress = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * progress))


# ─── Data loading ───────────────────────────────────────────────────────────────
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


# Preload full dataset into memory once. The original code re-opens a numpy memmap
# every single batch call, which adds ~1ms overhead per call and prevents the OS
# from caching the file efficiently.
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


# ─── Validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model):
    """Average loss over multiple batches for a stable estimate.

    Single-batch loss is noisy. Averaging over EVAL_ITERS batches gives
    reliable train/val comparison to detect overfitting.
    """
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / EVAL_ITERS
    model.train()
    return losses


# ─── Training ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetch_data()

    # use_grad_checkpoint=True: recompute layer activations during backward instead
    # of storing them. Saves ~60% peak memory (those (B,nh,T,8192) intermediates)
    # at the cost of one extra forward pass per layer during backward.
    # On M1 this is actually FASTER because less memory pressure = fewer GPU stalls.
    model = bdh.BDH(BDH_CONFIG, use_grad_checkpoint=True).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(
        f"Config: {MAX_ITERS} iters, micro_batch={MICRO_BATCH}, "
        f"accum={GRAD_ACCUM_STEPS}, effective_batch={EFFECTIVE_BATCH}, "
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
        # ── LR schedule ──
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ──
        # Process GRAD_ACCUM_STEPS micro-batches, accumulate gradients, then step.
        # Mathematically equivalent to a single large batch but fits in memory.
        optimizer.zero_grad()
        step_loss = 0.0
        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = get_batch("train")
            _, loss = model(x, y)
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
            step_loss += loss.item()
        step_loss /= GRAD_ACCUM_STEPS

        # ── Gradient clipping ──
        # Prevents rare large gradients from destabilizing weights.
        # Standard practice for transformers (Megatron-LM, PaLM, etc.)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        loss_acc += step_loss
        loss_steps += 1

        # ── Logging ──
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

        # ── Periodic validation ──
        if step > 0 and step % EVAL_FREQ == 0:
            losses = estimate_loss(model)
            print(
                f"  ── eval: train={losses['train']:.4f} val={losses['val']:.4f}"
            )

    # ── Final evaluation ──
    losses = estimate_loss(model)
    print(f"\nFinal: train={losses['train']:.4f} val={losses['val']:.4f}")

    # ── Generate sample ──
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
