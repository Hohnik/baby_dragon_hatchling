"""Benchmark: BDH vs BDH-HRM training on tiny Shakespeare.
Compares loss curves and throughput."""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import bdh
import bdh_hrm

device = torch.device("mps")

# Data
raw = np.fromfile(os.path.join(os.path.dirname(__file__), "input.txt"), dtype=np.uint8)
train_data = torch.from_numpy(raw[: int(0.9 * len(raw))].astype(np.int64))
val_data = torch.from_numpy(raw[int(0.9 * len(raw)) :].astype(np.int64))

T = 256
B = 4


def get_batch(data):
    ix = torch.randint(len(data) - T, (B,))
    x = torch.stack([data[i : i + T] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + T] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, data, n=10):
    model.eval()
    total = 0.0
    for _ in range(n):
        x, y = get_batch(data)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n


def train_model(model, name, n_steps=200, lr=1e-3):
    print(f"\n{'='*60}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} params, {n_steps} steps")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()

    # Warmup
    x, y = get_batch(train_data)
    with torch.autocast(device_type="mps", dtype=torch.float16):
        _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.mps.synchronize()

    losses = []
    t_start = time.perf_counter()
    for step in range(n_steps):
        optimizer.zero_grad()
        x, y = get_batch(train_data)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            torch.mps.synchronize()
            elapsed = time.perf_counter() - t_start
            tps = (step + 1) * B * T / elapsed if step > 0 else 0
            print(f"  step {step:>4} | loss {loss.item():.4f} | {tps:.0f} tok/s")

    torch.mps.synchronize()
    total_time = time.perf_counter() - t_start
    ms_per_step = total_time / n_steps * 1000
    tps = n_steps * B * T / total_time

    val = eval_loss(model, val_data)

    print(f"\n  Final: train_loss={losses[-1]:.4f} val_loss={val:.4f}")
    print(f"  Speed: {ms_per_step:.0f} ms/step | {tps:.0f} tok/s")
    print(f"  Total time: {total_time:.1f}s")

    return {
        "name": name,
        "params": n_params,
        "final_train_loss": losses[-1],
        "val_loss": val,
        "ms_per_step": ms_per_step,
        "tok_per_sec": tps,
        "losses": losses,
    }


if __name__ == "__main__":
    N_STEPS = 200

    # ─── BDH (original, optimized) ───
    torch.manual_seed(1337)
    bdh_config = bdh.BDHConfig(max_seq_len=T)
    bdh_model = bdh.BDH(bdh_config, use_grad_checkpoint=True).to(device)
    r_bdh = train_model(bdh_model, "BDH (6 shared layers)", N_STEPS)
    del bdh_model

    torch.mps.empty_cache()

    # ─── BDH-HRM (3H × 2L = 6 iterations, separate H/L params) ───
    torch.manual_seed(1337)
    hrm_config = bdh_hrm.BDHHRMConfig(max_seq_len=T, h_cycles=3, l_cycles=2)
    hrm_model = bdh_hrm.BDHHRM(hrm_config).to(device)
    r_hrm = train_model(hrm_model, "BDH-HRM (3H×2L, no-grad trick)", N_STEPS)
    del hrm_model

    # ─── Summary ───
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in [r_bdh, r_hrm]:
        print(
            f"  {r['name']:40s} | "
            f"{r['params']:>10,} params | "
            f"val={r['val_loss']:.4f} | "
            f"{r['ms_per_step']:>5.0f} ms/step | "
            f"{r['tok_per_sec']:>5.0f} tok/s"
        )

    # Quality per compute
    for r in [r_bdh, r_hrm]:
        loss_per_sec = (5.6 - r["val_loss"]) / (r["ms_per_step"] * N_STEPS / 1000)
        print(
            f"  {r['name']:40s} | "
            f"loss improvement/sec: {loss_per_sec:.4f}"
        )
