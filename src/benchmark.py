"""Quick benchmark: N-step training with quality + speed measurement."""

import argparse
import math
import time

import numpy as np
import torch

import bdh


def main():
    parser = argparse.ArgumentParser(description="Quick BDH benchmark")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = bdh.BDHConfig(max_seq_len=256)
    model = bdh.BDH(cfg, use_grad_checkpoint=True).to(device)
    n = sum(p.numel() for p in model.parameters())
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, STEPS = 4, args.steps

    print(f"Model: {n:,} params | N={N} | device={device}")
    print(f"Running {STEPS}-step benchmark...")

    raw = np.fromfile("input.txt", dtype=np.uint8)
    td = torch.from_numpy(raw[: int(0.9 * len(raw))].astype(np.int64))
    vd = torch.from_numpy(raw[int(0.9 * len(raw)) :].astype(np.int64))

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

    torch.manual_seed(1337)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    streamer = Streamer(td, B)
    model.train()

    t0 = time.perf_counter()
    for step in range(STEPS):
        warmup = min(200, STEPS)
        lr = 1e-3 * (step + 1) / warmup
        for pg in opt.param_groups:
            pg["lr"] = lr
        bs = max(64, (int(64 + (256 - 64) * step / warmup) // 64) * 64) if step < 200 else 256
        opt.zero_grad()
        xb, yb = streamer.next_batch(bs)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, loss, _ = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:>4}/{STEPS} loss={loss.item():.4f}")
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    # Eval
    model.eval()
    with torch.no_grad():
        total_l = 0.0
        for _ in range(20):
            ix = torch.randint(len(vd) - 256, (B,))
            xv = torch.stack([vd[i : i + 256] for i in ix]).to(device)
            yv = torch.stack([vd[i + 1 : i + 257] for i in ix]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, l, _ = model(xv, yv)
            total_l += l.item()
    val = total_l / 20

    print(f"\nResult: val={val:.4f} | {elapsed:.0f}s | {elapsed / STEPS * 1000:.0f}ms/step | {STEPS * B * 256 / elapsed:.0f} tok/s")


if __name__ == "__main__":
    main()
