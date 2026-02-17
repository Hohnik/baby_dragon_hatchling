"""Profile BDH: phase breakdown + per-op timing."""

import time

import numpy as np
import torch
import torch.nn.functional as F

import bdh


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = bdh.BDHConfig(max_seq_len=256)
    model = bdh.BDH(cfg, use_grad_checkpoint=False).to(device)
    n = sum(p.numel() for p in model.parameters())
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, T = 4, 256
    AMP = torch.float16

    print(f"Model: {n:,} params | N={N} | device={device}")

    raw = np.fromfile("input.txt", dtype=np.uint8)
    td = torch.from_numpy(raw[: int(0.9 * len(raw))].astype(np.int64))
    ix = torch.randint(len(td) - T, (B,))
    x = torch.stack([td[i : i + T] for i in ix]).to(device)
    y = torch.stack([td[i + 1 : i + 1 + T] for i in ix]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    model.train()

    # Warmup
    for _ in range(5):
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=AMP):
            _, loss, _ = model(x, y)
        loss.backward()
        opt.step()

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    sync()

    # Phase breakdown
    NR = 50
    fwd_t, bwd_t, opt_t = [], [], []
    for _ in range(NR):
        opt.zero_grad()
        sync()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=AMP):
            _, loss, _ = model(x, y)
        sync()
        t1 = time.perf_counter()
        loss.backward()
        sync()
        t2 = time.perf_counter()
        opt.step()
        sync()
        t3 = time.perf_counter()
        fwd_t.append((t1 - t0) * 1000)
        bwd_t.append((t2 - t1) * 1000)
        opt_t.append((t3 - t2) * 1000)

    fm = sum(fwd_t) / NR
    bm = sum(bwd_t) / NR
    om = sum(opt_t) / NR
    total = fm + bm + om

    print(f"\nPhase breakdown ({NR} runs, B={B}, T={T}):")
    print(f"  Forward:   {fm:>6.1f} ms  ({fm / total * 100:.0f}%)")
    print(f"  Backward:  {bm:>6.1f} ms  ({bm / total * 100:.0f}%)")
    print(f"  Optimizer: {om:>6.1f} ms  ({om / total * 100:.0f}%)")
    print(f"  Total:     {total:>6.1f} ms")
    print(f"  Throughput: {B * T / (total / 1000):.0f} tok/s")

    # Op-level breakdown
    def timed(fn, n=50):
        for _ in range(5):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return (time.perf_counter() - t0) / n * 1000

    layer = model.layers[0]
    attn = model.attn
    D, nh = cfg.n_embd, cfg.n_head
    x_in = model.ln(model.embed(x).unsqueeze(1))

    with torch.autocast(device_type=device.type, dtype=AMP):
        enc_flat = layer.encoder.permute(1, 0, 2).reshape(D, nh * N)
        x_flat = x_in.squeeze(1).reshape(B * T, D)
        x_lat = (x_flat @ enc_flat).view(B, T, nh, N).permute(0, 2, 1, 3)
        x_sp = F.relu(x_lat)
        QR = attn.rope(x_sp, T)
        scores = (QR @ QR.mT * attn._scale).tril(-1)
        V = x_in
        yKV = scores @ V
        yKV_ln = layer.ln(yKV)
        y_sp = F.relu(torch.einsum("bhtd,hdn->bhtn", yKV_ln, layer.encoder_v))
        xy_f = (x_sp * y_sp).transpose(1, 2).reshape(B, 1, T, nh * N)

        ops = [
            ("encode", timed(lambda: (x_flat @ enc_flat).view(B, T, nh, N).permute(0, 2, 1, 3))),
            ("relu", timed(lambda: F.relu(x_lat))),
            ("rope", timed(lambda: attn.rope(x_sp, T))),
            ("scores", timed(lambda: (QR @ QR.mT * attn._scale).tril(-1))),
            ("attn@V", timed(lambda: scores @ V)),
            ("encode_v", timed(lambda: torch.einsum("bhtd,hdn->bhtn", yKV_ln, layer.encoder_v))),
            ("hebbian", timed(lambda: x_sp * y_sp)),
            ("decode", timed(lambda: xy_f @ layer.decoder)),
        ]

    tot_ops = sum(ms for _, ms in ops)
    ops.sort(key=lambda x: -x[1])
    print(f"\nOp breakdown (1 layer):")
    for name, ms in ops:
        bar = "█" * int(ms / tot_ops * 30)
        print(f"  {ms:>5.2f} ms ({ms / tot_ops * 100:>4.1f}%) {name:12s} {bar}")
    print(f"  ─────")
    print(f"  {tot_ops:>5.2f} ms total (×{cfg.n_layer}={tot_ops * cfg.n_layer:.1f}ms)")


if __name__ == "__main__":
    main()
