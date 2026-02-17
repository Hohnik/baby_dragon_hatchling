"""Show BDH model architecture summary."""

import bdh


def main():
    cfg = bdh.BDHConfig()
    model = bdh.BDH(cfg)
    n = sum(p.numel() for p in model.parameters())
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head

    print("Baby Dragon Hatchling")
    print(f"  Parameters:  {n:,}")
    print(f"  Layers:      {cfg.n_layer}")
    print(f"  Heads:       {cfg.n_head}")
    print(f"  Embed dim:   {cfg.n_embd}")
    print(f"  Sparse dim:  {N} (per head)")
    print(f"  Vocab:       {cfg.vocab_size} (byte-level)")
    print(f"  Max seq len: {cfg.max_seq_len}")
    print(f"  Forget mode: {cfg.forget_mode}")
    print()
    print("Parameters:")
    for name, p in model.named_parameters():
        print(f"  {name:40s} {str(list(p.shape)):>25s}  ({p.numel():>10,})")


if __name__ == "__main__":
    main()
