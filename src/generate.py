"""Generate text from a trained BDH checkpoint."""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import bdh


def find_best_checkpoint(ckpt_dir: str) -> str | None:
    """Find the checkpoint with the lowest val loss."""
    pattern = os.path.join(ckpt_dir, "bdh_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    # Parse val loss from filename: bdh_*_val{loss}.pt
    def parse_val(f):
        try:
            return float(f.rsplit("val", 1)[1].replace(".pt", ""))
        except (IndexError, ValueError):
            return float("inf")
    return min(files, key=parse_val)


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained BDH model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file (default: best in src/checkpoints/)")
    parser.add_argument("--prompt", type=str, default="KING RICHARD III:\n",
                        help="Text prompt to start generation")
    parser.add_argument("--tokens", type=int, default=500,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k sampling (0 = disabled)")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        ckpt_path = find_best_checkpoint(ckpt_dir)
        if ckpt_path is None:
            print("No checkpoints found. Run `just train` first.")
            sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load
    print(f"Loading {os.path.basename(ckpt_path)} → {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = bdh.BDHConfig(**ckpt["config"])
    model = bdh.BDH(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | val_loss={ckpt.get('val_loss', '?'):.4f}")
    print(f"Config: N={config.mlp_internal_dim_multiplier * config.n_embd // config.n_head}, "
          f"D={config.n_embd}, heads={config.n_head}, layers={config.n_layer}")
    print("─" * 60)

    # Generate
    idx = torch.tensor(
        bytearray(args.prompt, "utf-8"), dtype=torch.long, device=device,
    ).unsqueeze(0)

    top_k = args.top_k if args.top_k > 0 else None
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.tokens,
                             temperature=args.temperature, top_k=top_k)

    text = bytes(out.to(torch.uint8).cpu().squeeze(0)).decode(errors="backslashreplace")
    print(text)


if __name__ == "__main__":
    main()
