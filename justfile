# Baby Dragon Hatchling — task runner
# Usage: just <recipe>    (run `just --list` to see all recipes)

# Default: show available recipes
default:
    @just --list

# ─── Training ───────────────────────────────────────────────────────────────────

# Train the model (3000 steps, ~8 min on M1)
train:
    cd src && uv run train.py

# ─── Inference ──────────────────────────────────────────────────────────────────

# Generate text from the best checkpoint (override with: just generate 300 0.9)
generate tokens="500" temperature="0.8":
    cd src && uv run generate.py --tokens {{tokens}} --temperature {{temperature}}

# Generate with custom prompt
generate-prompt prompt tokens="500" temperature="0.8":
    cd src && uv run generate.py --prompt "{{prompt}}" --tokens {{tokens}} --temperature {{temperature}}

# Generate from a specific checkpoint file
generate-from checkpoint tokens="500":
    cd src && uv run generate.py --checkpoint "{{checkpoint}}" --tokens {{tokens}}

# ─── Benchmarking & Profiling ───────────────────────────────────────────────────

# Quick N-step benchmark (default: 200 steps, ~30s)
benchmark steps="200":
    cd src && uv run benchmark.py --steps {{steps}}

# Profile a training step (phase + per-op breakdown)
profile:
    cd src && uv run profile_ops.py

# ─── Utilities ──────────────────────────────────────────────────────────────────

# Show model architecture summary
info:
    cd src && uv run info.py

# Install dependencies
setup:
    uv sync

# Remove checkpoints and caches
clean:
    rm -rf src/checkpoints/ src/.ruff_cache/ src/__pycache__/
    @echo "Cleaned checkpoints and caches"
