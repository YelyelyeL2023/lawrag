#!/usr/bin/env bash
# Start llama-server for the two supported hardware profiles.
#
# Profile A (default) — Qwen3.5-9B fully on an 8 GB card, ~50 tok/s.
# Profile B           — Qwen3.6-35B-A3B with MoE experts in system RAM.
#                       Needs ~20 GB free RAM; 64 GB total is comfortable,
#                       32 GB is tight once Postgres is also running.
set -euo pipefail

PROFILE="${1:-A}"
MODEL_DIR="${MODEL_DIR:-/opt/models}"
PORT="${PORT:-8080}"

case "$PROFILE" in
  A)
    exec llama-server \
      --model "$MODEL_DIR/Qwen3.5-9B-Q4_K_M.gguf" \
      --host 127.0.0.1 --port "$PORT" \
      --n-gpu-layers 999 \
      --ctx-size 32768 \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --flash-attn \
      --temp 0.0 \
      --jinja
    ;;
  B)
    # --n-cpu-moe keeps attention and shared weights on the GPU while the
    # expert tensors live in RAM. Raise the number if you hit OOM, lower it
    # for more speed. Memory bandwidth, not core count, is the bottleneck:
    # check your DDR5 is actually running at its rated speed.
    exec llama-server \
      --model "$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
      --host 127.0.0.1 --port "$PORT" \
      --n-gpu-layers 999 \
      --n-cpu-moe 28 \
      --ctx-size 32768 \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --flash-attn \
      --temp 0.0 \
      --jinja
    ;;
  *)
    echo "usage: $0 [A|B]" >&2; exit 1 ;;
esac
