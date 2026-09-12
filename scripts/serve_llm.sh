#!/usr/bin/env bash
# Start the generator as a container, for the two supported hardware profiles.
#
# Profile A — Qwen3.5-9B fully on an 8 GB card, ~50 tok/s.
# Profile B — Qwen3.6-35B-A3B with MoE experts in system RAM (default).
#             Needs ~20 GB free RAM; 64 GB total is comfortable, 32 GB is
#             tight once Postgres is also running.
#
#   scripts/serve_llm.sh            Profile B on the GPU
#   scripts/serve_llm.sh A          Profile A on the GPU
#   scripts/serve_llm.sh B --cpu    no card: CPU image, all layers on the CPU
#
# The weights themselves are not in the image: MODEL_DIR is mounted read-only
# at /models, so switching models is a variable, not a rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${1:-B}"
COMPOSE=(docker compose -f docker-compose.yml)
if [[ "${2:-}" == "--cpu" ]]; then
  export LLM_GPU_LAYERS=0
else
  COMPOSE+=(-f docker-compose.gpu.yml)
fi

case "${PROFILE^^}" in
  A)
    export LLM_MODEL_FILE="${LLM_MODEL_FILE:-Qwen3.5-9B-Q4_K_M.gguf}"
    # Dense model: no experts to place, so none go to the CPU.
    export LLM_N_CPU_MOE=0
    ;;
  B)
    export LLM_MODEL_FILE="${LLM_MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
    # Attention and shared weights stay on the GPU; the expert tensors live in
    # RAM. Raise this if you hit VRAM OOM, lower it for more speed. Memory
    # bandwidth, not core count, is the bottleneck: check your DDR5 is actually
    # running at its rated speed.
    export LLM_N_CPU_MOE="${LLM_N_CPU_MOE:-28}"
    ;;
  *)
    echo "usage: $0 [A|B] [--cpu]" >&2; exit 1 ;;
esac

echo "==> profile $PROFILE: $LLM_MODEL_FILE (n-cpu-moe=$LLM_N_CPU_MOE)"
"${COMPOSE[@]}" --profile llm up -d llm
echo "==> llama-server on http://127.0.0.1:${LLM_HOST_PORT:-8080}  (logs: docker compose logs -f llm)"
