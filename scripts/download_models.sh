#!/usr/bin/env bash
# Fetch BGE-M3 + BGE-reranker-v2-m3 into MODEL_DIR. CONNECTED side only.
#
# Runs scripts/download_models.py inside the lawrag `models` image — the same
# image that will later load the weights, so what downloads here is what loads
# there. Nothing is installed on the host.
#
#   scripts/download_models.sh
#   MODEL_DIR=/srv/weights scripts/download_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR"
echo "==> downloading into $MODEL_DIR"
MODEL_DIR="$MODEL_DIR" docker compose --profile download run --rm download-models "$@"

cat <<'MSG'

Carry the MODEL_DIR tree across the air gap, together with the images:

    docker save lawrag:models pgvector/pgvector:pg16 ghcr.io/ggml-org/llama.cpp:server-cuda \
      | gzip > lawrag-images.tar.gz

On the target: docker load < lawrag-images.tar.gz, put the tree at MODEL_DIR,
`docker compose up -d`. Nothing downloads on the query path.
MSG
