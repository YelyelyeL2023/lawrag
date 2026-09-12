# Windows wrapper for scripts/download_models.py. CONNECTED side only.
#
# The download runs inside the lawrag `models` image — the same image that
# later loads the weights. Nothing is installed into a host Python, and there
# is no virtualenv: the old .venv\Scripts\python.exe path is gone.
#
#   scripts\download_models.ps1
#   $env:MODEL_DIR="E:\weights"; scripts\download_models.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $env:MODEL_DIR) { $env:MODEL_DIR = "./models" }
New-Item -ItemType Directory -Force -Path $env:MODEL_DIR | Out-Null

Write-Host "==> downloading into $env:MODEL_DIR"
docker compose --profile download run --rm download-models @args

Write-Host ""
Write-Host "Carry the MODEL_DIR tree across the air gap together with the images:"
Write-Host "    docker save lawrag:models pgvector/pgvector:pg16 ghcr.io/ggml-org/llama.cpp:server-cuda -o lawrag-images.tar"
