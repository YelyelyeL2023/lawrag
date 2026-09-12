# Windows wrapper for scripts/download_models.py.
# Installs huggingface_hub into the repo venv (light — no torch needed just to
# download), then fetches BGE-M3 + BGE-reranker-v2-m3 into MODEL_DIR (D:\models).
#
#   scripts\download_models.ps1
#   $env:MODEL_DIR="E:\weights"; scripts\download_models.ps1   # custom location
$ErrorActionPreference = "Stop"
$root   = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "==> ensuring huggingface_hub is installed in the venv"
& $python -m pip install -U huggingface_hub

Write-Host "==> downloading models"
& $python (Join-Path $root "scripts\download_models.py")
