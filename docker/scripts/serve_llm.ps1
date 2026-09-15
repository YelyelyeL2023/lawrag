# Start the generator as a container on Windows.
#
#   docker\scripts\serve_llm.ps1              Profile B (Qwen3.6-35B-A3B MoE) on the GPU
#   docker\scripts\serve_llm.ps1 A            Profile A (Qwen3.5-9B dense) on the GPU
#   docker\scripts\serve_llm.ps1 B -Cpu       no card: CPU image, all layers on the CPU
#
# llama-server now comes from the llama.cpp image, so there is no llama-server.exe
# to install and no PATH to get right. Docker Desktop must have GPU support
# enabled (WSL2 + a current NVIDIA driver) for the GPU path. Blackwell
# (50-series, sm_120) needs a current llama.cpp image tag or it silently falls
# back to slow paths.
param([string]$Profile = "B", [switch]$Cpu)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$files = @("-f", "docker-compose.yml")
if ($Cpu) { $env:LLM_GPU_LAYERS = "0" } else { $files += @("-f", "docker-compose.gpu.yml") }

switch ($Profile.ToUpper()) {
  "A" {
    if (-not $env:LLM_MODEL_FILE) { $env:LLM_MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf" }
    $env:LLM_N_CPU_MOE = "0"      # dense model: no experts to place
  }
  "B" {
    if (-not $env:LLM_MODEL_FILE) { $env:LLM_MODEL_FILE = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" }
    # Raise LLM_N_CPU_MOE if you hit VRAM OOM; lower it for more speed.
    if (-not $env:LLM_N_CPU_MOE) { $env:LLM_N_CPU_MOE = "28" }
  }
  default { Write-Error "usage: serve_llm.ps1 [A|B] [-Cpu]"; exit 1 }
}

Write-Host "==> profile $Profile : $env:LLM_MODEL_FILE (n-cpu-moe=$env:LLM_N_CPU_MOE)"
docker compose @files --profile llm up -d llm
$port = if ($env:LLM_HOST_PORT) { $env:LLM_HOST_PORT } else { "8080" }
Write-Host "==> llama-server on http://127.0.0.1:$port  (logs: docker compose logs -f llm)"
