# Start llama-server for lawrag on Windows.
#   scripts\serve_llm.ps1          # Profile B (Qwen3.6-35B-A3B MoE) — default
#   scripts\serve_llm.ps1 A        # Profile A (Qwen3.5-9B dense)
#
# Needs llama-server.exe on PATH, or set $env:LLAMA_SERVER to its full path.
# Blackwell (50-series, sm_120): use a current llama.cpp build or it silently
# falls back to slow paths.
param([string]$Profile = "B")
$ErrorActionPreference = "Stop"

$ModelDir = $env:MODEL_DIR;    if (-not $ModelDir) { $ModelDir = "D:\models" }
$Port     = $env:PORT;         if (-not $Port)     { $Port     = "8080" }
$Server   = $env:LLAMA_SERVER; if (-not $Server)   { $Server   = "llama-server" }

switch ($Profile.ToUpper()) {
  "A" {
    & $Server --model (Join-Path $ModelDir "Qwen3.5-9B-Q4_K_M.gguf") `
      --host 127.0.0.1 --port $Port `
      --n-gpu-layers 999 --ctx-size 32768 `
      --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn --temp 0.0 --jinja
  }
  "B" {
    # MoE: attention + shared weights on the GPU, expert tensors in RAM.
    # Raise --n-cpu-moe if you hit VRAM OOM; lower it for more speed.
    & $Server --model (Join-Path $ModelDir "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf") `
      --host 127.0.0.1 --port $Port `
      --n-gpu-layers 999 --n-cpu-moe 28 --ctx-size 32768 `
      --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn --temp 0.0 --jinja
  }
  default { Write-Error "usage: serve_llm.ps1 [A|B]"; exit 1 }
}
