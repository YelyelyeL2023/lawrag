# Start `lawrag final-experiment` in the background, detached from this
# terminal, with its log in data/train/final-<timestamp>.log.
#
# The run takes ~6-7 h on an 8 GB laptop GPU. Every finished stage is saved
# under reports/final/stages, so after an interruption (crash, reboot) running
# this script again resumes where the run stopped. The process itself asks
# Windows not to idle-sleep until it ends; a closed lid still sleeps.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_experiment.ps1
#   Get-Content data\train\final-*.log -Tail 20 -Wait     # follow progress
#
# Extra arguments are passed through, e.g. -- --no-rerank.

param(
    [string]$ModelDir = $(if ($env:LAWRAG_MODEL_DIR) { $env:LAWRAG_MODEL_DIR }
                         elseif ($env:MODEL_DIR) { $env:MODEL_DIR } else { "models" }),
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Rest
)

$repo = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $repo "data\train\final-$stamp.log"
$out = Join-Path $repo "data\train\final-$stamp.out"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "final-experiment" }
if ($running) {
    Write-Host "already running: PID $($running.ProcessId -join ', ')"
    exit 1
}

$env:PYTHONUNBUFFERED = "1"
$pyArgs = @("-m", "lawrag.cli", "final-experiment", "--model-dir", $ModelDir)
if ($Rest) { $pyArgs += $Rest }
$p = Start-Process -FilePath "python" -ArgumentList $pyArgs -WorkingDirectory $repo `
    -RedirectStandardError $log -RedirectStandardOutput $out -WindowStyle Hidden -PassThru
Write-Host "started PID $($p.Id)"
Write-Host "log: $log"
