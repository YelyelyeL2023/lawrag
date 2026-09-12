# Windows equivalent of scripts/bootstrap.sh
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "==> starting Postgres"
docker compose up -d db
do { Start-Sleep -Seconds 1 } until (docker compose exec -T db pg_isready -U lawrag -d lawrag 2>$null)

Write-Host "==> installing lawrag"
pip install -e ".[dev]"

Write-Host "==> ingesting the SYNTHETIC sample corpus"
lawrag ingest data/sample --reset --no-verify

lawrag stats
lawrag eval

Write-Host ""
Write-Host "Ready. Try:"
Write-Host '    lawrag search "В какой срок нужно встать на регистрационный учет?"'
Write-Host '    lawrag search "Какой штраф за просрочку?" --as-of 2025-03-15'
