# Windows equivalent of scripts/bootstrap.sh.
#
# Docker Desktop is the only prerequisite — nothing is installed into a host
# Python, and there is no virtualenv to activate.
#
#   scripts\bootstrap.ps1          CPU
#   scripts\bootstrap.ps1 -Gpu     add the GPU overlay (embedder on the card)
param([switch]$Gpu)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$files = @("-f", "docker-compose.yml")
if ($Gpu) { $files += @("-f", "docker-compose.gpu.yml") }

Write-Host "==> building the lawrag image"
docker compose @files build app

Write-Host "==> starting Postgres"
docker compose @files up -d --wait db

Write-Host "==> ingesting the SYNTHETIC sample corpus"
docker compose @files run --rm cli ingest data/sample --reset --no-verify

Write-Host "==> corpus"
docker compose @files run --rm cli stats

Write-Host "==> retrieval eval"
docker compose @files run --rm cli eval

Write-Host "==> starting the API and console"
docker compose @files up -d app

Write-Host ""
Write-Host "Ready.  Console: http://127.0.0.1:8000/   API docs: http://127.0.0.1:8000/docs"
Write-Host ""
Write-Host '    docker compose run --rm cli search "В какой срок нужно встать на регистрационный учет?"'
Write-Host '    docker compose run --rm cli search "Какой штраф за просрочку?" --as-of 2025-03-15'
Write-Host '    docker compose run --rm cli ask    "Продлевается ли срок для малого бизнеса?"'
Write-Host ""
Write-Host 'ask needs the generator: docker compose --profile llm up -d llm'
