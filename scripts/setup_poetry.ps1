<#
Setup script to make the repository "clone-and-run" friendly on Windows.

Behavior:
- Creates a local venv at `.venv` if it does not exist.
- Installs `poetry` into the venv (so the environment is self-contained).
- Configures Poetry to create in-project virtualenvs and runs `poetry install`.

Usage (PowerShell):
  .\scripts\setup_poetry.ps1
#>

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "Project root: $projectRoot"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment at .venv..."
    python -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python executable not found at $python. Ensure you have Python 3.11 installed and available as 'python' on PATH.";
    exit 1
}

Write-Host "Upgrading pip and installing poetry into the venv..."
& $python -m pip install --upgrade pip
& $python -m pip install poetry

$poetryExe = Join-Path $projectRoot ".venv\Scripts\poetry.exe"
if (-not (Test-Path $poetryExe)) {
    Write-Host "poetry executable not found in venv; trying 'poetry' from PATH"
    $poetryCmd = "poetry"
} else {
    $poetryCmd = $poetryExe
}

Write-Host "Configuring poetry to create venvs inside project..."
& $poetryCmd config virtualenvs.in-project true --local

Write-Host "Running 'poetry install' to install project dependencies..."
& $poetryCmd install --with dev --no-interaction

Write-Host "Setup complete. To run training using the venv-poetry combo, use:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\poetry.exe run python src\ecommerce_ml\train.py" -ForegroundColor Cyan
