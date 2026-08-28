#!/usr/bin/env bash
set -euo pipefail

# Universal Git Bash setup with Poetry

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f pyproject.toml ]; then
  echo "pyproject.toml not found in project root"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtualenv .venv..."
  python3 -m venv .venv || python -m venv .venv
fi

VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

if [ ! -f "$VENV_PY" ]; then
  # Windows Git Bash: fallback to Scripts path
  VENV_PY=".venv/Scripts/python.exe"
  VENV_PIP=".venv/Scripts/pip.exe"
fi

echo "Installing Poetry into project venv..."
"$VENV_PIP" install --upgrade pip
"$VENV_PIP" install poetry

POETRY_BIN=".venv/bin/poetry"
if [ ! -f "$POETRY_BIN" ]; then
  POETRY_BIN=".venv/Scripts/poetry.exe"
fi

echo "Configuring Poetry to use in-project virtualenv and installing dependencies..."
"$POETRY_BIN" config virtualenvs.in-project true --local
"$POETRY_BIN" install --with dev --no-interaction

echo "Setup complete. To run in Git Bash:" 
echo "  $POETRY_BIN run python src/ecommerce_ml/train.py"
echo "  $POETRY_BIN run dvc repro"
