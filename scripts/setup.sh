#!/usr/bin/env bash
set -euo pipefail

# Universal Git Bash setup: create .venv and install dependencies from requirements.txt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found in project root"
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

echo "Upgrading pip and installing requirements..."
"$VENV_PIP" install --upgrade pip
"$VENV_PIP" install -r requirements.txt

echo "Setup complete. To run in Git Bash:" 
echo "  source .venv/bin/activate  # or .venv/Scripts/activate on Windows cmd"
echo "  ./scripts/run_train.sh"
