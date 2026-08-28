#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x .venv/bin/poetry ]; then
  .venv/bin/poetry run python src/ecommerce_ml/train.py
elif [ -x .venv/Scripts/poetry.exe ]; then
  .venv/Scripts/poetry.exe run python src/ecommerce_ml/train.py
else
  echo "Poetry executable not found in .venv. Run ./scripts/setup.sh first." >&2
  exit 1
fi
