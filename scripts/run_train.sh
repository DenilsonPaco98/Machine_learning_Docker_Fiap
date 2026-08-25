#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x .venv/bin/python ]; then
  .venv/bin/python src/ecommerce_ml/train.py
elif [ -x .venv/Scripts/python.exe ]; then
  .venv/Scripts/python.exe src/ecommerce_ml/train.py
else
  echo "Python executable not found in .venv. Run ./scripts/setup.sh first." >&2
  exit 1
fi
