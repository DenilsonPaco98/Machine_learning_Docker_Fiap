#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x .venv/bin/dvc ]; then
  .venv/bin/dvc repro
elif [ -x .venv/Scripts/dvc.exe ]; then
  .venv/Scripts/dvc.exe repro
else
  echo "dvc executable not found in .venv. Ensure you ran ./scripts/setup.sh or installed dvc." >&2
  exit 1
fi
