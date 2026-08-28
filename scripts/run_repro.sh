#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x .venv/bin/poetry ]; then
  .venv/bin/poetry run dvc repro
elif [ -x .venv/Scripts/poetry.exe ]; then
  .venv/Scripts/poetry.exe run dvc repro
else
  echo "Poetry executable not found in .venv. Ensure you ran ./scripts/setup.sh." >&2
  exit 1
fi
