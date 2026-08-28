#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS_DIR="./mlflow/artifacts"
IMAGE_NAME="ecommerce-trainer:latest"

if [ ! -d "$ARTIFACTS_DIR" ] || [ -z "$(ls -A "$ARTIFACTS_DIR")" ]; then
  echo "Aviso: $ARTIFACTS_DIR está vazio ou não existe. Execute o treino primeiro para gerar runs/artifacts." >&2
else
  echo "Usando artefatos existentes em $ARTIFACTS_DIR"
fi

echo "Construindo imagem Docker: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .
echo "Imagem construída: $IMAGE_NAME"
