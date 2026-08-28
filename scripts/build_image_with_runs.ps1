param(
  [string]$ArtifactsDir = "./mlflow/artifacts",
  [string]$ImageName = "ecommerce-trainer:latest"
)

if (-not (Test-Path $ArtifactsDir) -or -not (Get-ChildItem -Path $ArtifactsDir -Recurse -ErrorAction SilentlyContinue)) {
  Write-Warning "$ArtifactsDir está vazio ou não existe. Execute o treino primeiro para gerar runs/artifacts."
} else {
  Write-Host "Usando artefatos existentes em $ArtifactsDir"
}

Write-Host "Construindo imagem Docker: $ImageName"
docker build -t $ImageName .
Write-Host "Imagem construída: $ImageName"
