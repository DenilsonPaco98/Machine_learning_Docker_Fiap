# Tech Challenge - E-commerce Purchase Propensity

Projeto de engenharia de Machine Learning para previsão de propensão de compra em e-commerce, com pipeline reproduzível usando Scikit-Learn, DVC, MLflow, Poetry e Docker.

## Objetivo

Treinar e registrar um modelo clássico de classificação (RandomForest) com rastreabilidade de experimentos e versionamento de dados.

## Estrutura do projeto

- `src/ecommerce_ml/`: código da aplicação e pipeline
- `data/`: dados brutos e processados
- `models/`: modelos serializados
- `artifacts/`: relatórios e figuras geradas no treino
- `tests/`: testes unitários
- `scripts/`: automações de setup e execução
- `dvc.yaml`: definição do pipeline DVC
- `docker-compose.yml`: MLflow + app com persistência

## Requisitos

- Python 3.11+
- Poetry
- Docker (opcional, para execução containerizada)

## Setup rápido (Poetry)

### Linux / macOS / Git Bash

```bash
./scripts/setup.sh
```

### Windows PowerShell

```powershell
.\scripts\setup_poetry.ps1
```

### Instalação manual equivalente

```bash
poetry install --with dev
```

O projeto usa `poetry.lock` para reprodutibilidade de dependências.

## Variáveis de ambiente

Copie e ajuste o template:

```bash
cp .env.example .env
```

Variável principal:

- `MLFLOW_TRACKING_URI` (default no código: `http://127.0.0.1:8080`)

## Executar pipeline local (sem Docker)

### 1) Rodar o pipeline DVC

```bash
./scripts/run_repro.sh
```

### 2) Rodar treino diretamente

```bash
./scripts/run_train.sh
```

Ou com Poetry explicitamente:

```bash
poetry run dvc repro
poetry run python src/ecommerce_ml/train.py
```

## MLflow local

Suba o servidor MLflow no host:

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```

Depois acesse:

- `http://127.0.0.1:8080`

O treino registra parâmetros, métricas, artifacts e modelo no MLflow.

## Docker + MLflow com persistência (recomendado)

O `docker-compose.yml` sobe:

- `mlflow`: servidor MLflow
- `app`: container de treino

Com persistência em disco local:

- `./mlflow/db` -> banco SQLite do MLflow
- `./mlflow/artifacts` -> artifacts/runs/modelos

### Subir ambiente

```bash
docker compose up --build -d
```

### Ver logs do MLflow

```bash
docker compose logs -f mlflow
```

### Executar um novo treino (adiciona novo run persistido)

```bash
docker compose run --rm app
```

### Rebuild de imagem reutilizando artifacts já persistidos

```bash
./scripts/build_image_with_runs.sh
```

PowerShell:

```powershell
.\scripts\build_image_with_runs.ps1
```

## DVC

Pipeline definido em `dvc.yaml` com estágios:

- `download_data`
- `preprocess`
- `train`

Comandos úteis:

```bash
poetry run dvc repro
poetry run dvc status
```

## Modelo e outputs

- Modelo principal: `RandomForestClassifier(n_estimators=100, random_state=42)`
- Pipeline de features: imputação (`SimpleImputer`) + escala (`StandardScaler`)
- Modelo salvo em: `models/random_forest.joblib`
- Artefatos principais: `artifacts/classification_report.json` e `artifacts/confusion_matrix.png`

## Testes

```bash
poetry run pytest -q
```

## Relatórios

- `reports/pipeline_analysis.html`
- `reports/tech_challenge_fase2_validacao.html`