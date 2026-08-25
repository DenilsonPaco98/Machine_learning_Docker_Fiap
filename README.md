# Tech Challenge - E-commerce Purchase Propensity

## Objetivo

Treinar um modelo clássico de classificação para prever a propensão de compra de usuários de e-commerce usando um pipeline reproducível com Docker, DVC e MLflow.

## Estrutura do projeto

- `src/ecommerce_ml/`: código principal
- `data/`: dados brutos e processados
- `models/`: artefatos do modelo
- `tests/`: testes
- `pyproject.toml`: dependências e configuração do Poetry
- `dvc.yaml`: pipeline do DVC
- `Dockerfile`: container do projeto

## Requisitos

- Python 3.11
- Poetry
- Docker
- DVC

## Instalação

```bash
poetry install
```

Git Bash / Unix (recommended):

1. Execute o script de bootstrap (criará `.venv` e instalará dependências via `pip`):

```bash
./scripts/setup.sh
```

2. Rodar o pipeline DVC (reproduzir stages):

```bash
./scripts/run_repro.sh
```

3. Rodar o treino:

```bash
./scripts/run_train.sh
```

Windows PowerShell (alternative):

```powershell
.\scripts\setup_poetry.ps1
#.\.venv\Scripts\poetry.exe run python src\ecommerce_ml\train.py
```

## Variáveis de ambiente

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

## MLflow

Inicie o servidor local:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Nota: antes de executar os scripts de treino, certifique-se de que o MLflow server está rodando no host/porta configurados em `src/ecommerce_ml/config.py` (`settings.mlflow_tracking_uri`). Se o servidor estiver em outra porta, atualize `MLFLOW_TRACKING_URI` no `.env` ou em `settings`.

## Execução

```bash
poetry run python src/ecommerce_ml/train.py
```

Descrição da construção do modelo
 - Pipeline de features: `SimpleImputer` (mediana) + `StandardScaler` aplicado às features numéricas e encoding/transformações mínimas para categóricas.
 - Modelo: `RandomForestClassifier(n_estimators=100)` encapsulado em um `sklearn` `Pipeline` que combina o pré-processamento e o estimador.
 - Saídas: `models/random_forest.joblib` (joblib do pipeline treinado), artefatos de métricas em `artifacts/` e o modelo registrado no MLflow como um `pyfunc` model (`artifact_path=random_forest_model`).

Boas práticas:
 - Reproduzibilidade: use o `.env` para configurar `MLFLOW_TRACKING_URI`, seeds e caminhos de dados.
 - Registro: o script salva o joblib e registra um `pyfunc` no MLflow para evitar problemas de serialização com skops.

Validação rápida do projeto
 - Rode `./scripts/run_repro.sh` para reproduzir todo o pipeline DVC (download → preprocess → train).
 - Verifique o run no MLflow UI (ex.: `http://127.0.0.1:8080`) e confirme que `random_forest_model` aparece na lista de artifacts e que a seção Datasets está preenchida.

## Docker

```bash
docker build -t ecommerce-ml .
docker run --rm ecommerce-ml
```

Docker + MLflow (recommended)

Use `docker-compose` to run an MLflow server and the training container locally. This `docker-compose.yml` starts:
- a lightweight `mlflow` server (SQLite backend + local artifact root)
- the `app` service that runs the training script and registers the model

Run:

```bash
docker compose up --build
```

Notes:
- The compose setup mounts `./mlruns` and `./mlflow_artifacts` so MLflow runs and artifacts persist on the host.
- The `app` service sets `MLFLOW_TRACKING_URI=http://mlflow:8080` so the training script talks to the MLflow server inside the compose network. Ensure `src/ecommerce_ml/config.py` respects this env var (it does).
- For production, replace the SQLite backend and local artifact root with a proper DB (Postgres) and object storage (S3) — update `docker-compose.yml` accordingly.

## DVC

```bash
dvc init
dvc repro
```
