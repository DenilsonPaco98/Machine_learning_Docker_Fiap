from __future__ import annotations

import mlflow
from joblib import dump

# Support running as a script (python src/ecommerce_ml/train.py) and as a module
try:
    from .config import settings
    from .pipeline import DataProcessor, ModelTrainer
except Exception:
    # fallback for script execution where package context is not available
    from config import settings
    from pipeline import DataProcessor, ModelTrainer


def main() -> None:
    """Registra o modelo no MLflow usando o dataset processado do DVC.

    Essa versão delega preparação e treinamento para as classes em
    `pipeline.py` (POO, SRP)."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    processor = DataProcessor()
    trainer = ModelTrainer(processor)

    # Treinar um único RandomForest seguindo o pipeline do notebook
    processor = DataProcessor()

    # Se os dados ainda não estiverem processados, tente criar os arquivos
    raw_path = settings.raw_data_dir / "online_shoppers_intention.csv"
    if raw_path.exists():
        df_raw = processor.load_raw(raw_path)
        processor.split_train_test(df_raw, test_size=0.2, random_state=42)

    # Carrega os CSVs processados pelo DataProcessor
    train_path = settings.processed_data_dir / "train.csv"
    test_path = settings.processed_data_dir / "test.csv"

    import pandas as pd

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target = "Revenue"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].astype(int)

    # Construir pipeline com RandomForest
    model = processor.build_feature_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    # Save classification report and confusion matrix as artifacts
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt

    artifacts_dir = Path("./artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = artifacts_dir / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    import numpy as np
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    cm_path = artifacts_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    # Log to MLflow as a professional run
    run_name = "online_shoppers_random_forest"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model_type": "random_forest",
                "n_estimators": 100,
                "dataset_source": "DVC processed files or raw",
                "dataset_name": settings.kaggle_dataset_name,
                "test_size": 0.2,
                "random_state": 42,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "task": "binary_classification",
                "dataset": settings.kaggle_dataset_name,
                "model_type": "random_forest",
                "team": "data-science",
                "env": "dev",
            }
        )

        # Save and register model
        model_path = settings.models_dir / "random_forest.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, model_path)

        import numpy as np

        # Prefer using an MLflow PyFunc model wrapper; this creates a proper
        # MLflow model directory (with MLmodel) and avoids skops serialization
        # issues. We log the pyfunc model under artifact path 'random_forest_model'
        # and register it in the Model Registry.
        registered_name = f"{settings.model_name}-random_forest"

        class _PyFuncModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import joblib
                # `context.artifacts["model"]` is the path to the joblib file
                self._model = joblib.load(context.artifacts["model"])

            def predict(self, context, model_input):
                return self._model.predict(model_input)

        try:
            # Try logging the pyfunc model directly into the run and register it
            mlflow.pyfunc.log_model(
                artifact_path="random_forest_model",
                python_model=_PyFuncModel(),
                artifacts={"model": str(model_path)},
                registered_model_name=registered_name,
            )
        except Exception as exc:
            print("mlflow.pyfunc.log_model failed, attempting save+register fallback:", exc)
            # Fallback: save a local MLflow model dir and register from file:// URI
            tmp_dir = Path("./models") / "random_forest_mlflow"
            if tmp_dir.exists():
                import shutil

                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            mlflow.pyfunc.save_model(
                path=str(tmp_dir), python_model=_PyFuncModel(), artifacts={"model": str(model_path)}
            )
            artifact_uri = f"file://{tmp_dir.resolve()}"
            try:
                mlflow.register_model(artifact_uri, registered_name)
            except Exception as reg_exc:
                print("mlflow.register_model fallback failed:", reg_exc)

        # Try to fetch the registered model version and record it in the run
        try:
            client = mlflow.tracking.MlflowClient()
            # search_model_versions returns ModelVersion objects; filter by name
            versions = client.search_model_versions(f"name='{registered_name}'")
            if versions:
                # pick the latest numeric version
                ver_nums = [int(v.version) for v in versions if v.version is not None]
                latest_ver = str(max(ver_nums)) if ver_nums else None
                if latest_ver:
                    mlflow.set_tag("registered_model_version", latest_ver)
                    mlflow.log_param("registered_model_version", latest_ver)
                    print(f"Registered model version: {latest_ver}")
        except Exception as ver_exc:
            print("Unable to fetch or set registered model version:", ver_exc)

            # Ensure the dataset is logged in a way MLflow UI recognizes.
            # Prefer mlflow.data.log_dataset when available (MLflow >= certain versions).
            try:
                dataset_name = settings.kaggle_dataset_name
                data_api = getattr(mlflow, "data", None)

                # First, upload the processed train CSV as an artifact so MLflow can
                # reference it via a `runs:/.../artifacts/...` URI which the UI expects.
                mlflow.log_artifact(str(train_path), artifact_path="datasets")
                artifact_rel = f"datasets/{train_path.name}"
                runs_uri = f"runs:/{run.info.run_id}/artifacts/{artifact_rel}"

                if data_api and hasattr(data_api, "log_dataset"):
                    try:
                        data_api.log_dataset(runs_uri, name=dataset_name)
                    except TypeError:
                        data_api.log_dataset(runs_uri, dataset_name)
                else:
                    # Fallback: set explicit tags/params so the dataset name appears
                    mlflow.set_tag("mlflow.dataset.name", dataset_name)
                    mlflow.set_tag("dataset_path", runs_uri)
                    mlflow.log_param("dataset_name", dataset_name)
            except Exception as data_exc:
                print("Unable to log dataset via mlflow.data, falling back to tags:", data_exc)
                try:
                    mlflow.set_tag("dataset", settings.kaggle_dataset_name)
                except Exception:
                    pass

        # Log artifacts
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))

        print(f"Run ID (random_forest): {run.info.run_id}")
        print(metrics)


if __name__ == "__main__":
    main()
