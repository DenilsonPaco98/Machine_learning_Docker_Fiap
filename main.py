import mlflow
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("fake_classification_test")


def train_and_log_model(label: str, random_state: int, noise: float = 0.02):
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        flip_y=noise,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    with mlflow.start_run(run_name=label) as run:
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "solver": "lbfgs",
                "test_size": 0.25,
                "random_state": random_state,
                "noise_level": noise,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "dataset": "synthetic_classification",
                "task": "binary_classification",
                "quality": label,
            }
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="fake-classification-model",
        )

        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="fake-classification-model",
            )
            print(f"[{label}] Modelo registrado: {registered_model.name} v{registered_model.version}")
        except Exception as exc:
            print(f"[{label}] Registro do modelo não foi concluído: {exc}")

        print(f"[{label}] Run ID: {run.info.run_id}")
        print(f"[{label}] Metrics: {metrics}")


train_and_log_model("good_model", random_state=42, noise=0.02)
train_and_log_model("bad_model", random_state=7, noise=0.45)