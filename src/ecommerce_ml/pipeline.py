from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# support running as a script or as a package
try:
    from .config import settings
except Exception:
    from config import settings


class DataProcessor:
    """Responsável por carregar e pré-processar os dados.

    Segue o princípio de responsabilidade única (SRP). Essa classe não treina
    modelos — apenas prepara DataFrames prontos para treinamento/inferência.
    """

    def __init__(self) -> None:
        self.categorical_columns: List[str] = [
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
        ]
        # After notebook-style preprocessing most columns are numeric
        self.numeric_columns: List[str] = [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
        ]

    def load_raw(self, path: Path | str) -> pd.DataFrame:
        """Carrega o CSV bruto do caminho informado.

        Lança FileNotFoundError se o arquivo não existir.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return pd.read_csv(path)

    def split_train_test(
        self, df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide o DataFrame em conjuntos de treino e teste e salva em disco.

        Aplica pré-processamento no estilo do notebook antes da divisão
        (mapeamento de Month, label encoding de VisitorType, casts).
        """
        df = df.drop_duplicates().reset_index(drop=True)

        df = self.preprocess_notebook_style(df)

        train_df = df.sample(frac=1 - test_size, random_state=random_state)
        test_df = df.drop(train_df.index)

        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(settings.processed_data_dir / "train.csv", index=False)
        test_df.to_csv(settings.processed_data_dir / "test.csv", index=False)
        return train_df, test_df

    def preprocess_notebook_style(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações vistas no notebook: map Month, label-encode VisitorType,
        cast Weekend e Revenue para int.
        """
        df = df.copy()

        month_order = {
            "Feb": 1,
            "Mar": 2,
            "May": 3,
            "June": 4,
            "Jul": 5,
            "Aug": 6,
            "Sep": 7,
            "Oct": 8,
            "Nov": 9,
            "Dec": 10,
        }
        if "Month" in df.columns:
            df["Month"] = df["Month"].map(month_order).fillna(0).astype(int)

        if "VisitorType" in df.columns:
            le = LabelEncoder()
            df["VisitorType"] = le.fit_transform(df["VisitorType"].astype(str))

        if "Weekend" in df.columns:
            df["Weekend"] = df["Weekend"].astype(int)

        if "Revenue" in df.columns:
            df["Revenue"] = df["Revenue"].astype(int)

        return df

    def build_feature_pipeline(self, classifier=None) -> Pipeline:
        """Retorna o pipeline de pré-processamento + classificador.

        Por padrão o classificador usado é `RandomForestClassifier` (o melhor
        segundo o notebook). Aceita injeção de `classifier` apenas para testes
        e reprodutibilidade.
        """
        # Aplica imputação e escalonamento (como no notebook) e, por padrão,
        # usa RandomForest para classificação.
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )


class ModelTrainer:
    """Classe que encapsula lógica de treinamento e persistência do modelo.

    Segue injeção de dependência: recebe um DataProcessor para preparar dados.
    """

    def __init__(self, processor: DataProcessor) -> None:
        self.processor = processor
    def train_random_forest(self) -> dict:
        """Treina apenas o `RandomForest` usando o pipeline definido e retorna
        um dicionário com `name`, `model`, `metrics` e `path`.
        """
        train_path = settings.processed_data_dir / "train.csv"
        test_path = settings.processed_data_dir / "test.csv"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target = "Revenue"
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target].astype(int)
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target].astype(int)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        pipeline = self.processor.build_feature_pipeline(classifier=clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from joblib import dump

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        model_filename = "random_forest.joblib"
        model_path = settings.models_dir / model_filename
        dump(pipeline, model_path)

        return {"name": "random_forest", "model": pipeline, "metrics": metrics, "path": model_path}
