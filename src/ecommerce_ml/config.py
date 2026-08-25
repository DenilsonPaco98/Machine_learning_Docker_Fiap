from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    # Allow overriding via environment variable (useful in Docker/CI)
    mlflow_tracking_uri: str = (
        __import__("os").environ.get("MLFLOW_TRACKING_URI")
        or "http://127.0.0.1:8080"
    )
    experiment_name: str = "online_shoppers_purchase_propensity"
    model_name: str = "online-shoppers-propensity-model"
    kaggle_dataset_name: str = "henrysue/online-shoppers-intention"


settings = Settings()
