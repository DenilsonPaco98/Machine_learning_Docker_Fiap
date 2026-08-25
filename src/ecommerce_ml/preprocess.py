from __future__ import annotations

import pandas as pd

from config import settings


def prepare_data() -> None:
    """Carrega o dataset bruto, limpa colunas e salva arquivos processados."""
    raw_path = settings.raw_data_dir / "online_shoppers_intention.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_path}")

    df = pd.read_csv(raw_path)
    df = df.copy()

    target = "Revenue"
    if target not in df.columns:
        raise KeyError(f"Coluna alvo '{target}' não foi encontrada no dataset.")

    df["Weekend"] = df["Weekend"].astype(str)
    df["VisitorType"] = df["VisitorType"].astype(str)

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_df.to_csv(settings.processed_data_dir / "train.csv", index=False)
    test_df.to_csv(settings.processed_data_dir / "test.csv", index=False)

    print(f"Dados processados em: {settings.processed_data_dir}")


if __name__ == "__main__":
    prepare_data()
