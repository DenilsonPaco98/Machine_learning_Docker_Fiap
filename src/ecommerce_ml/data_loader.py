from __future__ import annotations

from pathlib import Path

try:
    from .config import settings
except Exception:
    # allow running as script from project root
    from config import settings


def download_dataset() -> Path:
    """Download dataset from Kaggle or fallback to local CSV if available.

    Behavior:
    - If there is already a CSV in `data/raw/`, use it and return its path.
    - Otherwise try to import `kagglehub` and download the dataset.
    - If `kagglehub` is missing or download fails and no local CSV exists,
      raise an informative error explaining required steps (install package
      and add `~/.kaggle/kaggle.json` or set KAGGLE_USERNAME/KAGGLE_KEY).
    """
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    # 1) If CSV already present, use it (allow offline runs)
    existing = list(settings.raw_data_dir.glob("*.csv"))
    if existing:
        chosen = existing[0]
        print(f"Using existing local CSV: {chosen}")
        return chosen

    # 2) Try to download via kagglehub
    try:
        import kagglehub
    except Exception:
        raise ModuleNotFoundError(
            "kagglehub is not installed and no local CSV found. "
            "Install kagglehub (`pip install kagglehub`) or place the CSV in data/raw/; "
            "also ensure your Kaggle credentials are available at ~/.kaggle/kaggle.json or via KAGGLE_USERNAME/KAGGLE_KEY."
        )

    try:
        dataset_path = kagglehub.dataset_download(settings.kaggle_dataset_name)
        downloaded_file = next(Path(dataset_path).glob("*.csv"), None)

        if downloaded_file is None:
            raise FileNotFoundError(f"Nenhum CSV foi encontrado em {dataset_path}")

        target_path = settings.raw_data_dir / downloaded_file.name
        target_path.write_bytes(downloaded_file.read_bytes())
        print(f"Dataset baixado e salvo em: {target_path}")
        return target_path
    except Exception as exc:  # noqa: BLE001 - surface helpful message
        # final fallback: if local CSV present now, use it; otherwise raise
        existing = list(settings.raw_data_dir.glob("*.csv"))
        if existing:
            print("Download falhou, mas local CSV já existe; usando local file.")
            return existing[0]
        raise RuntimeError(
            "Erro ao baixar o dataset com kagglehub: " + str(exc) +
            ".\nColoque o arquivo CSV em data/raw/online_shoppers_intention.csv "
            "ou configure suas credenciais Kaggle em ~/.kaggle/kaggle.json"
        )


if __name__ == "__main__":
    download_dataset()
