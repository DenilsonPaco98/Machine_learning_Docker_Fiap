from pathlib import Path


def test_required_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    required = [
        root / "pyproject.toml",
        root / ".gitignore",
        root / ".dockerignore",
        root / ".env.example",
        root / "Dockerfile",
        root / "dvc.yaml",
        root / "README.md",
        root / "src" / "ecommerce_ml" / "train.py",
    ]

    for path in required:
        assert path.exists(), f"Arquivo ausente: {path}"
