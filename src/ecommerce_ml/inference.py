from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


class InferenceValidator:
    """Valida e sanitiza entradas de inferência para o modelo.

    Usa políticas conservadoras: exige colunas necessárias, converte tipos
    básicos e rejeita entradas com formatos inválidos levantando `ValueError`.
    """

    required_columns: List[str] = [
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
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]

    def validate(self, payload: Dict | pd.DataFrame) -> pd.DataFrame:
        """Recebe um dicionário (uma amostra) ou DataFrame e retorna um DataFrame válido.

        Levanta ValueError em caso de problemas de formato ou ausência de colunas.
        """
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
        elif isinstance(payload, pd.DataFrame):
            df = payload.copy()
        else:
            raise ValueError("Payload must be a dict or pandas.DataFrame")

        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Type conversions / basic sanitization
        # Weekend: allow truthy/falsy values
        df["Weekend"] = df["Weekend"].astype(int)

        # Ensure numeric columns are numeric
        numeric_cols = [
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
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")

        # VisitorType normalization: map common variants
        df["VisitorType"] = df["VisitorType"].astype(str).str.strip()

        return df[self.required_columns]
