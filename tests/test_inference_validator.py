import pytest

import pandas as pd

from ecommerce_ml.inference import InferenceValidator


def make_valid_payload():
    return {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 10,
        "ProductRelated_Duration": 200.0,
        "BounceRates": 0.02,
        "ExitRates": 0.01,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": "May",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }


def test_validator_accepts_valid_dict():
    v = InferenceValidator()
    payload = make_valid_payload()
    df = v.validate(payload)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1


def test_validator_rejects_missing_columns():
    v = InferenceValidator()
    payload = make_valid_payload()
    payload.pop("Month")
    with pytest.raises(ValueError):
        v.validate(payload)
