"""Acceptance test (spec §6): family detection.

Classification and regression datasets resolve to the correct
problem_family and primary_metric; a time-series-shaped dataset does NOT
get fabricated into "forecast" ([V2], not implemented). Fully
deterministic — no LLM, no training — fast.
"""

import pandas as pd
import pytest

from config import FAMILY_METRICS
from tools.task_detection import detect_problem_family


@pytest.mark.parametrize("dataset_path,target,expected_family", [
    ("dataset/loan_dataset.csv", "Loan_Status", "binary"),
    ("dataset/Iris.csv", "Species", "multiclass"),
    ("dataset/Housing.csv", "price", "regression"),
])
def test_family_detection_real_datasets(dataset_path, target, expected_family):
    result = detect_problem_family(dataset_path, target)
    assert result["family"] == expected_family
    assert result["primary_metric"] == FAMILY_METRICS[expected_family]["primary_metric"]
    assert result["confidence"] > 0.5


def test_family_detection_no_target_is_hard_stop():
    result = detect_problem_family("dataset/loan_dataset.csv", None)
    assert result["family"] is None
    assert result["primary_metric"] is None
    assert result["confidence"] == 0.0


def test_family_detection_time_series_not_fabricated_as_forecast(tmp_path):
    """A date-indexed numeric target should resolve to a real V1 family
    (regression — detect_problem_family has no temporal awareness) rather
    than inventing an unsupported "forecast" label it can't actually back
    with a working pipeline (forecast's MODEL_REGISTRY entry is empty).
    """
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=60, freq="D"),
        "sales": [100 + i * 1.5 for i in range(60)],
    })
    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)

    result = detect_problem_family(str(csv_path), "sales")
    assert result["family"] != "forecast"
    assert result["family"] in ("regression", None)
