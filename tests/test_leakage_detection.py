"""Acceptance test (spec §6): leakage safety.

"Injecting a target-leaking feature (e.g. an ID correlated with the
target) is flagged by check_leakage." Fully deterministic — no LLM, no
training — fast. Bypasses the real cleaning pipeline's own identifier-
uniqueness filter on purpose (constructs train/test parquet directly) to
test check_leakage's OWN detection independent of that earlier safety
net — i.e. what if a low-cardinality, ID-shaped column slips past
cleaning because it repeats across rows and never gets flagged by the
nunique/n_rows>=0.95 heuristic in tools/cleaning.py.
"""

import os

import numpy as np
import pandas as pd
import pytest

from agents.critic import _build_findings
from tools.leakage import check_leakage
from tools.metric_check import check_metric_choice
from tools.overfitting import check_overfitting
from tools.validation_check import check_validation


def _write_cleaned_dir(tmp_path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    cleaned_dir = tmp_path / "cleaned"
    os.makedirs(cleaned_dir, exist_ok=True)
    train_df.to_parquet(cleaned_dir / "train.parquet", index=False)
    test_df.to_parquet(cleaned_dir / "test.parquet", index=False)
    return str(cleaned_dir)


def _synthetic_split(rng, leak_values):
    n = 100
    target = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "numeric__weak_signal": rng.normal(0, 1, n),
        "numeric__batch_id": leak_values(target, rng, n),
        "target": target,
    })
    return df.iloc[:70].reset_index(drop=True), df.iloc[70:].reset_index(drop=True)


def test_check_leakage_flags_id_shaped_leaking_feature(tmp_path):
    rng = np.random.default_rng(42)
    # low-cardinality (10 buckets over 100 rows -> nunique/n=0.10, well under the 0.95
    # identifier-uniqueness threshold tools/cleaning.py uses) but PERFECTLY determines
    # the target -> a leak that could slip past the earlier cleaning-stage safety net.
    leak = lambda target, rng, n: target * 5 + rng.integers(0, 5, n)
    train_df, test_df = _synthetic_split(rng, leak)
    cleaned_path = _write_cleaned_dir(tmp_path, train_df, test_df)

    feature_importance = {"numeric__batch_id": 0.95, "numeric__weak_signal": 0.05}
    result = check_leakage(cleaned_path, feature_importance, family="binary")

    assert result["detected"] is True
    dominant = next(f for f in result["findings"] if f["type"] == "suspicious_high_importance")
    assert dominant["feature"] == "numeric__batch_id"
    assert dominant["looks_like_id_or_date"] is True

    # And the Critic's own severity-escalation logic must treat an ID-shaped dominant
    # feature as "high" (forces reject) — not the same as a genuinely informative one.
    overfit = check_overfitting({"train": {"f1": 0.9}, "val": {"f1": 0.9}, "test": {"f1": 0.89}}, "binary")
    metric = check_metric_choice("f1", "binary", {"0": 0.5, "1": 0.5})
    validation = check_validation({}, "binary")
    findings = _build_findings(overfit, result, metric, validation)
    leakage_finding = next(f for f in findings if f["type"] == "leakage")
    assert leakage_finding["severity"] == "high"


def test_check_leakage_does_not_flag_genuinely_informative_dominant_feature(tmp_path):
    """Contrast case: a dominant feature that does NOT look like an ID/date
    (e.g. Iris's real PetalLengthCm situation) must NOT force a reject —
    see PROGRESS.md 2026-07-20 deviations log for the real bug this guards.
    """
    rng = np.random.default_rng(7)
    leak = lambda target, rng, n: target * 5 + rng.integers(0, 5, n)
    train_df, test_df = _synthetic_split(rng, leak)
    train_df = train_df.rename(columns={"numeric__batch_id": "numeric__petal_length"})
    test_df = test_df.rename(columns={"numeric__batch_id": "numeric__petal_length"})
    cleaned_path = _write_cleaned_dir(tmp_path, train_df, test_df)

    feature_importance = {"numeric__petal_length": 0.95, "numeric__weak_signal": 0.05}
    result = check_leakage(cleaned_path, feature_importance, family="binary")

    dominant = next(f for f in result["findings"] if f["type"] == "suspicious_high_importance")
    assert dominant["looks_like_id_or_date"] is False

    overfit = check_overfitting({"train": {"f1": 0.9}, "val": {"f1": 0.9}, "test": {"f1": 0.89}}, "binary")
    metric = check_metric_choice("f1", "binary", {"0": 0.5, "1": 0.5})
    validation = check_validation({}, "binary")
    findings = _build_findings(overfit, result, metric, validation)
    leakage_finding = next(f for f in findings if f["type"] == "leakage")
    assert leakage_finding["severity"] == "moderate"  # not "high" — doesn't force reject


def test_check_leakage_flags_train_test_distribution_shift(tmp_path):
    """Second real leakage-safety mechanism: KS distribution test between
    the leakage-safe train split and the untouched test split.
    """
    rng = np.random.default_rng(3)
    n = 100
    train_df = pd.DataFrame({
        "numeric__stable": rng.normal(0, 1, n),
        "target": rng.integers(0, 2, n),
    })
    test_df = pd.DataFrame({
        "numeric__stable": rng.normal(50, 1, n),  # shifted distribution
        "target": rng.integers(0, 2, n),
    })
    cleaned_path = _write_cleaned_dir(tmp_path, train_df, test_df)

    result = check_leakage(cleaned_path, {}, family="binary")
    assert result["detected"] is True
    shift = next(f for f in result["findings"] if f["type"] == "distribution_shift")
    assert shift["feature"] == "numeric__stable"
    assert shift["p_value"] < 0.01
