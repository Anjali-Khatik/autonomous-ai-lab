"""Deterministic family detection + quick profile for the Research Planner.

Generalized: inference from target dtype/cardinality only, no
dataset-specific assumptions. V1 only supports binary/multiclass/
regression — forecast/cluster/anomaly are [V2] stubs (config.MODEL_REGISTRY
is empty for them), so detection never returns those families even if the
data superficially looks like one; it reports low confidence instead.
"""

import pandas as pd

from config import FAMILY_METRICS

MULTICLASS_MAX_CARDINALITY_FOR_HIGH_CONFIDENCE = 20
INTEGER_LOW_CARDINALITY_THRESHOLD = 20
REGRESSION_UNIQUE_RATIO_HIGH_CONFIDENCE = 0.05


def detect_problem_family(df_path: str, target: str | None) -> dict:
    """Infer problem_family from target dtype/cardinality/class balance.

    Returns {"family": str|None, "primary_metric": str|None,
             "signals": {...}, "confidence": float}.
    family is None when V1 can't confidently support what it sees
    (no target column, or an ambiguous/unsupported case) — caller must
    treat that as a hard stop, not guess.
    """
    df = pd.read_csv(df_path)

    if target is None or target not in df.columns:
        return {
            "family": None, "primary_metric": None,
            "signals": {"reason": "no valid target_column provided — V1 only supports supervised families"},
            "confidence": 0.0,
        }

    y = df[target].dropna()
    n = len(y)
    if n == 0:
        return {"family": None, "primary_metric": None, "signals": {"reason": "target column is entirely null"}, "confidence": 0.0}

    n_unique = int(y.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(y)
    is_integer_like = is_numeric and bool((y.dropna() % 1 == 0).all())

    signals = {
        "target_dtype": str(y.dtype), "n_unique": n_unique, "n_rows": n,
        "is_numeric": is_numeric, "is_integer_like": is_integer_like,
    }

    if n_unique == 2:
        family, confidence = "binary", 0.95
        signals["class_balance"] = {str(k): round(float(v), 4) for k, v in y.value_counts(normalize=True).items()}
    elif not is_numeric and n_unique > 2:
        family = "multiclass"
        confidence = 0.9 if n_unique <= MULTICLASS_MAX_CARDINALITY_FOR_HIGH_CONFIDENCE else 0.55
        signals["class_balance"] = {str(k): round(float(v), 4) for k, v in y.value_counts(normalize=True).items()}
    elif is_numeric and is_integer_like and n_unique <= INTEGER_LOW_CARDINALITY_THRESHOLD:
        family, confidence = "multiclass", 0.6
        signals["note"] = "numeric integer target with low cardinality — treated as multiclass; likely ordinal, override if this is really a regression target"
        signals["class_balance"] = {str(k): round(float(v), 4) for k, v in y.value_counts(normalize=True).items()}
    elif is_numeric:
        unique_ratio = n_unique / n
        family = "regression"
        confidence = 0.9 if unique_ratio > REGRESSION_UNIQUE_RATIO_HIGH_CONFIDENCE else 0.7
        signals["unique_ratio"] = round(unique_ratio, 4)
    else:
        family, confidence = None, 0.0
        signals["reason"] = "could not confidently classify target dtype/cardinality into a V1-supported family"

    primary_metric = FAMILY_METRICS[family]["primary_metric"] if family else None
    return {"family": family, "primary_metric": primary_metric, "signals": signals, "confidence": round(confidence, 2)}


def quick_profile(df_path: str) -> dict:
    """Fast profile: shape, dtypes, missing %, memory estimate."""
    df = pd.read_csv(df_path)
    dtype_counts = {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()}
    return {
        "shape": list(df.shape),
        "dtypes": dtype_counts,
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 4),
    }
