"""Deterministic EDA for the Data Scientist agent.

Generalized across arbitrary tabular datasets: no dataset-specific column
names or assumptions. Everything here is computed by pandas/numpy/scipy —
the LLM only interprets this output afterward, it never produces numbers.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew

IDENTIFIER_UNIQUE_RATIO = 0.95


def _infer_column_types(df: pd.DataFrame, target: str) -> dict:
    numeric_cols, categorical_cols, datetime_cols = [], [], []
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue
        if series.dtype == object:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9:
                datetime_cols.append(col)
                continue
        categorical_cols.append(col)
    return {"numeric": numeric_cols, "categorical": categorical_cols, "datetime": datetime_cols}


def likely_identifier_columns(df: pd.DataFrame, target: str) -> list[str]:
    n = len(df)
    flagged = []
    for col in df.columns:
        if col == target or n == 0:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique / n >= IDENTIFIER_UNIQUE_RATIO:
            flagged.append(col)
    return flagged


def run_eda(df_path: str, target: str, family: str) -> dict:
    """Numeric/categorical/datetime EDA summary + quality score.

    Returns:
        {
          "shape": [rows, cols],
          "column_types": {"numeric": [...], "categorical": [...], "datetime": [...]},
          "missing_pct": float,               # overall
          "missing_by_column": {col: pct},
          "duplicate_rows": int,
          "likely_identifier_columns": [...],
          "numeric_summary": {col: {mean, std, min, max, outlier_pct}},
          "categorical_summary": {col: {n_unique, top_value, top_value_pct}},
          "correlations": {col: corr_with_first_numeric_or_target},
          "class_balance": {...} | None,      # classification families only
          "target_skewness": float | None,    # regression families only
          "quality_score": float,             # 0-1, higher is cleaner
        }
    """
    df = pd.read_csv(df_path)
    n_rows, n_cols = df.shape

    col_types = _infer_column_types(df, target)
    missing_by_column = (df.isna().mean() * 100).round(2).to_dict()
    missing_pct = round(float(df.isna().mean().mean() * 100), 2)
    duplicate_rows = int(df.duplicated().sum())
    identifier_cols = likely_identifier_columns(df, target)

    numeric_summary = {}
    outlier_fracs = []
    for col in col_types["numeric"]:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_pct = float(((series < lo) | (series > hi)).mean() * 100)
        else:
            outlier_pct = 0.0
        outlier_fracs.append(outlier_pct / 100)
        numeric_summary[col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "outlier_pct": round(outlier_pct, 2),
        }

    categorical_summary = {}
    for col in col_types["categorical"]:
        vc = df[col].value_counts(dropna=True)
        if vc.empty:
            continue
        categorical_summary[col] = {
            "n_unique": int(df[col].nunique(dropna=True)),
            "top_value": str(vc.index[0]),
            "top_value_pct": round(float(vc.iloc[0] / vc.sum() * 100), 2),
        }

    correlations = {}
    if col_types["numeric"]:
        corr_matrix = df[col_types["numeric"]].corr(numeric_only=True)
        if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            with_target = df[col_types["numeric"] + [target]].corr(numeric_only=True)[target].drop(target, errors="ignore")
            correlations = {k: round(float(v), 4) for k, v in with_target.items() if pd.notna(v)}
        elif len(col_types["numeric"]) > 1:
            first = col_types["numeric"][0]
            correlations = {k: round(float(v), 4) for k, v in corr_matrix[first].items() if k != first and pd.notna(v)}

    class_balance = None
    target_skewness = None
    if family in ("binary", "multiclass") and target in df.columns:
        vc = df[target].value_counts(normalize=True, dropna=True)
        class_balance = {str(k): round(float(v), 4) for k, v in vc.items()}
    elif family == "regression" and target in df.columns:
        target_vals = df[target].dropna()
        if len(target_vals) > 2:
            target_skewness = round(float(skew(target_vals)), 4)

    mean_outlier_frac = float(np.mean(outlier_fracs)) if outlier_fracs else 0.0
    duplicate_frac = duplicate_rows / n_rows if n_rows else 0.0
    quality_score = round(max(0.0, 1 - (missing_pct / 100 * 0.5 + duplicate_frac * 0.3 + mean_outlier_frac * 0.2)), 4)

    return {
        "shape": [n_rows, n_cols],
        "column_types": col_types,
        "missing_pct": missing_pct,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "likely_identifier_columns": identifier_cols,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "class_balance": class_balance,
        "target_skewness": target_skewness,
        "quality_score": quality_score,
    }
