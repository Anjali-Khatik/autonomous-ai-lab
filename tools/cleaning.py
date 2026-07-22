"""Leakage-safe cleaning pipeline for the Data Scientist agent.

Generalized: works on any tabular dataset with a target column, no
dataset-specific column names. Fits imputation/encoding/scaling on the
TRAIN split only (rule §3), persists the fitted ColumnTransformer, and
applies it to both splits.

`cleaned_data_path` in the return value is a DIRECTORY (not a single
file) containing:
  train.parquet   - cleaned train split (features + target column)
  test.parquet    - cleaned, UNTOUCHED test split (features + target column)
  pipeline.joblib - fitted sklearn ColumnTransformer

The separate test split lives here (not created later by ML Engineer) so
the leakage-safety rule ("fit transforms on train only") is enforced at
the one place transforms are fit, and every downstream agent reads the
same untouched test set from a well-known path.
"""

import hashlib
import os
import uuid

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tools.eda import likely_identifier_columns

OUTPUT_ROOT = "outputs/data_scientist"
TEST_SIZE = 0.2


def _hash_files(*paths: str) -> str:
    sha = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            sha.update(f.read())
    return f"sha256:{sha.hexdigest()}"


def build_clean_pipeline(df_path: str, target: str, family: str, seed: int, run_id: str | None = None) -> dict:
    """Fit cleaning pipeline on TRAIN split only, persist it, apply to both splits.

    Returns:
        {
          "cleaned_data_path": str,   # directory with train.parquet + test.parquet
          "cleaned_data_hash": str,   # sha256 over both parquet files
          "pipeline_path": str,
          "report": {
            "rows_before": int, "dropped_columns": [...],
            "final_shape": [rows_total, n_features],
            "numeric_columns": [...], "categorical_columns": [...],
            "train_rows": int, "test_rows": int,
          }
        }
    """
    df = pd.read_csv(df_path)
    rows_before = len(df)

    identifier_cols = likely_identifier_columns(df, target)
    feature_df = df.drop(columns=[target] + identifier_cols)
    y = df[target]

    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    stratify = y if family in ("binary", "multiclass") else None
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=TEST_SIZE, random_state=seed, stratify=stratify
    )

    transformers = []
    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

    ct = ColumnTransformer(transformers)
    X_train_t = ct.fit_transform(X_train)
    X_test_t = ct.transform(X_test)
    feature_names = list(ct.get_feature_names_out())

    train_df = pd.DataFrame(X_train_t, columns=feature_names)
    train_df[target] = y_train.reset_index(drop=True)
    test_df = pd.DataFrame(X_test_t, columns=feature_names)
    test_df[target] = y_test.reset_index(drop=True)

    run_id = run_id or uuid.uuid4().hex[:12]
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    pipeline_path = os.path.join(out_dir, "pipeline.joblib")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    joblib.dump(ct, pipeline_path)

    report = {
        "rows_before": rows_before,
        "dropped_columns": identifier_cols,
        "final_shape": [rows_before, len(feature_names)],
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    return {
        "cleaned_data_path": out_dir,
        "cleaned_data_hash": _hash_files(train_path, test_path),
        "pipeline_path": pipeline_path,
        "report": report,
    }
