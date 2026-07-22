"""Filter + embedded feature selection for the Data Scientist agent.

Generalized: operates on whatever numeric feature columns the cleaning
pipeline produced (post one-hot encoding), no dataset-specific names.

Feature EXTRACTION (PCA/LDA/autoencoder) is intentionally NOT implemented
here — spec marks it conditional and off-by-default for tree models
(§3 Agent 2). TODO: gate it on dimensionality/collinearity if a future
dataset needs it.
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

from config import DEFAULT_SEED

CORRELATION_THRESHOLD = 0.90
IMPORTANCE_PRUNE_THRESHOLD = 0.01
MIN_FEATURES = 3


def _drop_correlated(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if X.shape[1] < 2:
        return X, []
    corr = X.corr().abs()
    dropped = []
    kept = list(X.columns)
    for i, col_i in enumerate(X.columns):
        if col_i in dropped:
            continue
        for col_j in X.columns[i + 1:]:
            if col_j in dropped:
                continue
            if corr.loc[col_i, col_j] > CORRELATION_THRESHOLD:
                dropped.append(col_j)
    kept = [c for c in kept if c not in dropped]
    return X[kept], dropped


def select_features(cleaned_path: str, target: str, family: str) -> dict:
    """Filter (variance + correlation) then embedded (tree importance) selection.

    Returns:
        {"feature_list": [...], "dropped": [{"feature": str, "reason": str}], "method": str}
    """
    train_df = pd.read_parquet(os.path.join(cleaned_path, "train.parquet"))
    X = train_df.drop(columns=[target])
    y = train_df[target]

    dropped: list[dict] = []

    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X)
    zero_var_cols = [c for c, keep in zip(X.columns, vt.get_support()) if not keep]
    for c in zero_var_cols:
        dropped.append({"feature": c, "reason": "zero_variance"})
    X = X.drop(columns=zero_var_cols)

    X_before_corr = list(X.columns)
    X, corr_dropped = _drop_correlated(X)
    for c in corr_dropped:
        dropped.append({"feature": c, "reason": f"correlation>{CORRELATION_THRESHOLD}"})

    if X.shape[1] > MIN_FEATURES:
        if family == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=DEFAULT_SEED)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=DEFAULT_SEED)
        model.fit(X, y)
        importances = dict(zip(X.columns, model.feature_importances_))
        low_importance = [c for c, imp in importances.items() if imp < IMPORTANCE_PRUNE_THRESHOLD]
        keepable = X.shape[1] - len(low_importance)
        if keepable < MIN_FEATURES:
            low_importance = sorted(importances, key=importances.get)[: max(0, X.shape[1] - MIN_FEATURES)]
        for c in low_importance:
            dropped.append({"feature": c, "reason": f"embedded_importance<{IMPORTANCE_PRUNE_THRESHOLD}"})
        X = X.drop(columns=low_importance)

    return {
        "feature_list": list(X.columns),
        "dropped": dropped,
        "method": "variance_threshold + correlation_filter + embedded_importance(random_forest)",
    }
