"""Standalone single-model/single-split evaluation.

Reuses the same scoring logic read_leaderboard() uses internally, for
callers (e.g. Critic, re-evaluation after a retrain) that need one
model's honest test metrics without re-reading the whole leaderboard.
"""

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import confusion_matrix

from config import FAMILY_METRICS
from tools.engine_wrapper import manual_secondary_metrics, split_metrics


def evaluate_on_test(predictor_path: str, model_name: str, test_path: str, family: str) -> dict:
    """Family-appropriate metrics for a single leaderboard model on the untouched test set.

    Returns a metrics dict matching ModelResult["metrics"]["test"] shape.
    """
    predictor = TabularPredictor.load(predictor_path)
    test = pd.read_parquet(test_path)
    target = predictor.label
    primary_metric = FAMILY_METRICS[family]["primary_metric"]

    lb_test = split_metrics(predictor, test, family, primary_metric)
    metric_cols = [c for c in lb_test.columns if c not in (
        "score_val", "eval_metric", "pred_time_test", "pred_time_val", "fit_time",
        "pred_time_test_marginal", "pred_time_val_marginal", "fit_time_marginal",
        "stack_level", "can_infer", "fit_order",
    )]

    test_metrics = {k: round(float(v), 4) for k, v in lb_test.loc[model_name, metric_cols].items() if pd.notna(v)}

    class_labels = list(predictor.class_labels) if family in ("binary", "multiclass") else None
    y_pred = predictor.predict(test.drop(columns=[target]), model=model_name)
    test_metrics.update(manual_secondary_metrics(family, test[target], y_pred, class_labels))

    if family in ("binary", "multiclass"):
        cm = confusion_matrix(test[target], y_pred, labels=class_labels)
        test_metrics["confusion_matrix"] = cm.tolist()

    return test_metrics
