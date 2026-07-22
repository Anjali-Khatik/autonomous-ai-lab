"""AutoGluon TabularPredictor wrapper for the ML Engineer agent.

Fit-once -> read-leaderboard, per spec §3 Agent 3's API note. NOT a
per-model training loop: candidate_models become a `hyperparameters`
restriction passed to one `fit()` call.

Run layout under outputs/ml_engineer/<run_id>/:
  predictor/          AutoGluon's own save dir (returned as predictor_path)
  train_fit.parquet   the train split actually fit on (train.parquet minus
                       the val holdout carved out below)
  val.parquet         held-out validation split, passed to fit() as
                       `tuning_data` so AutoGluon's internal validation is
                       OUR val split, not an opaque auto-holdout — this is
                       what makes a genuine "val" entry in ModelResult
                       possible at all (AutoGluon does not expose which
                       rows it validated on if you let it choose).

read_leaderboard() locates train_fit.parquet/val.parquet from
predictor_path's parent directory by this convention (spec's function
signature only takes predictor_path + test_path).
"""

import os
import random
import uuid

import numpy as np
import pandas as pd
import psutil
from autogluon.tabular import TabularPredictor
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from config import AUTOGLUON_MODEL_KEY, AUTOGLUON_METRIC, AUTOGLUON_PROBLEM_TYPE, ENSEMBLE_SENTINEL, FAMILY_METRICS

OUTPUT_ROOT = "outputs/ml_engineer"
VAL_SIZE = 0.2

# family -> AutoGluon-native secondary metric names (must be valid autogluon.core.metrics
# entries; anything spec asks for that AutoGluon doesn't support natively as a scalar
# — per_class_recall (array, not scalar), rmsle (not in this AutoGluon version) — is
# computed manually below instead of listed here).
AUTOGLUON_SECONDARY_METRICS = {
    "binary": ["roc_auc", "recall", "precision", "mcc"],
    "multiclass": ["balanced_accuracy"],
    "regression": ["r2", "mape"],
}


def profile_hardware() -> dict:
    """Detect CPU cores/RAM. GPU detection is best-effort: only meaningful if
    torch is installed; this repo's V1 stack is CPU-only tree/linear models
    (no torch dependency — see PROGRESS.md deviations log), so num_gpus is 0
    unless torch happens to be present and reports a device.
    """
    num_cpus = psutil.cpu_count(logical=True) or 1
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except ImportError:
        num_gpus = 0

    if num_gpus > 0:
        tier = "gpu"
    elif num_cpus >= 8:
        tier = "large_cpu"
    elif num_cpus >= 4:
        tier = "medium_cpu"
    else:
        tier = "small_cpu"

    return {"num_gpus": num_gpus, "num_cpus": num_cpus, "ram_gb": ram_gb, "tier": tier}


def _hyperparameters_from_candidates(candidate_models: list[dict], excluded_keys: set[str] | None = None) -> dict:
    """Build the AutoGluon hyperparameters restriction. excluded_keys is applied
    AFTER the candidate-list/fallback-to-all-registry logic in both branches, so a
    retrain exclusion (Critic flagged this specific learner for overfitting) can
    never be silently undone by falling back to the unrestricted registry.
    """
    names = [c["name"] for c in candidate_models if c.get("name") in AUTOGLUON_MODEL_KEY]
    if not names:
        names = list(AUTOGLUON_MODEL_KEY.keys())
    keys = {AUTOGLUON_MODEL_KEY[n] for n in names}
    if excluded_keys:
        keys -= set(excluded_keys)
    if not keys:
        raise ValueError(f"no candidate learners remain after excluding {excluded_keys} — all options exhausted")
    return {k: {} for k in sorted(keys)}


def fit_predictor(
    cleaned_path: str,
    feature_list: list[str],
    target: str,
    family: str,
    candidate_models: list[dict],
    budget: dict,
    seed: int,
    hw: dict,
    run_id: str | None = None,
    excluded_keys: set[str] | None = None,
) -> dict:
    """Fit ONE AutoGluon TabularPredictor under budget['wall_clock_s'].

    excluded_keys (not in spec's signature — graph.py retrain-loop bookkeeping,
    see PROGRESS.md deviations log): AutoGluon learner keys to exclude from
    consideration, e.g. because Critic flagged that specific learner for
    overfitting on a prior attempt.

    Returns {"predictor_path": str | None, "fit_summary": {...}, "error": str | None}.
    predictor_path is None on failure — caller must check `error` before
    proceeding to read_leaderboard.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_full = pd.read_parquet(os.path.join(cleaned_path, "train.parquet"))
    train_full = train_full[feature_list + [target]]

    stratify = train_full[target] if family in ("binary", "multiclass") else None
    train_fit, val = train_test_split(train_full, test_size=VAL_SIZE, random_state=seed, stratify=stratify)

    problem_type = AUTOGLUON_PROBLEM_TYPE[family]
    primary_metric = FAMILY_METRICS[family]["primary_metric"]
    eval_metric = AUTOGLUON_METRIC.get(primary_metric, primary_metric)

    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    predictor_path = os.path.join(run_dir, "predictor")

    train_fit_path = os.path.join(run_dir, "train_fit.parquet")
    val_path = os.path.join(run_dir, "val.parquet")
    train_fit.to_parquet(train_fit_path, index=False)
    val.to_parquet(val_path, index=False)

    error = None
    fit_summary: dict = {
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "time_limit_s": budget.get("wall_clock_s"),
    }
    try:
        # ENSEMBLE_SENTINEL isn't a hyperparameters key — it means Critic flagged
        # AutoGluon's own auto-built WeightedEnsemble for overfitting, which can't be
        # excluded via `hyperparameters` (it's not a single learner). Handled via
        # fit()'s own fit_weighted_ensemble flag instead, so a retry after that flag
        # is genuinely different, not silently identical.
        excluded_keys = excluded_keys or set()
        disable_ensemble = ENSEMBLE_SENTINEL in excluded_keys
        learner_excluded_keys = excluded_keys - {ENSEMBLE_SENTINEL}

        hyperparameters = _hyperparameters_from_candidates(candidate_models, learner_excluded_keys)
        fit_summary["hyperparameters_used"] = hyperparameters
        fit_summary["fit_weighted_ensemble"] = not disable_ensemble

        predictor = TabularPredictor(
            label=target,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=predictor_path,
            verbosity=1,
        )
        predictor.fit(
            train_data=train_fit,
            tuning_data=val,
            time_limit=budget.get("wall_clock_s"),
            hyperparameters=hyperparameters,
            num_cpus=hw.get("num_cpus", "auto"),
            num_gpus=hw.get("num_gpus", 0),
            fit_weighted_ensemble=not disable_ensemble,
        )
        fit_summary["model_names"] = predictor.model_names()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        predictor_path = None

    return {"predictor_path": predictor_path, "fit_summary": fit_summary, "error": error}


def manual_secondary_metrics(family: str, y_true, y_pred, class_labels: list | None) -> dict:
    extra = {}
    if family == "multiclass" and class_labels:
        per_class = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
        extra["per_class_recall"] = {str(c): round(float(r), 4) for c, r in zip(class_labels, per_class)}
    if family == "regression":
        y_true_arr, y_pred_arr = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        if np.all(y_true_arr > -1) and np.all(y_pred_arr > -1):
            rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_arr) - np.log1p(y_true_arr)) ** 2)))
            extra["rmsle"] = round(rmsle, 4)
        else:
            extra["rmsle"] = None  # undefined when values <= -1 present
    return extra


# AutoGluon's Scorer always orients "higher score = better", which means genuinely
# lower-is-better error metrics (rmse, mae, mape, ...) come back NEGATED from
# leaderboard()/score(). Verified directly against autogluon.core.metrics.REGRESSION_METRICS:
# calling the rmse scorer on a toy array returns -0.5477, not +0.5477. r2/pearsonr/spearmanr
# are natively higher-is-better and are NOT negated. Un-negate by our (post-rename) name here
# so downstream consumers (Critic, Experiment Manager) see the real metric.
NEGATED_REGRESSION_METRICS = {"rmse", "mae", "mape", "rmsle", "mse"}


def split_metrics(predictor: TabularPredictor, data: pd.DataFrame, family: str, primary_metric: str) -> pd.DataFrame:
    extra = AUTOGLUON_SECONDARY_METRICS.get(family, [])
    lb = predictor.leaderboard(data=data, extra_metrics=extra, silent=True)
    lb = lb.set_index("model")
    rename = {"score_test": primary_metric}
    if "balanced_accuracy" in lb.columns and family == "multiclass":
        rename["balanced_accuracy"] = "balanced_acc"
    if "mean_absolute_percentage_error" in lb.columns:
        rename["mean_absolute_percentage_error"] = "mape"
    lb = lb.rename(columns=rename)

    if family == "regression":
        for col in NEGATED_REGRESSION_METRICS:
            if col in lb.columns:
                lb[col] = lb[col].abs()

    return lb


def _model_size_mb(predictor_path: str, model_name: str) -> float:
    model_dir = os.path.join(predictor_path, "models", model_name)
    if not os.path.isdir(model_dir):
        return 0.0
    total = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(model_dir) for f in files)
    return round(total / (1024 ** 2), 4)


def read_leaderboard(predictor_path: str, test_path: str, family: str) -> list[dict]:
    """Load predictor, score every leaderboard model on train_fit/val/test,
    and build one ModelResult dict per row. Replaces per-model training.
    """
    predictor = TabularPredictor.load(predictor_path)
    run_dir = os.path.dirname(predictor_path)
    train_fit = pd.read_parquet(os.path.join(run_dir, "train_fit.parquet"))
    val = pd.read_parquet(os.path.join(run_dir, "val.parquet"))
    test = pd.read_parquet(test_path)

    target = predictor.label
    primary_metric = FAMILY_METRICS[family]["primary_metric"]

    lb_train = split_metrics(predictor, train_fit, family, primary_metric)
    lb_val = split_metrics(predictor, val, family, primary_metric)
    lb_test = split_metrics(predictor, test, family, primary_metric)

    metric_cols = [c for c in lb_test.columns if c not in (
        "score_val", "eval_metric", "pred_time_test", "pred_time_val", "fit_time",
        "pred_time_test_marginal", "pred_time_val_marginal", "fit_time_marginal",
        "stack_level", "can_infer", "fit_order",
    )]

    class_labels = list(predictor.class_labels) if family in ("binary", "multiclass") else None
    n_test = len(test)

    results = []
    for model_name in lb_test.index:
        y_pred_test = predictor.predict(test.drop(columns=[target]), model=model_name)
        manual_extras = manual_secondary_metrics(family, test[target], y_pred_test, class_labels)

        test_metrics = {k: round(float(v), 4) for k, v in lb_test.loc[model_name, metric_cols].items() if pd.notna(v)}
        test_metrics.update(manual_extras)
        if family in ("binary", "multiclass"):
            cm = confusion_matrix(test[target], y_pred_test, labels=class_labels)
            test_metrics["confusion_matrix"] = cm.tolist()

        train_metrics = {k: round(float(v), 4) for k, v in lb_train.loc[model_name, metric_cols].items() if pd.notna(v)}
        val_metrics = {k: round(float(v), 4) for k, v in lb_val.loc[model_name, metric_cols].items() if pd.notna(v)}

        try:
            fi = predictor.feature_importance(
                test, model=model_name, subsample_size=min(200, n_test), num_shuffle_sets=2, silent=True
            )
            feature_importance = {k: round(float(v), 4) for k, v in fi["importance"].items()}
        except Exception:
            feature_importance = {}

        pred_time_test = float(lb_test.loc[model_name, "pred_time_test"])
        results.append({
            "name": model_name,
            "model_path": predictor_path,
            "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
            "timings": {
                "train_s": round(float(lb_train.loc[model_name, "fit_time"]), 4),
                "infer_ms_per_row": round((pred_time_test / n_test) * 1000, 4) if n_test else 0.0,
                "size_mb": _model_size_mb(predictor_path, model_name),
            },
            "feature_importance": feature_importance,
            "hpo_trials": 0,  # AutoGluon default configs, no hyperparameter_tune_kwargs in V1
        })

    return results
