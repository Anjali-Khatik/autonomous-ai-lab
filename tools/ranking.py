"""aggregate_and_rank() + log_mlflow() for the Experiment Manager, and
success_criteria_check() reused by Chief Scientist.
"""

import os

import mlflow

from config import METRIC_DIRECTION

MLFLOW_DB_PATH = "outputs/mlflow.db"
MLFLOW_EXPERIMENT = "autonomous-ai-lab"


def aggregate_and_rank(trained_models: list[dict], primary_metric: str) -> dict:
    """Build a comparison table from real test metrics, rank by primary_metric.

    Returns:
        {
          "ranked_models": [name, ...],   # best first
          "table": [{"rank": int, "model": str, primary_metric: float,
                     "infer_ms": float, "size_mb": float}, ...]
        }
    """
    direction = METRIC_DIRECTION.get(primary_metric, "max")
    reverse = direction == "max"

    scored = [
        (m["name"], m["metrics"]["test"].get(primary_metric), m["timings"]["infer_ms_per_row"], m["timings"]["size_mb"])
        for m in trained_models
    ]
    scored = [s for s in scored if s[1] is not None]
    scored.sort(key=lambda s: s[1], reverse=reverse)

    ranked_models = [name for name, *_ in scored]
    table = [
        {"rank": i + 1, "model": name, primary_metric: round(score, 4), "infer_ms": round(infer_ms, 4), "size_mb": round(size_mb, 4)}
        for i, (name, score, infer_ms, size_mb) in enumerate(scored)
    ]

    return {"ranked_models": ranked_models, "table": table}


def log_mlflow(run_id: str, trained_models: list[dict], ranked: list[str]) -> None:
    """Log this run's models/metrics/artifacts to MLflow (local file store
    under outputs/mlruns — never sent anywhere external).
    """
    os.makedirs("outputs", exist_ok=True)
    # mlflow 3.x deprecated the plain filesystem store (raises unless
    # MLFLOW_ALLOW_FILE_STORE=true is set) — use a local sqlite backend instead.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("ranked_models", ranked)
        for m in trained_models:
            for split, metrics in m["metrics"].items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{m['name']}.{split}.{k}", v)
            for k, v in m["timings"].items():
                mlflow.log_metric(f"{m['name']}.timing.{k}", v)


def success_criteria_check(metrics: dict, plan: dict, primary_metric: str) -> dict:
    """Pass/fail each success criterion in plan['success_criteria']. Returns scorecard.

    `primary_metric` isn't in the spec's 2-arg signature — added because
    plan['success_criteria']['primary_metric_target'] is just a number; you
    need to know which key in `metrics` it refers to. Same class of minimal
    signature addition as fit_predictor/build_clean_pipeline's run_id param.
    """
    success_criteria = (plan or {}).get("success_criteria", {}) or {}
    direction = METRIC_DIRECTION.get(primary_metric, "max")

    scorecard = {}
    for criterion, target in success_criteria.items():
        if criterion == "primary_metric_target":
            actual = metrics.get(primary_metric)
            key = "primary_metric_target_met"
        else:
            actual = metrics.get(criterion)
            key = f"{criterion}_met"

        if actual is None:
            scorecard[key] = None  # can't evaluate — metric not present
        elif criterion == "primary_metric_target":
            scorecard[key] = actual >= target if direction == "max" else actual <= target
        else:
            scorecard[key] = actual == target

    return scorecard
