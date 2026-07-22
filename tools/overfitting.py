"""Train/val/test gap analysis vs family-calibrated thresholds.

Generalized across families via one important scale decision: RMSE/MAE
are unbounded and dataset-scale-dependent (a housing dataset priced in
millions needs a wildly different absolute threshold than one priced in
tens), so the bounded 0.03/0.05/0.10 thresholds below can't apply to them
directly without per-dataset calibration this V1 doesn't have. r2 is
bounded like f1/macro_f1 (~0-1, higher-is-better), so for regression we
use r2 for the gap when it's available in the metrics, falling back to a
relative (percentage) gap on the primary metric otherwise.
"""

from config import FAMILY_METRICS, METRIC_DIRECTION, OVERFIT_THRESHOLDS


def _severity(gap: float) -> str:
    if gap <= OVERFIT_THRESHOLDS["none"]:
        return "none"
    if gap <= OVERFIT_THRESHOLDS["mild"]:
        return "mild"
    if gap <= OVERFIT_THRESHOLDS["moderate"]:
        return "moderate"
    return "severe"


def check_overfitting(metrics: dict, family: str) -> dict:
    """Gap analysis vs thresholds (None<0.03 / Mild 0.03-0.05 / Moderate 0.05-0.10 / Severe>0.10).

    Returns:
        {"severity": str, "gaps": {"train_vs_val": float|None, "train_vs_test": float|None},
         "detected": bool, "gap_metric": str, "train": float|None, "val": float|None, "test": float|None}
    """
    primary_metric = FAMILY_METRICS[family]["primary_metric"]

    if family == "regression" and "r2" in metrics.get("train", {}):
        gap_metric = "r2"
        relative = False
    else:
        gap_metric = primary_metric
        relative = METRIC_DIRECTION.get(primary_metric, "max") == "min"

    train_v = metrics.get("train", {}).get(gap_metric)
    val_v = metrics.get("val", {}).get(gap_metric)
    test_v = metrics.get("test", {}).get(gap_metric)

    def gap(base, other):
        if base is None or other is None:
            return None
        if relative:
            denom = abs(base) if abs(base) > 1e-9 else 1e-9
            return round((other - base) / denom, 4)
        return round(base - other, 4)

    gaps = {"train_vs_val": gap(train_v, val_v), "train_vs_test": gap(train_v, test_v)}
    numeric_gaps = [g for g in gaps.values() if g is not None]
    worst_gap = max(numeric_gaps) if numeric_gaps else 0.0
    severity = _severity(worst_gap)

    return {
        "severity": severity,
        "gaps": gaps,
        "detected": severity in ("moderate", "severe"),
        "gap_metric": gap_metric,
        "train": train_v,
        "val": val_v,
        "test": test_v,
    }
