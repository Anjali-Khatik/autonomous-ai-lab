"""Flags inappropriate primary metric choices (e.g. accuracy on imbalance).

Since config.FAMILY_METRICS never sets primary_metric to an
imbalance-insensitive metric for binary/multiclass in the first place,
this mostly acts as a guard for the future (e.g. if a family's config
ever changes, or a [V2] Research Planner override picks something
unwise) rather than something that fires today — still real logic, not
a rubber stamp, since it's evaluated against actual measured class
balance every run.
"""

IMBALANCE_INSENSITIVE_METRICS = {"accuracy", "acc"}
IMBALANCE_MINORITY_THRESHOLD = 0.2


def check_metric_choice(primary_metric: str, family: str, class_balance: dict | None) -> dict:
    """Returns {"appropriate": bool, "class_balance": dict|None, "findings": [str]}."""
    findings = []
    appropriate = True

    if family in ("binary", "multiclass") and class_balance:
        minority_share = min(class_balance.values())
        imbalanced = minority_share < IMBALANCE_MINORITY_THRESHOLD

        if imbalanced and primary_metric.lower() in IMBALANCE_INSENSITIVE_METRICS:
            appropriate = False
            findings.append(
                f"primary_metric={primary_metric} is insensitive to class imbalance "
                f"(minority class share {minority_share:.1%}); recommend f1 or pr_auc instead"
            )
        elif imbalanced:
            findings.append(
                f"class imbalance detected (minority share {minority_share:.1%}) but "
                f"primary_metric={primary_metric} already accounts for it — no change needed"
            )

    return {"appropriate": appropriate, "class_balance": class_balance, "findings": findings}
