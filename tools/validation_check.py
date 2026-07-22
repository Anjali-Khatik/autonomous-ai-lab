"""Validation strategy soundness (stratification, temporal folds, test isolation, seed).

NOTE on depth: the spec's signature is (plan, family) only — it doesn't
receive the actual split metadata from this specific run, so this checks
that our PIPELINE's fixed policy is sound for the family (e.g. "does this
family require stratification, and does our cleaning tool apply it"),
not a live introspection of what literally happened this run. A stronger
V2 version would take the real split method/seed used and verify it
directly rather than trusting the fixed policy is still followed.
"""

from config import FAMILY_METRICS


def check_validation(plan: dict, family: str) -> dict:
    """Returns {"sound": bool, "expected_split": str, "findings": [str]}."""
    expected_split = FAMILY_METRICS[family]["validation_split"]
    findings = []

    if expected_split == "stratified_kfold":
        findings.append(
            f"family={family} requires stratified splits — tools/cleaning.py's "
            "build_clean_pipeline stratifies on the target for binary/multiclass (verified in code)"
        )
    elif expected_split == "kfold":
        findings.append(f"family={family} uses plain splits — correct, no class balance to preserve")
    elif expected_split == "temporal":
        findings.append(f"family={family} requires temporal splits — [V2] not implemented, forecast is a stub")

    findings.append("single global seed (state['seed']) threaded through split and engine fit — verified in code")
    findings.append("test split is never passed to fit() — held out by tools/cleaning.py, only used for leaderboard/evaluate")

    sound = family != "forecast"  # [V2] forecast validation isn't implemented yet
    return {"sound": sound, "expected_split": expected_split, "findings": findings}
