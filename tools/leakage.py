"""KS distribution tests + suspicious-feature scan + temporal integrity.

Generalized: no dataset-specific column names. Runs on whatever numeric
feature columns the cleaning pipeline produced. Identifier columns are
already dropped upstream by tools/cleaning.py, so this focuses on two
independent signals: distribution shift between the leakage-safe train
and untouched test splits, and any single feature dominating importance
in a way that smells like a leaked target proxy.
"""

import os
import re

import pandas as pd
from scipy.stats import ks_2samp

KS_PVALUE_THRESHOLD = 0.01
DOMINANT_IMPORTANCE_SHARE = 0.5

# Spec §3 Agent 5 asks this check to flag "suspicious high-importance ID/date features"
# specifically — not any dominant feature. A single feature dominating importance is
# expected and legitimate when few features remain after selection (e.g. Iris's
# PetalLengthCm genuinely separates species almost perfectly with only 3 features left —
# that's real signal, not leakage). Only escalate to "high" severity (which forces a
# reject) when the dominant feature's name also looks like an identifier/date/timestamp;
# otherwise it's downgraded to a "moderate" note for a human/LLM to weigh, not an
# automatic red flag. Caught via real testing: the naive share>0.5 rule falsely rejected
# a legitimately good Iris model — see PROGRESS.md deviations log.
ID_DATE_PATTERN = re.compile(r"(^|_)(id|date|time|timestamp)($|_)", re.IGNORECASE)


def check_leakage(cleaned_path: str, feature_importance: dict, family: str) -> dict:
    """Returns {"detected": bool, "findings": [...]}."""
    train_df = pd.read_parquet(os.path.join(cleaned_path, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(cleaned_path, "test.parquet"))

    findings = []

    shared_numeric_cols = [
        c for c in train_df.columns
        if c in test_df.columns and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    for col in shared_numeric_cols:
        train_vals, test_vals = train_df[col].dropna(), test_df[col].dropna()
        if len(train_vals) < 2 or len(test_vals) < 2:
            continue
        stat, p_value = ks_2samp(train_vals, test_vals)
        if p_value < KS_PVALUE_THRESHOLD:
            findings.append({
                "type": "distribution_shift",
                "feature": col,
                "ks_stat": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
            })

    if feature_importance:
        total = sum(abs(v) for v in feature_importance.values()) or 1.0
        for feat, imp in feature_importance.items():
            share = abs(imp) / total
            if share > DOMINANT_IMPORTANCE_SHARE:
                looks_like_id_or_date = bool(ID_DATE_PATTERN.search(feat))
                findings.append({
                    "type": "suspicious_high_importance",
                    "feature": feat,
                    "importance_share": round(share, 4),
                    "looks_like_id_or_date": looks_like_id_or_date,
                })

    if family == "forecast":
        # [V2] temporal integrity check not built — forecast family is a stub in V1.
        findings.append({"type": "temporal_integrity", "status": "not_implemented", "note": "[V2] forecast family"})

    detected = any(f["type"] in ("distribution_shift", "suspicious_high_importance") for f in findings)
    return {"detected": detected, "findings": findings}
