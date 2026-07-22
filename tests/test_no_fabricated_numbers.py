"""Acceptance test (spec §6): no fabricated numbers.

"With business_params = None, the Business Analyst output contains
impact: null and no dollar figures anywhere." Also verifies the same
holds even when business_params IS supplied, since compute_impact isn't
implemented in V1 (see agents/business_analyst.py docstring) — supplying
params must never cause a number to be invented just because it was asked
for. Makes a real Fireworks call (temperature=0, but not perfectly
deterministic — see PROGRESS.md notes on Critic's LLM variability).
"""

import re

import pytest

from agents.business_analyst import business_analyst_node

DOLLAR_PATTERN = re.compile(r"\$\s?\d|USD\s?\d|\d\s?USD", re.IGNORECASE)

FAKE_STATE = {
    "ranked_models": ["XGBoost"],
    "trained_models": [{
        "name": "XGBoost",
        "model_path": "unused",
        "metrics": {"train": {"f1": 0.9}, "val": {"f1": 0.88}, "test": {"f1": 0.887, "roc_auc": 0.94, "recall": 0.87, "precision": 0.9}},
        "timings": {"train_s": 10.0, "infer_ms_per_row": 0.02, "size_mb": 5.0},
        "feature_importance": {"tenure": 0.31, "monthly_charges": 0.28, "contract_type": 0.15},
        "hpo_trials": 0,
    }],
}


@pytest.mark.llm
def test_no_fabricated_numbers_without_business_params():
    state = {**FAKE_STATE, "business_params": None}
    result = business_analyst_node(state)

    assert result["narrative"]["impact"] is None
    summary = result["narrative"]["summary"]
    assert not DOLLAR_PATTERN.search(summary), f"summary invented a dollar figure with no business_params given: {summary!r}"


@pytest.mark.llm
def test_no_fabricated_numbers_even_with_business_params_supplied():
    """compute_impact isn't implemented in V1 (tools/impact.py is a [V2]
    stub) — supplying real params must not cause the LLM to just make up
    the arithmetic itself instead.
    """
    state = {**FAKE_STATE, "business_params": {"cost_per_false_negative_usd": 500, "confirmation_fraction": 0.05}}
    result = business_analyst_node(state)

    assert result["narrative"]["impact"] is None
    summary = result["narrative"]["summary"]
    assert not DOLLAR_PATTERN.search(summary), f"summary invented a dollar figure despite compute_impact not being implemented: {summary!r}"
