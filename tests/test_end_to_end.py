"""Acceptance test (spec §6): end-to-end.

"A small classification CSV runs Planner->...->Chief Scientist and
produces a decision with a winner and GO/NO-GO." Uses the actual compiled
graph via invoke() (not manual chaining), raw user_objective + dataset_path
only — Research Planner detects the family itself. Iris is used because
every prior manual run has reliably reached Chief Scientist (approve or
conditional, never reject) on this dataset. Slow (real AutoGluon fit +
multiple LLM calls).
"""

import pytest

from graph import build_graph


@pytest.mark.slow
@pytest.mark.llm
def test_end_to_end_classification_produces_a_decision():
    graph = build_graph(hitl=False)
    initial_state = {
        "user_objective": "Classify the species of an iris flower from its measurements.",
        "dataset_path": "dataset/Iris.csv",
        "target_column": "Species",
        "constraints": {},
        "seed": 42,
        "run_id": "pytest_end_to_end",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
        "retrain_count": 0,
        "excluded_models": [],
        "excluded_learner_keys": [],
        "business_params": None,
        "hitl": False,
    }

    final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    assert final_state.get("error") is None, f"pipeline hit a hard error: {final_state.get('error')}"
    assert final_state["problem_family"] == "multiclass"

    decision = final_state.get("decision")
    assert decision is not None, "no decision produced — Chief Scientist was never reached"
    assert decision["winner"]
    assert decision["recommendation"] in ("GO", "GO-WITH-CONDITIONS", "NO-GO")
    assert decision["confidence"] in ("high", "medium", "low")
    assert isinstance(decision["rationale"], list) and len(decision["rationale"]) > 0
