"""Acceptance test (spec §6): Critic reject -> retrain loop (headline test).

"Feed a deliberately overfit model -> Critic verdict reject -> graph
routes to ml_engineer -> retrain_count increments -> loop terminates at
MAX_RETRAINS." Uses housing (regression), which every prior manual test
run has shown genuinely, repeatedly overfits on its 545 rows — a real
case, not an artificially forced one. Runs the actual compiled graph via
invoke(), not manual node chaining. Slow (multiple real AutoGluon fits +
LLM calls, ~1-3 minutes).
"""

import pytest

from config import MAX_RETRAINS
from graph import build_graph


@pytest.mark.slow
@pytest.mark.llm
def test_retrain_loop_fires_and_terminates():
    graph = build_graph(hitl=False)
    initial_state = {
        "user_objective": "Predict the sale price of a house from its features.",
        "dataset_path": "dataset/Housing.csv",
        "target_column": "price",
        "constraints": {},
        "seed": 42,
        "run_id": "pytest_retrain_loop",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
        "retrain_count": 0,
        "excluded_models": [],
        "excluded_learner_keys": [],
        "business_params": None,
        "hitl": False,
    }

    final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    # retrain_count must never exceed the cap — this is the safety-critical
    # invariant the whole loop exists to enforce.
    assert final_state.get("retrain_count", 0) <= MAX_RETRAINS

    # The loop must reach a well-defined terminal outcome, not hang or crash
    # uncaught: either a real decision (quality gate eventually passed) or a
    # clean, explained failure (every candidate exhausted the quality gate).
    if final_state.get("decision"):
        assert final_state["decision"]["winner"]
        assert final_state["decision"]["recommendation"] in ("GO", "GO-WITH-CONDITIONS", "NO-GO")
    else:
        assert final_state.get("error"), "pipeline ended with neither a decision nor an explained error"

    # The mechanism itself must have actually engaged, not just terminated
    # trivially on the very first pass.
    engaged = final_state.get("retrain_count", 0) > 0 or bool(final_state.get("excluded_models"))
    assert engaged, "retrain loop never fired — housing's known overfit case should have triggered at least one reject"
