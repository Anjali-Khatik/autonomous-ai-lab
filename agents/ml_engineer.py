"""Agent 3 — ML Engineer. [V1: core] Wraps one AutoGluon TabularPredictor.fit() + leaderboard().

Execution agent only — no LLM. Fits ONE predictor under the compute
budget, then reads honest per-model metrics from its leaderboard on the
untouched test set Data Scientist produced.
"""

import os

from state import LabState
from tools.engine_wrapper import fit_predictor, profile_hardware, read_leaderboard
from tools.serialize import register_predictor


def ml_engineer_node(state: LabState) -> dict:
    """LangGraph node. Reads candidate_models/cleaned_data_path/feature_list/
    problem_family/compute_budget/seed from state, returns trained_models.

    On a hard fit failure, returns state["error"] instead of crashing the
    graph (spec rule: "on failure record the error, don't crash the run").
    """
    cleaned_path = state["cleaned_data_path"]
    feature_list = state["feature_list"]
    target = state["target_column"]
    family = state["problem_family"]
    candidate_models = state["candidate_models"]
    budget = state["compute_budget"]
    seed = state["seed"]
    run_id = state.get("run_id")
    excluded_keys = set(state.get("excluded_learner_keys", []))

    hw = profile_hardware()

    fit_result = fit_predictor(
        cleaned_path=cleaned_path,
        feature_list=feature_list,
        target=target,
        family=family,
        candidate_models=candidate_models,
        budget=budget,
        seed=seed,
        hw=hw,
        run_id=run_id,
        excluded_keys=excluded_keys,
    )

    if fit_result["error"] is not None:
        return {"trained_models": [], "error": f"ml_engineer: {fit_result['error']}"}

    predictor_path = fit_result["predictor_path"]
    test_path = os.path.join(cleaned_path, "test.parquet")

    trained_models = read_leaderboard(predictor_path, test_path, family)

    register_predictor(predictor_path, {
        "run_id": run_id,
        "family": family,
        "fit_summary": fit_result["fit_summary"],
        "hardware": hw,
    })

    return {"trained_models": trained_models}
