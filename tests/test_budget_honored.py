"""Acceptance test (spec §6): budget honored.

"A tiny wall_clock_s still returns a valid best-so-far ModelResult."
Real AutoGluon fit (tiny budget keeps it fast — no LLM involved, this is
an execution agent per spec).
"""

import os

import pandas as pd

from tools.cleaning import build_clean_pipeline
from tools.engine_wrapper import fit_predictor, profile_hardware, read_leaderboard

TINY_BUDGET = {"wall_clock_s": 5, "max_trials": None, "cost_cap_usd": None}


def test_tiny_budget_still_returns_valid_model_result(tmp_path):
    run_id = "budget_test"
    clean_result = build_clean_pipeline(
        df_path="dataset/Iris.csv", target="Species", family="multiclass", seed=42, run_id=run_id,
    )

    train_df = pd.read_parquet(os.path.join(clean_result["cleaned_data_path"], "train.parquet"))
    feature_list = [c for c in train_df.columns if c != "Species"]

    hw = profile_hardware()
    fit_result = fit_predictor(
        cleaned_path=clean_result["cleaned_data_path"],
        feature_list=feature_list,
        target="Species",
        family="multiclass",
        candidate_models=[],  # empty -> falls back to full allowed registry
        budget=TINY_BUDGET,
        seed=42,
        hw=hw,
        run_id=run_id,
    )

    assert fit_result["error"] is None, f"tiny budget caused a hard failure instead of a best-so-far result: {fit_result['error']}"
    assert fit_result["predictor_path"] is not None

    test_path = os.path.join(clean_result["cleaned_data_path"], "test.parquet")
    trained_models = read_leaderboard(fit_result["predictor_path"], test_path, "multiclass")

    assert len(trained_models) >= 1
    top = trained_models[0]
    assert top["metrics"]["test"].get("macro_f1") is not None
    assert 0.0 <= top["metrics"]["test"]["macro_f1"] <= 1.0
