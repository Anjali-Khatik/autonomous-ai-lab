"""Manual validation of Research Planner, standalone AND as the first full
end-to-end pipeline run (raw user_objective + dataset_path -> decision)
across all 6 other real agents — the actual V1 exit-criteria test from
PROGRESS.md, run for the first time now that every agent exists.

Real LLM calls throughout (not stubbed) — this is the definitive
full-pipeline validation, worth the extra time/cost.
"""

import json
import sys

sys.path.insert(0, ".")

from agents.business_analyst import business_analyst_node
from agents.chief_scientist import chief_scientist_node
from agents.critic import critic_node
from agents.data_scientist import data_scientist_node
from agents.experiment_manager import experiment_manager_node
from agents.ml_engineer import ml_engineer_node
from agents.research_planner import research_planner_node
from tools.task_detection import detect_problem_family

CASES = [
    {"name": "loan", "dataset_path": "dataset/loan_dataset.csv", "target_column": "Loan_Status",
     "user_objective": "Predict whether a loan application will be approved."},
    {"name": "iris", "dataset_path": "dataset/Iris.csv", "target_column": "Species",
     "user_objective": "Classify the species of an iris flower from its measurements."},
    {"name": "housing", "dataset_path": "dataset/Housing.csv", "target_column": "price",
     "user_objective": "Predict the sale price of a house from its features."},
]


def check_detection_standalone():
    print(f"\n{'=' * 70}\nStandalone detect_problem_family checks\n{'=' * 70}")
    expected = {"loan": "binary", "iris": "multiclass", "housing": "regression"}
    for case in CASES:
        result = detect_problem_family(case["dataset_path"], case["target_column"])
        status = "OK" if result["family"] == expected[case["name"]] else "FAIL"
        print(f"  {status}: {case['name']} -> {result['family']} (confidence={result['confidence']}, expected={expected[case['name']]})")
        assert result["family"] == expected[case["name"]]

    no_target = detect_problem_family(CASES[0]["dataset_path"], None)
    print(f"  OK: no target_column -> family={no_target['family']} (must be None, hard stop)")
    assert no_target["family"] is None


def run_full_pipeline(case: dict, success_metric_target: float | None = None) -> dict:
    print(f"\n{'=' * 70}\nFull pipeline (raw input -> decision): {case['name']}\n{'=' * 70}")

    constraints = {}
    if success_metric_target is not None:
        constraints["success_metric_target"] = success_metric_target

    state = {
        "user_objective": case["user_objective"],
        "dataset_path": case["dataset_path"],
        "target_column": case["target_column"],
        "constraints": constraints,
        "seed": 42,
        "run_id": f"rptest_{case['name']}",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
        "retrain_count": 0,
        "business_params": None,
        "hitl": False,
    }

    state.update(research_planner_node(state))
    print(f"problem_family={state.get('problem_family')} primary_metric={state.get('primary_metric')}")
    print(f"plan: {json.dumps(state.get('plan'), indent=2)}")
    if state.get("error"):
        print("ERROR:", state["error"])
        return state

    state.update(data_scientist_node(state))
    state.update(ml_engineer_node(state))
    if state.get("error"):
        print("ERROR:", state["error"])
        return state
    state.update(experiment_manager_node(state))
    state.update(critic_node(state))
    if state["critic_verdict"] in ("approve", "conditional"):
        state.update(business_analyst_node(state))
        state.update(chief_scientist_node(state))
        print(f"\ncritic_verdict={state['critic_verdict']}")
        print("decision:", json.dumps(state["decision"], indent=2))
    else:
        print(f"\ncritic_verdict={state['critic_verdict']} — graph would route back to ml_engineer/experiment_manager, not forward to business_analyst/chief_scientist")

    return state


if __name__ == "__main__":
    check_detection_standalone()

    for case in CASES:
        run_full_pipeline(case)

    print(f"\n\n{'#' * 70}\nExtra: user-supplied success_metric_target flows through to Chief Scientist's scorecard\n{'#' * 70}")
    run_full_pipeline(CASES[1], success_metric_target=0.5)  # deliberately low bar iris should clear
