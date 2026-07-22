"""Manual validation of the compiled graph via real invoke() calls — not
manual node chaining like the earlier per-agent tests. This is the actual
headline milestone: does the Critic reject -> retry -> (possibly) refer_alt
loop really fire through the compiled LangGraph, not just in isolated
verdict-logic checks.

housing (regression) is used for the retry-loop case since every prior
test run has shown it genuinely, repeatedly overfits on this small
545-row dataset — a real case, not artificially forced.
"""

import json
import sys

sys.path.insert(0, ".")

from graph import build_graph

CASES = [
    {"name": "loan", "dataset_path": "dataset/loan_dataset.csv", "target_column": "Loan_Status",
     "user_objective": "Predict whether a loan application will be approved."},
    {"name": "iris", "dataset_path": "dataset/Iris.csv", "target_column": "Species",
     "user_objective": "Classify the species of an iris flower from its measurements."},
    {"name": "housing", "dataset_path": "dataset/Housing.csv", "target_column": "price",
     "user_objective": "Predict the sale price of a house from its features."},
]


def initial_state(case: dict) -> dict:
    return {
        "user_objective": case["user_objective"],
        "dataset_path": case["dataset_path"],
        "target_column": case["target_column"],
        "constraints": {},
        "seed": 42,
        "run_id": f"graphtest_{case['name']}",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
        "retrain_count": 0,
        "excluded_models": [],
        "excluded_learner_keys": [],
        "business_params": None,
        "hitl": False,
    }


def run_case(case: dict):
    print(f"\n{'=' * 70}\nGraph invoke: {case['name']}\n{'=' * 70}")
    graph = build_graph(hitl=False)
    final_state = graph.invoke(initial_state(case), config={"recursion_limit": 50})

    print("problem_family:", final_state.get("problem_family"))
    print("retrain_count:", final_state.get("retrain_count"))
    print("excluded_learner_keys:", final_state.get("excluded_learner_keys"))
    print("excluded_models:", final_state.get("excluded_models"))
    print("critic_verdict:", final_state.get("critic_verdict"))
    print("error:", final_state.get("error"))
    if final_state.get("decision"):
        print("decision:", json.dumps(final_state["decision"], indent=2))
    return final_state


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
