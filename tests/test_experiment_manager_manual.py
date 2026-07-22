"""Manual end-to-end validation of Experiment Manager against ML Engineer's
REAL output (spec build order: "test against real upstream output").
"""

import json
import sys

sys.path.insert(0, ".")

from agents.data_scientist import data_scientist_node
from agents.experiment_manager import experiment_manager_node
from agents.ml_engineer import ml_engineer_node
from config import FAMILY_METRICS

CASES = [
    {"name": "loan (binary)", "dataset_path": "dataset/loan_dataset.csv", "target_column": "Loan_Status", "problem_family": "binary"},
    {"name": "iris (multiclass)", "dataset_path": "dataset/Iris.csv", "target_column": "Species", "problem_family": "multiclass"},
    {"name": "housing (regression)", "dataset_path": "dataset/Housing.csv", "target_column": "price", "problem_family": "regression"},
]


def run_case(case: dict) -> None:
    print(f"\n{'=' * 70}\n{case['name']}\n{'=' * 70}")

    state = {
        "dataset_path": case["dataset_path"],
        "target_column": case["target_column"],
        "problem_family": case["problem_family"],
        "primary_metric": FAMILY_METRICS[case["problem_family"]]["primary_metric"],
        "seed": 42,
        "run_id": f"emtest_{case['name'].split()[0]}",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
    }

    import agents.data_scientist as ds
    ds._propose_candidate_models = lambda family, eda, feats: [
        {"name": n, "why": "test-harness stub, LLM skipped for speed"}
        for n in __import__("config").MODEL_REGISTRY[family][:3]
    ]

    state.update(data_scientist_node(state))
    state.update(ml_engineer_node(state))
    if state.get("error"):
        print("ERROR upstream:", state["error"])
        return

    em_result = experiment_manager_node(state)

    print("primary_metric:", state["primary_metric"])
    print("ranked_models:", em_result["ranked_models"])
    print("comparison_table:")
    print(json.dumps(em_result["comparison_table"], indent=2))
    print("retrain_count:", em_result["retrain_count"])


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
