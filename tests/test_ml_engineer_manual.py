"""Manual end-to-end validation of ML Engineer against Data Scientist's
REAL output (spec build order: "test against real upstream output").

Research Planner isn't built yet, so family/target/budget are supplied
directly as test-harness inputs.
"""

import json
import sys

sys.path.insert(0, ".")

from agents.data_scientist import data_scientist_node
from agents.ml_engineer import ml_engineer_node

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
        "seed": 42,
        "run_id": f"mltest_{case['name'].split()[0]}",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
    }

    import agents.data_scientist as ds
    ds._propose_candidate_models = lambda family, eda, feats: [
        {"name": n, "why": "test-harness stub, LLM skipped for speed"}
        for n in __import__("config").MODEL_REGISTRY[family][:3]
    ]

    ds_result = data_scientist_node(state)
    state.update(ds_result)
    print("candidate_models:", [c["name"] for c in state["candidate_models"]])
    print("feature_list:", state["feature_list"])

    ml_result = ml_engineer_node(state)

    if state.get("error") or ml_result.get("error"):
        print("ERROR:", ml_result.get("error"))
        return

    trained_models = ml_result["trained_models"]
    print(f"\n{len(trained_models)} leaderboard models:")
    for m in trained_models:
        print(f"\n--- {m['name']} ---")
        print("  metrics.train:", m["metrics"]["train"])
        print("  metrics.val:  ", m["metrics"]["val"])
        print("  metrics.test: ", {k: v for k, v in m["metrics"]["test"].items() if k != "confusion_matrix"})
        if "confusion_matrix" in m["metrics"]["test"]:
            print("  confusion_matrix:", m["metrics"]["test"]["confusion_matrix"])
        print("  timings:", m["timings"])
        print("  feature_importance:", m["feature_importance"])
        print("  hpo_trials:", m["hpo_trials"])


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
