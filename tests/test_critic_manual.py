"""Manual end-to-end validation of Critic against Experiment Manager's
REAL output (spec build order: "test against real upstream output").

Headline milestone: loan dataset's top-ranked RandomForest already shows
a real severe overfit (train f1=1.0 vs test f1~0.89) from prior testing,
so the reject path is exercised naturally, not artificially forced.
"""

import json
import sys

sys.path.insert(0, ".")

from agents.critic import critic_node
from agents.data_scientist import data_scientist_node
from agents.experiment_manager import experiment_manager_node
from agents.ml_engineer import ml_engineer_node
from config import FAMILY_METRICS, MAX_RETRAINS

CASES = [
    {"name": "loan (binary)", "dataset_path": "dataset/loan_dataset.csv", "target_column": "Loan_Status", "problem_family": "binary"},
    {"name": "iris (multiclass)", "dataset_path": "dataset/Iris.csv", "target_column": "Species", "problem_family": "multiclass"},
    {"name": "housing (regression)", "dataset_path": "dataset/Housing.csv", "target_column": "price", "problem_family": "regression"},
]


def run_case(case: dict, retrain_count: int = 0) -> dict:
    print(f"\n{'=' * 70}\n{case['name']}  (retrain_count={retrain_count})\n{'=' * 70}")

    state = {
        "dataset_path": case["dataset_path"],
        "target_column": case["target_column"],
        "problem_family": case["problem_family"],
        "primary_metric": FAMILY_METRICS[case["problem_family"]]["primary_metric"],
        "seed": 42,
        "run_id": f"critictest_{case['name'].split()[0]}",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
        "plan": {},
        "retrain_count": retrain_count,
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
        return state
    state.update(experiment_manager_node(state))

    top_name = state["ranked_models"][0]
    top_model = next(m for m in state["trained_models"] if m["name"] == top_name)
    print(f"top-ranked model: {top_name}")
    print(f"  train primary metric: {top_model['metrics']['train'].get(state['primary_metric'])}")
    print(f"  test primary metric:  {top_model['metrics']['test'].get(state['primary_metric'])}")
    if "r2" in top_model["metrics"]["train"]:
        print(f"  train r2: {top_model['metrics']['train']['r2']}  test r2: {top_model['metrics']['test']['r2']}")

    critic_result = critic_node(state)
    state.update(critic_result)

    print(f"\ncritic_verdict: {state['critic_verdict']}")
    print("critic_findings:")
    print(json.dumps(state["critic_findings"], indent=2))

    return state


if __name__ == "__main__":
    for case in CASES:
        run_case(case, retrain_count=0)

    print(f"\n\n{'#' * 70}\nExtra: forcing retrain_count >= MAX_RETRAINS ({MAX_RETRAINS}) on loan\nto verify refer_alt routing\n{'#' * 70}")
    run_case(CASES[0], retrain_count=MAX_RETRAINS)
