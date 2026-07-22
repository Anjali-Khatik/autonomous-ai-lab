"""Manual end-to-end validation of Business Analyst against Critic's REAL
output. Tests both business_params=None (must stay qualitative) and a
business_params dict supplied (must still decline computing impact, since
compute_impact is [V2]/not implemented) — the no-fabrication acceptance
test from spec §6 in miniature.
"""

import json
import sys

sys.path.insert(0, ".")

from agents.business_analyst import business_analyst_node
from agents.critic import critic_node
from agents.data_scientist import data_scientist_node
from agents.experiment_manager import experiment_manager_node
from agents.ml_engineer import ml_engineer_node
from config import FAMILY_METRICS


def build_state(business_params=None):
    state = {
        "dataset_path": "dataset/Iris.csv",
        "target_column": "Species",
        "problem_family": "multiclass",
        "primary_metric": FAMILY_METRICS["multiclass"]["primary_metric"],
        "seed": 42,
        "run_id": "batest_iris",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
        "plan": {},
        "retrain_count": 0,
        "business_params": business_params,
    }

    import agents.data_scientist as ds
    ds._propose_candidate_models = lambda family, eda, feats: [
        {"name": n, "why": "test-harness stub, LLM skipped for speed"}
        for n in __import__("config").MODEL_REGISTRY[family][:3]
    ]

    state.update(data_scientist_node(state))
    state.update(ml_engineer_node(state))
    state.update(experiment_manager_node(state))
    state.update(critic_node(state))
    return state


def run(business_params, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    state = build_state(business_params)
    print("critic_verdict:", state["critic_verdict"], "(business_analyst only reached on approve/conditional per graph edges)")

    result = business_analyst_node(state)
    print(json.dumps(result, indent=2))
    assert result["narrative"]["impact"] is None, "impact must be null in V1 no matter what"
    print("\nOK: impact is null, as required in V1.")


if __name__ == "__main__":
    run(None, "business_params = None")
    run({"cost_per_false_negative_usd": 500, "confirmation_fraction": 0.05}, "business_params supplied (must still decline to compute)")
