"""Manual end-to-end validation of Chief Scientist against real upstream
output. Full pipeline test uses Iris (critic_verdict='conditional' with
real data) -> should cap at GO-WITH-CONDITIONS regardless of scorecard.
The pure GO path (critic_verdict='approve' + criteria met) is exercised
directly against the deterministic recommendation function since none of
our 3 real datasets currently produce a clean 'approve' verdict (their
real models all have a genuine caveat - see Critic's PROGRESS.md notes).
"""

import json
import sys

sys.path.insert(0, ".")

from agents.business_analyst import business_analyst_node
from agents.chief_scientist import _determine_recommendation, chief_scientist_node
from agents.critic import critic_node
from agents.data_scientist import data_scientist_node
from agents.experiment_manager import experiment_manager_node
from agents.ml_engineer import ml_engineer_node
from config import FAMILY_METRICS


def run_full_pipeline():
    print(f"\n{'=' * 70}\nFull pipeline: iris (multiclass, real data)\n{'=' * 70}")

    state = {
        "dataset_path": "dataset/Iris.csv",
        "target_column": "Species",
        "problem_family": "multiclass",
        "primary_metric": FAMILY_METRICS["multiclass"]["primary_metric"],
        "seed": 42,
        "run_id": "cstest_iris",
        "compute_budget": {"wall_clock_s": 60, "max_trials": None, "cost_cap_usd": None},
        "plan": {"success_criteria": {"primary_metric_target": 0.85}},
        "retrain_count": 0,
        "business_params": None,
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
    state.update(business_analyst_node(state))

    print("critic_verdict:", state["critic_verdict"])
    result = chief_scientist_node(state)
    print(json.dumps(result, indent=2))

    # Critic's approve-vs-conditional LLM call isn't perfectly deterministic across runs
    # even at temperature=0 (observed both outcomes for this same real finding across
    # separate runs) - so assert the invariant (recommendation matches whatever verdict
    # actually came back), not a specific hardcoded verdict.
    expected = _determine_recommendation(state["critic_verdict"], result["decision"]["success_scorecard"])
    assert result["decision"]["recommendation"] == expected, f"recommendation {result['decision']['recommendation']} doesn't match deterministic logic's {expected} for verdict={state['critic_verdict']}"
    assert result["decision"]["recommendation"] != "NO-GO", "graph only routes approve/conditional here — NO-GO would mean critic_verdict was reject/refer_alt, a routing bug"
    print(f"\nOK: recommendation ({result['decision']['recommendation']}) correctly matches deterministic logic for verdict={state['critic_verdict']}.")


def check_deterministic_recommendation_logic():
    print(f"\n{'=' * 70}\nDirect check: deterministic recommendation logic\n{'=' * 70}")

    cases = [
        ("approve", {"primary_metric_target_met": True}, "GO"),
        ("approve", {"primary_metric_target_met": False}, "GO-WITH-CONDITIONS"),
        ("approve", {}, "GO"),  # no criteria given = nothing to fail
        ("conditional", {"primary_metric_target_met": True}, "GO-WITH-CONDITIONS"),
        ("reject", {}, "NO-GO"),
        ("refer_alt", {}, "NO-GO"),
    ]
    for verdict, scorecard, expected in cases:
        actual = _determine_recommendation(verdict, scorecard)
        status = "OK" if actual == expected else "FAIL"
        print(f"  {status}: verdict={verdict!r} scorecard={scorecard} -> {actual} (expected {expected})")
        assert actual == expected


if __name__ == "__main__":
    check_deterministic_recommendation_logic()
    run_full_pipeline()
