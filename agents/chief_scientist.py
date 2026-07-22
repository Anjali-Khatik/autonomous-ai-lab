"""Agent 7 — Chief Scientist. [V1: thin] Final GO/NO-GO synthesis.

Same pattern as Critic: the recommendation/confidence enums are decided
deterministically in Python (grounded in critic_verdict + measured success
criteria), not left to the LLM — the LLM only composes the rationale text
and next_steps from those grounded inputs, never invents a benchmark or
number itself.
"""

from llm.client import reason
from state import LabState
from tools.ranking import success_criteria_check

RECOMMENDATION_CONFIDENCE = {"GO": "high", "GO-WITH-CONDITIONS": "medium", "NO-GO": "low"}

NARRATIVE_SCHEMA = {
    "rationale": ["one grounded reason per item, referencing only the evidence given"],
    "next_steps": ["one recommended next step per item"],
}


def _top_model(state: LabState) -> dict:
    ranked_models = state["ranked_models"]
    trained_models = state["trained_models"]
    top_name = ranked_models[0]
    return next(m for m in trained_models if m["name"] == top_name)


def _determine_recommendation(critic_verdict: str, scorecard: dict) -> str:
    if critic_verdict in ("reject", "refer_alt"):
        # Defensive only — graph routing (§4) sends reject/refer_alt back to
        # ml_engineer/experiment_manager, never forward to Chief Scientist.
        return "NO-GO"
    criteria_met = [v for v in scorecard.values() if v is not None]
    all_met = all(criteria_met) if criteria_met else True  # no criteria given = nothing to fail
    if critic_verdict == "approve" and all_met:
        return "GO"
    return "GO-WITH-CONDITIONS"


def chief_scientist_node(state: LabState) -> dict:
    """LangGraph node. Reads ranked_models/critic_verdict/critic_findings/
    trained_models/plan/narrative from state, returns decision.
    """
    critic_verdict = state["critic_verdict"]
    plan = state.get("plan", {})
    primary_metric = state["primary_metric"]

    top_model = _top_model(state)
    test_metrics = top_model["metrics"]["test"]

    scorecard = success_criteria_check(test_metrics, plan, primary_metric)
    recommendation = _determine_recommendation(critic_verdict, scorecard)
    confidence = RECOMMENDATION_CONFIDENCE[recommendation]

    system = (
        "You are the Chief Scientist giving the final synthesis of an ML pipeline run. "
        "The recommendation and confidence are ALREADY DECIDED by policy - your job is only "
        "to write grounded rationale bullets and next steps, using ONLY the evidence given. "
        "Never invent a benchmark, number, or fact not present in the evidence."
    )
    user = (
        f"Winning model: {top_model['name']}\n"
        f"Test metrics: {test_metrics}\n"
        f"Critic verdict: {critic_verdict}\n"
        f"Critic findings: {state.get('critic_findings', [])}\n"
        f"Success criteria scorecard: {scorecard}\n"
        f"Decided recommendation: {recommendation} (confidence: {confidence})\n\n"
        "Write 2-4 rationale bullets explaining the recommendation, and 2-3 next_steps."
    )
    llm_result = reason(system, user, response_schema=NARRATIVE_SCHEMA)

    decision = {
        "winner": top_model["name"],
        "rationale": llm_result.get("rationale", []),
        "recommendation": recommendation,
        "confidence": confidence,
        "success_scorecard": scorecard,
        "next_steps": llm_result.get("next_steps", []),
    }

    return {"decision": decision}
