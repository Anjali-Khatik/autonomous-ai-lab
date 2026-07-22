"""Agent 6 — Business/Impact Analyst. [V2] V1 ships as a qualitative stub only — no invented figures.

Translates the approved model's REAL test metrics and feature importances
into a plain-language summary. Impact/ROI computation is [V2]
(tools/impact.py::compute_impact, not implemented) — impact is always
null in V1 regardless of whether business_params was supplied, since
there's no tool yet to do that arithmetic. This agent never invents a
dollar figure on its own.
"""

from llm.client import reason
from state import LabState


def _top_model(state: LabState) -> dict:
    ranked_models = state["ranked_models"]
    trained_models = state["trained_models"]
    top_name = ranked_models[0]
    return next(m for m in trained_models if m["name"] == top_name)


def business_analyst_node(state: LabState) -> dict:
    """LangGraph node. Reads the approved model's metrics/feature_importance
    and business_params from state, returns narrative (impact always null in V1).
    """
    top_model = _top_model(state)
    test_metrics = top_model["metrics"]["test"]
    top_features = dict(sorted(top_model["feature_importance"].items(), key=lambda kv: -abs(kv[1]))[:5])
    business_params = state.get("business_params")

    system = (
        "You are a business analyst translating ML model results into plain language for a "
        "non-technical stakeholder. You are given REAL computed test metrics and feature "
        "importances - use ONLY these numbers. Never invent dollar figures, ROI, or any "
        "business metric not directly given to you."
    )
    if business_params:
        system += (
            " business_params were supplied, but no impact-calculation tool exists yet in "
            "this version of the pipeline - explicitly say that dollar impact can't be "
            "computed yet, don't attempt the arithmetic yourself."
        )
    else:
        system += " No business_params were supplied, so explicitly state that no dollar impact can be computed."

    user = (
        f"Test metrics: {test_metrics}\n"
        f"Top feature importances: {top_features}\n"
        f"business_params: {business_params}\n\n"
        "Write one short plain-language paragraph: (1) translate the metrics for a "
        "non-technical reader, (2) note what the top features suggest is driving "
        "predictions. Do not compute or invent any business impact or dollar figures."
    )
    summary = reason(system, user)

    return {"narrative": {"summary": summary, "impact": None}}
