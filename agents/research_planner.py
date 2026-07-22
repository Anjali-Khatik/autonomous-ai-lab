"""Agent 1 — Research Planner. [V1: thin] Built LAST per spec build order.

Turns a user objective + dataset into a checkable plan. Family detection
is deterministic (tools/task_detection.py); the LLM only confirms/
overrides it with a one-line rationale, constrained to the 3 families V1
actually supports.

success_criteria is deliberately NOT LLM-generated: plan.success_criteria
only gets a primary_metric_target if the user supplied one via
constraints — a plausible-sounding target dreamed up with no grounding
(e.g. "f1 >= 0.85") would be exactly the kind of invented number rule §1
prohibits, even though it's a goal rather than a measurement. If the user
gave no target, success_criteria stays empty and Chief Scientist's
scorecard has nothing to check (GO/NO-GO then rests on critic_verdict
alone) rather than being graded against a number nobody actually asked for.
"""

from config import FAMILY_METRICS
from llm.client import reason
from state import LabState
from tools.task_detection import detect_problem_family, quick_profile

SUPPORTED_V1_FAMILIES = ("binary", "multiclass", "regression")

FAMILY_SCHEMA = {
    "family_confirmed": "one of: binary, multiclass, regression",
    "family_rationale": "one-line justification grounded only in the signals/profile given",
}


def research_planner_node(state: LabState) -> dict:
    """LangGraph node. Reads dataset_path/target_column/user_objective/
    constraints from state, returns problem_family/primary_metric/plan.
    """
    dataset_path = state["dataset_path"]
    target = state.get("target_column")
    user_objective = state.get("user_objective", "")
    constraints = state.get("constraints") or {}

    detection = detect_problem_family(dataset_path, target)
    profile = quick_profile(dataset_path)

    if detection["family"] is None:
        return {"error": f"research_planner: could not detect a V1-supported problem family — {detection['signals']}"}

    system = (
        "You are the Research Planner in an ML pipeline. A deterministic tool has already "
        "detected the likely problem family from the target column's dtype/cardinality - you "
        "may confirm it or override it, but ONLY to one of: binary, multiclass, regression "
        "(this system doesn't support anything else yet). Ground your rationale only in the "
        "detection signals and dataset profile given - don't invent statistics not shown to you."
    )
    user = (
        f"User objective: {user_objective!r}\n"
        f"Deterministic detection: {detection}\n"
        f"Dataset profile: {profile}\n\n"
        "Confirm or override the family, and give a one-line rationale."
    )
    llm_result = reason(system, user, response_schema=FAMILY_SCHEMA)

    confirmed_family = llm_result.get("family_confirmed")
    override_note = None
    if confirmed_family not in SUPPORTED_V1_FAMILIES:
        override_note = f"LLM returned unsupported/invalid family {confirmed_family!r} — kept deterministic detection {detection['family']!r}"
        confirmed_family = detection["family"]

    primary_metric = FAMILY_METRICS[confirmed_family]["primary_metric"]

    success_criteria = {}
    if "success_metric_target" in constraints:
        success_criteria["primary_metric_target"] = constraints["success_metric_target"]

    plan = {
        "objective": user_objective,
        "success_criteria": success_criteria,
        "constraints": constraints,
        "family_rationale": llm_result.get("family_rationale", ""),
    }
    if override_note:
        plan["override_note"] = override_note

    return {"problem_family": confirmed_family, "primary_metric": primary_metric, "plan": plan}
