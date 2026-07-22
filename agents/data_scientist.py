"""Agent 2 — Data Scientist. [V1: core] Built FIRST.

Investigates the dataset, cleans it leakage-safely, selects features, and
proposes candidate models. Purely a tool-calling node: EDA/cleaning/
selection are deterministic Python; only the candidate-model rationale
comes from the LLM (never a number).
"""

from config import MODEL_REGISTRY
from llm.client import reason
from state import LabState
from tools.cleaning import build_clean_pipeline
from tools.eda import run_eda
from tools.feature_select import select_features

CANDIDATE_MODEL_SCHEMA = {
    "candidate_models": [{"name": "string, must be one of the allowed model names", "why": "one-line rationale"}]
}


def _propose_candidate_models(family: str, eda_summary: dict, feature_list: list[str]) -> list[dict]:
    allowed = MODEL_REGISTRY.get(family, [])
    if not allowed:
        return []

    system = (
        "You are a data scientist shortlisting 3-5 candidate ML models for a tabular "
        "problem. You may ONLY choose from the allowed model list given. Do not invent "
        "metrics or performance numbers - you have not seen any results yet, only EDA."
    )
    user = (
        f"Problem family: {family}\n"
        f"Allowed models: {allowed}\n"
        f"Number of selected features: {len(feature_list)}\n"
        f"EDA summary: {eda_summary}\n\n"
        "Pick 3-5 models from the allowed list and give a one-line 'why' for each, "
        "grounded in the EDA (e.g. dataset size, missingness, feature types, class balance)."
    )
    result = reason(system, user, response_schema=CANDIDATE_MODEL_SCHEMA)
    candidates = result.get("candidate_models", [])
    return [c for c in candidates if c.get("name") in allowed]


def data_scientist_node(state: LabState) -> dict:
    """LangGraph node. Reads dataset_path/problem_family/seed from state,
    returns the fields this agent owns (see spec §3 Agent 2 output contract).
    """
    dataset_path = state["dataset_path"]
    target = state["target_column"]
    family = state["problem_family"]
    seed = state["seed"]
    run_id = state.get("run_id")

    eda_summary = run_eda(dataset_path, target, family)

    clean_result = build_clean_pipeline(dataset_path, target, family, seed, run_id=run_id)

    selection_result = select_features(clean_result["cleaned_data_path"], target, family)

    candidate_models = _propose_candidate_models(family, eda_summary, selection_result["feature_list"])

    dataset_profile = {
        "shape": eda_summary["shape"],
        "numeric": len(eda_summary["column_types"]["numeric"]),
        "categorical": len(eda_summary["column_types"]["categorical"]),
        "datetime": len(eda_summary["column_types"]["datetime"]),
        "missing_pct": eda_summary["missing_pct"],
        "quality_score": eda_summary["quality_score"],
        "eda_summary": eda_summary,
        "cleaning_report": clean_result["report"],
        "feature_selection": {"dropped": selection_result["dropped"], "method": selection_result["method"]},
    }

    return {
        "dataset_profile": dataset_profile,
        "cleaned_data_path": clean_result["cleaned_data_path"],
        "cleaned_data_hash": clean_result["cleaned_data_hash"],
        "feature_list": selection_result["feature_list"],
        "candidate_models": candidate_models,
    }
