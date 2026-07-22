"""Agent 4 — Experiment Manager. [V1: thin]

Aggregates trained_models into a ranked comparison table and owns
retrain_count. No LLM call in V1: the output contract (spec §3 Agent 4)
has no narrative field to write an LLM summary into, so the "light
narrative summary" mentioned in spec responsibilities is deferred —
TODO once a downstream consumer (e.g. UI) actually needs it.
"""

from state import LabState
from tools.ranking import aggregate_and_rank, log_mlflow


def experiment_manager_node(state: LabState) -> dict:
    """LangGraph node. Reads trained_models/primary_metric/run_id from
    state, returns ranked_models + retrain_count (initialized if unset).

    excluded_models (graph.py retrain-loop bookkeeping, not in spec — see
    PROGRESS.md deviations log) are filtered out before ranking: on the
    refer_alt path a model already exhausted its retries and is
    permanently disqualified, so re-ranking must promote the next
    surviving model rather than re-offering the same one.
    """
    trained_models = state["trained_models"]
    primary_metric = state["primary_metric"]
    run_id = state.get("run_id", "unknown_run")
    excluded_models = set(state.get("excluded_models", []))

    candidates = [m for m in trained_models if m["name"] not in excluded_models]
    result = aggregate_and_rank(candidates, primary_metric)

    try:
        log_mlflow(run_id, trained_models, result["ranked_models"])
    except Exception as e:
        # MLflow logging is observability, not pipeline-critical — don't fail the run over it.
        print(f"experiment_manager: mlflow logging failed (non-fatal): {e}")

    return {
        "ranked_models": result["ranked_models"],
        "comparison_table": result["table"],
        "retrain_count": state.get("retrain_count", 0),
    }
