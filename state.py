"""Shared state contract for the Autonomous AI Lab LangGraph pipeline.

One LabState flows through every node. Large artifacts (data, models) live
on disk; state holds paths + hashes only (never raw DataFrames or model
objects — they aren't JSON-serializable and don't belong in a checkpointed
graph state).
"""

from typing import TypedDict


class BudgetSpec(TypedDict):
    wall_clock_s: int
    max_trials: int | None
    cost_cap_usd: float | None


class ModelResult(TypedDict):
    name: str
    model_path: str
    metrics: dict          # {"train": {...}, "val": {...}, "test": {...}}
    timings: dict           # {"train_s": float, "infer_ms_per_row": float, "size_mb": float}
    feature_importance: dict  # {feature: score}
    hpo_trials: int


class LabState(TypedDict):
    # run config
    run_id: str
    seed: int
    compute_budget: BudgetSpec
    business_params: dict | None
    hitl: bool

    # raw input (spec §3 Agent 1 input contract; not listed in spec §1 —
    # added here since Research Planner cannot run without them)
    dataset_path: str
    user_objective: str
    target_column: str | None
    constraints: dict

    # research planner
    problem_family: str
    primary_metric: str
    plan: dict

    # data scientist
    dataset_profile: dict
    cleaned_data_path: str
    cleaned_data_hash: str
    feature_list: list[str]
    candidate_models: list[dict]

    # ml engineer
    trained_models: list[ModelResult]

    # experiment manager
    ranked_models: list[str]
    comparison_table: list[dict]  # spec §3 Agent 4 output example; not in spec §1 — same
                                   # class of gap as dataset_path etc. above
    retrain_count: int
    excluded_models: list[str]    # graph.py retrain-loop bookkeeping — model names that
                                   # exhausted retries (refer_alt) and are excluded from
                                   # re-ranking. Not in spec — see PROGRESS.md deviations log.

    # critic
    critic_verdict: str | None
    critic_findings: list[dict]
    excluded_learner_keys: list[str]  # graph.py retrain-loop bookkeeping — AutoGluon learner
                                       # keys excluded from the next ml_engineer retry because
                                       # Critic flagged that specific model for overfitting.
                                       # Not in spec — see PROGRESS.md deviations log.

    # final
    narrative: dict | None
    decision: dict | None
    error: str | None
