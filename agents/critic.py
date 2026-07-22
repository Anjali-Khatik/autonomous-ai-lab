"""Agent 5 — Critic. [V1: core] Quality gate + retrain-loop trigger — the headline milestone.

Deterministic tools decide the safety-critical boundary (severe
overfitting/leakage -> reject or refer_alt, with MAX_RETRAINS enforced
in Python); the LLM only adjudicates the ambiguous approve-vs-conditional
case and writes the feedback narrative. This mirrors rule §"numbers from
code" — the retrain-loop routing must stay deterministic and testable
regardless of what any LLM proposes.
"""

from config import MAX_RETRAINS
from llm.client import reason
from state import LabState
from tools.leakage import check_leakage
from tools.metric_check import check_metric_choice
from tools.overfitting import check_overfitting
from tools.validation_check import check_validation

SEVERE_SEVERITIES = ("severe", "high")

VERDICT_SCHEMA = {"critic_verdict": "approve|conditional", "feedback": "one paragraph explaining the decision, grounded only in the findings given"}


def _build_findings(overfit: dict, leakage: dict, metric: dict, validation: dict) -> list[dict]:
    findings = []

    if overfit["detected"]:
        findings.append({
            "type": "overfitting",
            "severity": overfit["severity"],
            "evidence": {
                "gap_metric": overfit["gap_metric"],
                "train": overfit["train"], "val": overfit["val"], "test": overfit["test"],
                "gaps": overfit["gaps"],
            },
            "recommendation": (
                "reduce model complexity / add regularization; retrain"
                if overfit["severity"] == "severe"
                else "monitor; consider regularization if it worsens on retrain"
            ),
        })

    for f in leakage["findings"]:
        if f["type"] == "distribution_shift":
            findings.append({
                "type": "leakage", "severity": "moderate", "evidence": f,
                "recommendation": f"investigate train/test distribution shift in {f['feature']}; consider re-splitting",
            })
        elif f["type"] == "suspicious_high_importance":
            if f["looks_like_id_or_date"]:
                findings.append({
                    "type": "leakage", "severity": "high", "evidence": f,
                    "recommendation": f"investigate whether {f['feature']} leaks target information; consider dropping and retraining",
                })
            else:
                findings.append({
                    "type": "leakage", "severity": "moderate", "evidence": f,
                    "recommendation": (
                        f"{f['feature']} dominates importance ({f['importance_share']:.0%}) but doesn't look like "
                        "an ID/date column - likely genuine signal, not leakage; verify manually if unsure"
                    ),
                })

    if not metric["appropriate"]:
        findings.append({
            "type": "metric_choice", "severity": "moderate",
            "evidence": {"class_balance": metric["class_balance"]},
            "recommendation": "; ".join(metric["findings"]),
        })

    if not validation["sound"]:
        findings.append({
            "type": "validation", "severity": "moderate", "evidence": {},
            "recommendation": "; ".join(validation["findings"]),
        })

    return findings


def critic_node(state: LabState) -> dict:
    """LangGraph node. Reads ranked_models/trained_models/problem_family/
    primary_metric/plan/retrain_count from state, returns critic_verdict +
    critic_findings. Verdict in approve|reject|conditional|refer_alt.
    """
    ranked_models = state["ranked_models"]
    trained_models = state["trained_models"]
    family = state["problem_family"]
    plan = state.get("plan", {})
    retrain_count = state.get("retrain_count", 0)

    top_name = ranked_models[0]
    top_model = next(m for m in trained_models if m["name"] == top_name)

    class_balance = state.get("dataset_profile", {}).get("eda_summary", {}).get("class_balance")

    overfit = check_overfitting(top_model["metrics"], family)
    leakage = check_leakage(state["cleaned_data_path"], top_model["feature_importance"], family)
    metric = check_metric_choice(state["primary_metric"], family, class_balance)
    validation = check_validation(plan, family)

    findings = _build_findings(overfit, leakage, metric, validation)

    forced_verdict = None
    if any(f["severity"] in SEVERE_SEVERITIES for f in findings):
        forced_verdict = "refer_alt" if retrain_count >= MAX_RETRAINS else "reject"

    system = (
        "You are the Critic in an ML pipeline's quality gate. You are given deterministic "
        "tool findings about the top-ranked model - you do not see raw data or compute any "
        "numbers yourself. "
    )
    if forced_verdict is not None:
        system += (
            f"A severe/high-severity finding was detected, so the verdict is ALREADY DECIDED "
            f"as '{forced_verdict}' by policy - just write a one-paragraph feedback explaining "
            f"why, grounded only in the findings given. Set critic_verdict to '{forced_verdict}'."
        )
    else:
        system += (
            "No severe/high-severity finding was detected. Choose 'approve' if findings are "
            "empty or trivial, or 'conditional' if there are real-but-minor issues worth "
            "flagging. Write a one-paragraph feedback explaining the decision."
        )

    user = f"Model: {top_name}\nFindings: {findings}\nRetrain count: {retrain_count}/{MAX_RETRAINS}"
    llm_result = reason(system, user, response_schema=VERDICT_SCHEMA)

    if forced_verdict is not None:
        verdict = forced_verdict
    else:
        verdict = llm_result.get("critic_verdict")
        if verdict not in ("approve", "conditional"):
            verdict = "conditional"  # safe default if LLM returns something unexpected

    # LLM feedback text isn't persisted into critic_findings: spec §3 Agent 5's own output
    # example has no narrative/feedback field on a finding (only type/severity/evidence/
    # recommendation) — same reasoning as Experiment Manager's narrative skip. Printed here
    # for visibility during manual testing only.
    print(f"critic feedback ({verdict}): {llm_result.get('feedback', '')}")

    return {"critic_verdict": verdict, "critic_findings": findings}
