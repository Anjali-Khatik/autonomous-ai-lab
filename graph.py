"""LangGraph assembly: 7 agent nodes + FAILED node, linear edges + the
Critic retry-loop conditional edge (the headline milestone).

Routing (spec §4):
  research_planner -> data_scientist -> ml_engineer -> experiment_manager -> critic
  critic --(reject & retrain_count < MAX_RETRAINS)--> ml_engineer         # increment retrain_count
  critic --(reject & retrain_count >= MAX_RETRAINS)--> experiment_manager # refer_alt, try next model ONCE
  critic --(approve | conditional)--> business_analyst -> chief_scientist -> END
  any node hard error --> FAILED (state["error"] set) -> END

Two small deterministic prep nodes (prepare_retry, prepare_refer_alt) sit
between critic and its two loop targets — LangGraph conditional-edge
functions only choose a route, they don't mutate state, so incrementing
retrain_count / recording exclusions needs its own node. Not named in the
spec's node list, but required to implement the routing spec §4 actually
describes. See PROGRESS.md deviations log.

critic_node's own forced-verdict logic already enforces MAX_RETRAINS in
Python regardless of what the LLM proposes (see agents/critic.py) — this
graph's routing just dispatches on whatever verdict comes back. retrain_count
is never reset once the cap is hit, so if the ONE alternate model tried
after refer_alt also fails, the next critic pass naturally returns
refer_alt again (not reject) and the graph below recognizes that as
"already used the one extra try" and terminates via FAILED instead of
cascading to a third model.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.business_analyst import business_analyst_node
from agents.chief_scientist import chief_scientist_node
from agents.critic import critic_node
from agents.data_scientist import data_scientist_node
from agents.experiment_manager import experiment_manager_node
from agents.ml_engineer import ml_engineer_node
from agents.research_planner import research_planner_node
from config import leaderboard_name_to_key
from state import LabState


def _safe_node(name: str, fn):
    """Wrap a node so any raised exception becomes state['error'] and routes
    to FAILED, instead of crashing the whole graph run (spec rule: "any hard
    error --> FAILED node, write error to state").
    """
    def wrapped(state: LabState) -> dict:
        try:
            return fn(state)
        except Exception as e:
            return {"error": f"{name}: {type(e).__name__}: {e}"}
    wrapped.__name__ = f"safe_{name}"
    return wrapped


def _prepare_retry_node(state: LabState) -> dict:
    """On Critic reject: increment retrain_count, and — only if the
    rejection was genuinely for overfitting (not e.g. a pure validation/
    metric-choice finding) — exclude the flagged model's specific
    AutoGluon learner key from the next ml_engineer attempt.
    """
    retrain_count = state.get("retrain_count", 0) + 1
    excluded_keys = list(state.get("excluded_learner_keys", []))

    overfit_flagged = any(f["type"] == "overfitting" for f in state.get("critic_findings", []))
    if overfit_flagged:
        top_name = state["ranked_models"][0]
        key = leaderboard_name_to_key(top_name)
        if key and key not in excluded_keys:
            excluded_keys.append(key)

    return {"retrain_count": retrain_count, "excluded_learner_keys": excluded_keys}


def _prepare_refer_alt_node(state: LabState) -> dict:
    """On Critic refer_alt (retrain cap hit): permanently exclude the
    exhausted top model from re-ranking so Experiment Manager promotes the
    next-best surviving model.
    """
    excluded_models = list(state.get("excluded_models", []))
    top_name = state["ranked_models"][0]
    if top_name not in excluded_models:
        excluded_models.append(top_name)
    return {"excluded_models": excluded_models}


def _failed_node(state: LabState) -> dict:
    """Terminal node. Fills in a contextual error message if the graph
    arrived here via critic's give-up route rather than a raised exception
    (which already set state['error'] via _safe_node).
    """
    if state.get("error"):
        return {}
    return {
        "error": (
            f"critic: no candidate model passed the quality gate "
            f"(final verdict={state.get('critic_verdict')!r}) after retrain + one alternate-model attempt"
        )
    }


def _after(next_node: str):
    """Generic post-node router: error -> failed, else continue."""
    def _route(state: LabState) -> str:
        return "failed" if state.get("error") else next_node
    return _route


def _critic_router(state: LabState) -> str:
    if state.get("error"):
        return "failed"
    verdict = state["critic_verdict"]
    if verdict in ("approve", "conditional"):
        return "business_analyst"
    if verdict == "reject":
        return "prepare_retry"
    if verdict == "refer_alt":
        if state.get("excluded_models"):
            # already used the one alternate-model attempt and it also failed
            return "failed"
        return "prepare_refer_alt"
    return "failed"  # defensive — unexpected verdict value


def build_graph(hitl: bool = False):
    """Assemble and compile the pipeline graph.

    hitl=True adds interrupts after research_planner (review the plan
    before compute is spent) and after chief_scientist (review the
    decision before it's considered final) — spec §4's HITL checkpoints.
    Requires a checkpointer to resume from an interrupt, so hitl=True
    always compiles with a MemorySaver; hitl=False (default here, matches
    "off for benchmarking" per spec) compiles without one for a plain
    single-shot invoke().
    """
    graph = StateGraph(LabState)

    graph.add_node("research_planner", _safe_node("research_planner", research_planner_node))
    graph.add_node("data_scientist", _safe_node("data_scientist", data_scientist_node))
    graph.add_node("ml_engineer", _safe_node("ml_engineer", ml_engineer_node))
    graph.add_node("experiment_manager", _safe_node("experiment_manager", experiment_manager_node))
    graph.add_node("critic", _safe_node("critic", critic_node))
    graph.add_node("prepare_retry", _safe_node("prepare_retry", _prepare_retry_node))
    graph.add_node("prepare_refer_alt", _safe_node("prepare_refer_alt", _prepare_refer_alt_node))
    graph.add_node("business_analyst", _safe_node("business_analyst", business_analyst_node))
    graph.add_node("chief_scientist", _safe_node("chief_scientist", chief_scientist_node))
    graph.add_node("failed", _failed_node)

    graph.set_entry_point("research_planner")

    graph.add_conditional_edges("research_planner", _after("data_scientist"), {"data_scientist": "data_scientist", "failed": "failed"})
    graph.add_conditional_edges("data_scientist", _after("ml_engineer"), {"ml_engineer": "ml_engineer", "failed": "failed"})
    graph.add_conditional_edges("ml_engineer", _after("experiment_manager"), {"experiment_manager": "experiment_manager", "failed": "failed"})
    graph.add_conditional_edges("experiment_manager", _after("critic"), {"critic": "critic", "failed": "failed"})

    graph.add_conditional_edges("critic", _critic_router, {
        "business_analyst": "business_analyst",
        "prepare_retry": "prepare_retry",
        "prepare_refer_alt": "prepare_refer_alt",
        "failed": "failed",
    })
    graph.add_edge("prepare_retry", "ml_engineer")
    graph.add_edge("prepare_refer_alt", "experiment_manager")

    graph.add_conditional_edges("business_analyst", _after("chief_scientist"), {"chief_scientist": "chief_scientist", "failed": "failed"})
    graph.add_conditional_edges("chief_scientist", _after(END), {END: END, "failed": "failed"})

    graph.add_edge("failed", END)

    if hitl:
        return graph.compile(checkpointer=MemorySaver(), interrupt_after=["research_planner", "chief_scientist"])
    return graph.compile()
