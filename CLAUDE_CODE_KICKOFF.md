# Kickoff prompt for Claude Code

> Paste the block below as your first message to Claude Code, with
> `Autonomous-AI-Lab-Build-Spec.md`, `requirements.txt`, `.env.example`,
> and `PROGRESS.md` in the repo root. (Optionally add
> `Autonomous-AI-Lab-Product-Spec.md` for intent/context.)

---

We're building the **Autonomous AI Lab** — an agentic AutoML system (LangGraph
pipeline of reasoning agents). The full build contract is in
`Autonomous-AI-Lab-Build-Spec.md` at the repo root. Treat that file as the
source of truth and follow it exactly.

**Scope for now: V1 only.** Anything marked `[V2]` in the spec must be a stub
(a function with a docstring and a `TODO`), not a real implementation.

**Non-negotiable rules (from the spec — restating so they're front of mind):**
- All numbers come from deterministic tools, never from an LLM. Never fabricate
  metrics, dollar figures, or ROI.
- Agents are LLM planners that call trusted Python tools. Keep that boundary.
- Fit every data transform on the TRAIN split only; persist the fitted pipeline.
- One global seed threaded through split, engine, and tools.
- Enforce the compute budget and the Critic retrain cap (`MAX_RETRAINS`).
- Pass file paths/hashes through `LabState`, never model objects.

**AutoML engine = AutoGluon `TabularPredictor`.** It is fit ONCE under a
`time_limit`; per-model results come from `predictor.leaderboard(test_data)`.
Do NOT write a per-model training loop. Keep a **separate untouched test split**
that is never passed to `fit()` — only to `leaderboard()` — so the Critic's
overfitting check stays honest.

**Environment:** Python 3.11. ROCm PyTorch and `requirements.txt` are already
installed — don't reinstall or change torch. Read `.env.example` for the env
vars (Fireworks API key etc.).

**How I want you to work:**
1. First, scaffold the repo structure from §0 of the spec plus `state.py`,
   `config.py`, and `llm/client.py`. Show me the tree and the state/config files,
   then stop.
2. After I confirm, build agents in the exact order in §5 (Data Scientist first,
   Research Planner last), **one agent at a time**. For each agent: implement its
   tools, then the LangGraph node, then a quick test against the previous agent's
   real output. Stop after each agent for my review before moving on.
3. Track progress in `PROGRESS.md` — update it at the end of each agent.
4. The **Critic → ML Engineer retrain loop** is the headline feature. Build and
   test it as its own milestone: intentionally overfit a model, confirm the
   Critic catches it, the graph routes back to the ML Engineer, and the loop
   terminates at `MAX_RETRAINS`.

Start with step 1 (scaffold + state + config), then stop and wait for me.
