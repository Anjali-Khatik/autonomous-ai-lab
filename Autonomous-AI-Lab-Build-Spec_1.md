# Autonomous AI Lab — Build Specification (for Claude Code)

> Hand this document to Claude Code as the source of truth. It defines all seven agents (purpose, responsibilities, tools, input, output), the shared state contract, the graph wiring, the build order, and acceptance tests. Build **V1 first**; leave V2 items as clearly-marked stubs.

---

## HOW TO USE THIS DOCUMENT (read first, Claude Code)

**Non-negotiable rules — apply to every file you write:**

1. **Numbers come from code, never from an LLM.** Any metric, count, or statistic must be produced by a deterministic Python tool and passed to the LLM as data. LLM prompts may *interpret* numbers; they may never compute or invent them. Never fabricate dollar figures, ROI, or business metrics.
2. **LLM vs code boundary.** Agents are LLM planners that call trusted tools. Reasoning (choosing models, interpreting EDA, writing explanations, GO/NO-GO framing) = LLM. Execution (EDA, cleaning, training, HPO, metrics, leakage tests) = Python.
3. **Leakage safety.** Fit every transform on the **train split only**; persist the fitted pipeline; apply to val/test. Never fit on the full dataset.
4. **One global seed** threaded through split, engine, and tools.
5. **Compute budget is enforced.** No open-ended HPO — stop at the budget and return best-so-far.
6. **Pass paths/hashes in state, not model objects** (they aren't JSON-serializable).
7. **Build V1 depth only.** Where an agent/tool is marked `[V2]`, write a stub with a `TODO` and a docstring, not a full implementation.
8. **Don't invent business logic that isn't specified.** Leave a clear `TODO` and a comment.

**Build order** (each agent is built and tested against the *real* output of the previous one):
`Data Scientist → ML Engineer → Experiment Manager → Critic (+retrain loop) → [V2] Business Analyst → Chief Scientist → Research Planner`
(Research Planner is built last on purpose — by then the exact structured output the Data Scientist needs is known.)

---

## 0. Tech Stack & Repo Layout

**Stack:** Python 3.11 · LangGraph (orchestration) · **AutoGluon `TabularPredictor`** (v1.5.0, tabular subpackage) as the AutoML engine · Fireworks AI for reasoning LLM calls (Llama/Qwen) · scikit-learn (EDA/cleaning/metrics) · MLflow (tracking) · joblib + AutoGluon's own predictor save (serialization) · pandas/numpy. GPU training on AMD ROCm — **install ROCm PyTorch before AutoGluon** (AutoGluon has a hard torch dependency); CPU fallback otherwise.

**Suggested repo structure:**
```
autonomous-ai-lab/
├── graph.py                 # LangGraph assembly, edges, conditional routing
├── state.py                 # LabState TypedDict (§1)
├── config.py                # seed, budget defaults, model registry, family metrics
├── agents/
│   ├── research_planner.py
│   ├── data_scientist.py
│   ├── ml_engineer.py
│   ├── experiment_manager.py
│   ├── critic.py
│   ├── business_analyst.py  # [V2] stub
│   └── chief_scientist.py
├── tools/
│   ├── task_detection.py
│   ├── eda.py
│   ├── cleaning.py
│   ├── feature_select.py
│   ├── engine_wrapper.py    # AutoGluon TabularPredictor behind one interface
│   ├── evaluate.py
│   ├── serialize.py
│   ├── ranking.py
│   ├── overfitting.py
│   ├── leakage.py
│   ├── metric_check.py
│   └── validation_check.py
├── llm/
│   └── client.py            # Fireworks client + prompt helpers
├── tests/
│   └── test_*.py            # acceptance tests (§6)
└── PROGRESS.md
```

**LLM helper contract** (`llm/client.py`):
```python
def reason(system: str, user: str, response_schema: dict | None = None) -> dict | str:
    """Call the reasoning model. If response_schema given, instruct JSON-only output and parse it.
    Used ONLY for reasoning/narrative — never to compute metrics."""
```

---

## 1. Shared State Contract (`state.py`)

One typed object flows through the graph. Large artifacts live on disk; state holds paths + hashes.

```python
class BudgetSpec(TypedDict):
    wall_clock_s: int          # e.g. 300
    max_trials: int | None     # optional cap
    cost_cap_usd: float | None # optional

class ModelResult(TypedDict):
    name: str
    model_path: str
    metrics: dict              # {"train": {...}, "val": {...}, "test": {...}}
    timings: dict              # {"train_s": float, "infer_ms_per_row": float, "size_mb": float}
    feature_importance: dict   # {feature: score}
    hpo_trials: int

class LabState(TypedDict):
    # run config
    run_id: str
    seed: int
    compute_budget: BudgetSpec
    business_params: dict | None        # user-supplied ONLY; never invented
    hitl: bool                          # human-in-the-loop checkpoints on/off

    # research planner
    problem_family: str                 # binary|multiclass|regression|forecast|cluster|anomaly
    primary_metric: str
    plan: dict

    # data scientist
    dataset_profile: dict
    cleaned_data_path: str
    cleaned_data_hash: str
    feature_list: list[str]
    candidate_models: list[dict]        # [{"name":..., "why":...}]

    # ml engineer
    trained_models: list[ModelResult]

    # experiment manager
    ranked_models: list[str]            # model names, best first
    retrain_count: int

    # critic
    critic_verdict: str | None          # approve|reject|conditional|refer_alt
    critic_findings: list[dict]

    # final
    narrative: dict | None              # [V2] grounded explanation
    decision: dict | None               # GO / NO-GO
    error: str | None                   # set on hard failure → FAILED node
```

---

## 2. Problem-Family Branching (`config.py`)

The pipeline branches on `problem_family`. Metrics/engine objective switch per family.

| family | primary_metric | secondary | validation split |
|--------|----------------|-----------|------------------|
| binary | f1 (pr_auc if imbalanced) | roc_auc, recall, precision, mcc | stratified k-fold |
| multiclass | macro_f1 | balanced_acc, per_class_recall | stratified k-fold |
| regression | rmse (mae if outlier-heavy) | r2, mape, rmsle | k-fold |
| forecast `[V2]` | mae / smape | rmse, coverage | **temporal** split |
| cluster `[V2]` | silhouette | davies_bouldin, ch_index | n/a (no target) |
| anomaly `[V2]` | pr_auc (if labels) | precision@k | semi-supervised |

Store a `FAMILY_METRICS` dict and a `MODEL_REGISTRY` (candidate model names per family) in `config.py`.

---

## 3. THE AGENTS

Each agent below: **Purpose · Responsibilities · Tools Used (with signatures) · Input · Output · Depth**. JSON values are illustrative *shapes* — real values are produced by tools at runtime.

---

### AGENT 1 — Research Planner  `[V1: thin]`

**Purpose.** Turn a user objective + dataset into a checkable plan: detect the problem family, set the primary metric and success criteria, capture constraints.

**Responsibilities.**
- Run task detection on the raw dataset; confirm/override the detected family with a one-line LLM rationale.
- Set `primary_metric` from `FAMILY_METRICS`.
- Capture constraints (interpretability, latency, GPU availability, retraining cadence) and success criteria.
- If `hitl`, pause for user confirmation of the plan before compute is spent.

**Tools Used.**
```python
# tools/task_detection.py
def detect_problem_family(df_path: str, target: str | None) -> dict:
    """Deterministic inference from target dtype, cardinality, datetime index, class balance.
    Returns {"family": str, "primary_metric": str, "signals": {...}, "confidence": float}."""

def quick_profile(df_path: str) -> dict:
    """Fast profile: shape, dtypes, missing %, memory estimate. Returns dataset_profile stub."""
```
LLM: confirms/overrides the detected family and writes the plan narrative.

**Input.**
```json
{
  "user_objective": "string, free text",
  "dataset_path": "/data/raw.csv",
  "target_column": "string | null",
  "constraints": {"interpretability": true, "max_latency_ms": 100, "gpu_available": true},
  "compute_budget": {"wall_clock_s": 300, "max_trials": null, "cost_cap_usd": null}
}
```

**Output** (writes into `LabState`).
```json
{
  "problem_family": "binary",
  "primary_metric": "f1",
  "plan": {
    "objective": "string",
    "success_criteria": {"primary_metric_target": 0.85},
    "constraints": {"interpretability": true, "max_latency_ms": 100},
    "family_rationale": "one-line LLM justification for the detected family"
  }
}
```

---

### AGENT 2 — Data Scientist  `[V1: core]`

**Purpose.** Investigate the dataset, clean it leakage-safely, do family-appropriate feature work, and shortlist candidate models.

**Responsibilities.**
- EDA (numeric/categorical/datetime) → structured summary + quality score.
- Cleaning: missing values, duplicates, outliers, encoding, scaling — **fit on train only**, persist pipeline.
- Feature selection (filter/embedded). **Feature extraction (PCA/LDA/autoencoder) is conditional** — only when very high-dimensional or highly collinear; off by default for tree models.
- LLM proposes 3–5 candidate models from `MODEL_REGISTRY[family]` with rationale.
- Persist cleaned data; record path + hash.

**Tools Used.**
```python
# tools/eda.py
def run_eda(df_path: str, target: str, family: str) -> dict:
    """Stats, distributions, correlations, outliers, class balance. Returns eda_summary + quality_score."""

# tools/cleaning.py
def build_clean_pipeline(df_path: str, target: str, family: str, seed: int) -> dict:
    """Fit imputation/encoding/scaling on TRAIN split only, persist fitted pipeline,
    apply to all splits. Returns {cleaned_data_path, cleaned_data_hash, pipeline_path, report}."""

# tools/feature_select.py
def select_features(cleaned_path: str, target: str, family: str) -> dict:
    """Filter + embedded selection. Returns {feature_list, dropped, method}.
    Optional conditional extraction gated on dimensionality/collinearity."""
```
LLM: interprets EDA + picks candidate models with a `why` each.

**Input.** Reads from `LabState`: `dataset_path`(from plan), `problem_family`, `primary_metric`, `plan`, `seed`.

**Output.**
```json
{
  "dataset_profile": {
    "shape": [100000, 50],
    "numeric": 35, "categorical": 15, "datetime": 2,
    "missing_pct": 2.5, "quality_score": 0.85,
    "eda_summary": { "...": "family-appropriate summary" }
  },
  "cleaned_data_path": "/data/cleaned.parquet",
  "cleaned_data_hash": "sha256:...",
  "feature_list": ["f1", "f2", "..."],
  "candidate_models": [
    {"name": "XGBoost", "why": "handles the detected characteristics well"},
    {"name": "LightGBM", "why": "fast, strong tabular baseline"},
    {"name": "RandomForest", "why": "stable interpretable baseline"}
  ]
}
```

---

### AGENT 3 — ML Engineer  `[V1: core — wrap AutoGluon, don't hand-roll HPO]`

**Purpose.** Train and tune candidate models under the compute budget using **one AutoGluon `TabularPredictor`**, then return honest per-model metrics from its leaderboard on an untouched test set.

> **API note (important for Claude Code):** AutoGluon does **not** train one model at a time. `TabularPredictor.fit(train_data, time_limit=..., presets=...)` trains many models + a stacked ensemble in a single call under `time_limit`. Per-model results come from `predictor.leaderboard(test_data)`. So the wrapper is **fit-once → read leaderboard**, not a `train_one` loop. The `candidate_models` from the Data Scientist become a **hyperparameter/estimator restriction** passed to `fit(hyperparameters=...)`, not separate training runs.

**Responsibilities.**
- Profile hardware (ROCm GPU vs CPU) and set AutoGluon's `num_gpus`/`num_cpus` accordingly.
- Map `problem_family` → AutoGluon `problem_type` (`binary`/`multiclass`/`regression`) and `eval_metric` (from `FAMILY_METRICS`).
- Fit **one** `TabularPredictor` under `compute_budget.wall_clock_s` (→ `time_limit`), optionally restricting learners via `hyperparameters` derived from `candidate_models`.
- Read `predictor.leaderboard(test_data)` for per-model metrics on the **held-out test set** (AutoGluon manages its own internal validation; keep a separate untouched test split for honest reporting).
- Persist the predictor directory (AutoGluon's native save) + record path, timings, size, and per-model feature importance (`predictor.feature_importance`).
- Honor the budget: `time_limit` caps it; on failure record the error, don't crash the run.

**Tools Used.**
```python
# tools/engine_wrapper.py
def fit_predictor(cleaned_path: str, feature_list: list[str], target: str,
                  family: str, candidate_models: list[dict],
                  budget: BudgetSpec, seed: int, hw: dict) -> dict:
    """Fit ONE AutoGluon TabularPredictor under budget.time_limit.
    - problem_type/eval_metric derived from `family`.
    - `candidate_models` -> AutoGluon `hyperparameters` restriction (optional).
    - num_gpus/num_cpus from `hw`.
    Returns {"predictor_path": str, "fit_summary": {...}}."""

def read_leaderboard(predictor_path: str, test_path: str, family: str) -> list[ModelResult]:
    """Load predictor, run leaderboard(test_data), and build one ModelResult per row
    (name, metrics{val,test}, timings, feature_importance). This replaces per-model training."""

def profile_hardware() -> dict:
    """Detect ROCm GPU + CPU cores/RAM. Returns {"num_gpus": int, "num_cpus": int, "tier": str}."""

# tools/evaluate.py
def evaluate_on_test(predictor_path: str, model_name: str, test_path: str, family: str) -> dict:
    """Family-appropriate metrics for a single leaderboard model on the untouched test set."""

# tools/serialize.py
def register_predictor(predictor_path: str, meta: dict) -> str:
    """Record the AutoGluon predictor dir + metadata in the run's artifact store; return path."""
```
LLM: none (execution agent). Optional progress narration.

**Input.** Reads `candidate_models`, `cleaned_data_path`, `feature_list`, `problem_family`, `compute_budget`, `seed`.

**Output.**
```json
{
  "trained_models": [
    {
      "name": "XGBoost",
      "model_path": "/models/xgboost.pkl",
      "metrics": {
        "train": {"f1": 0.892, "roc_auc": 0.945},
        "val":   {"f1": 0.889, "roc_auc": 0.942},
        "test":  {"f1": 0.887, "roc_auc": 0.940, "confusion_matrix": [[950,50],[26,974]]}
      },
      "timings": {"train_s": 145.3, "infer_ms_per_row": 0.023, "size_mb": 45.2},
      "feature_importance": {"f1": 0.31, "f2": 0.28},
      "hpo_trials": 50
    }
  ]
}
```
> Note: same train/val/test split across all models; metrics are RAW (not ranked).

---

### AGENT 4 — Experiment Manager  `[V1: thin · V2: full]`

**Purpose.** Aggregate metrics, rank models by the primary metric, package the top-N for the Critic, and own the retrain-loop counter.

**Responsibilities.**
- Aggregate test metrics from all trained models into one comparison table.
- Rank by `primary_metric`; expose secondary rankings (speed, size).
- Package top 3 (best + alternatives) for the Critic.
- Own `retrain_count`; on a Critic REJECT, coordinate re-ranking after retrain.
- Log runs/metrics/artifacts to MLflow.
- `[V2]` Pareto trade-off analysis + comparison visualizations.

**Tools Used.**
```python
# tools/ranking.py
def aggregate_and_rank(trained_models: list[ModelResult], primary_metric: str) -> dict:
    """Build comparison table, rank by primary metric. Returns {ranked_models, table}."""

def log_mlflow(run_id: str, trained_models: list[ModelResult], ranked: list[str]) -> None: ...
```
LLM: light narrative summary only.

**Input.** Reads `trained_models`, `primary_metric`.

**Output.**
```json
{
  "ranked_models": ["XGBoost", "LightGBM", "RandomForest"],
  "comparison_table": [
    {"rank": 1, "model": "XGBoost", "f1": 0.887, "infer_ms": 0.023, "size_mb": 45.2},
    {"rank": 2, "model": "LightGBM", "f1": 0.871, "infer_ms": 0.019, "size_mb": 12.3}
  ]
}
```

---

### AGENT 5 — Critic  `[V1: core — the differentiator + retrain loop]`

**Purpose.** Quality gate. Validate the top model for overfitting, data leakage, metric appropriateness, and validation soundness. Approve, or reject with actionable feedback and route back for a retrain.

**Responsibilities.**
- Overfitting: compare train/val/test gaps against family-calibrated thresholds.
- Data leakage: distribution tests (KS) between train/test; flag suspicious high-importance ID/date features; temporal integrity for forecasts.
- Metric appropriateness (e.g. reject accuracy-as-primary on imbalance).
- Validation strategy (stratification, temporal folds, test isolation, seed consistency).
- Assign severity; make a decision; generate constructive feedback.
- Emit the verdict that drives graph routing.

**Tools Used.**
```python
# tools/overfitting.py
def check_overfitting(metrics: dict, family: str) -> dict:
    """Gap analysis vs thresholds (None<0.03 / Mild 0.03-0.05 / Moderate 0.05-0.10 / Severe>0.10;
    calibrate per family). Returns {severity, gaps, detected: bool}."""

# tools/leakage.py
def check_leakage(cleaned_path: str, feature_importance: dict, family: str) -> dict:
    """KS distribution tests + suspicious-feature scan + temporal integrity.
    Returns {detected: bool, findings: [...]}."""

# tools/metric_check.py
def check_metric_choice(primary_metric: str, family: str, class_balance: dict) -> dict: ...

# tools/validation_check.py
def check_validation(plan: dict, family: str) -> dict: ...
```
LLM: synthesizes tool findings into a verdict + feedback text.

**Input.** Reads `ranked_models`, `trained_models`, `problem_family`, `primary_metric`, `plan`, `retrain_count`.

**Output.**
```json
{
  "critic_verdict": "reject",
  "critic_findings": [
    {"type": "overfitting", "severity": "severe", "evidence": {"train_f1": 0.99, "test_f1": 0.72, "gap": 0.27},
     "recommendation": "reduce max_depth; add regularization; retrain"}
  ]
}
```
Verdict ∈ `approve | reject | conditional | refer_alt`.

---

### AGENT 6 — Business / Impact Analyst  `[V2 — parameter-driven; V1 stub]`

**Purpose.** Translate results into plain language. In V1, ship a **stub** that emits qualitative one-liners only.

**Responsibilities (V2).**
- Translate metrics to plain language grounded in computed numbers (e.g. "recall 0.87 ⇒ ~87% of positives caught").
- Compute impact/ROI **only** from `business_params` supplied by the user. If none, stay qualitative and explicitly decline to invent figures.
- Feature-importance → business insight narrative.

**Tools Used.**
```python
# [V2] tools/impact.py
def compute_impact(metrics: dict, business_params: dict) -> dict:
    """Arithmetic on USER-SUPPLIED params ONLY. If business_params is None, return {} and a note.
    NEVER fabricate ARPU/revenue/ROI."""
```
LLM: writes narrative from grounded inputs only.

**Input.** Reads approved model `metrics`, `feature_importance`, `business_params`.

**Output (V1 stub).**
```json
{ "narrative": {"summary": "qualitative translation of metrics; no invented figures", "impact": null} }
```

---

### AGENT 7 — Chief Scientist  `[V1: thin · V2: full]`

**Purpose.** Final synthesis. Declare the winning model, explain why (from Critic + metrics only), and give a GO / GO-WITH-CONDITIONS / NO-GO decision with a confidence level and next steps.

**Responsibilities.**
- Confirm winner from ranked + Critic-approved models.
- Compose grounded rationale (evidence only — no invented benchmarks).
- Check success criteria from the plan; make the decision + confidence.
- `[V2]` deployment roadmap, resource plan, phased rollout.

**Tools Used.**
```python
# tools/ranking.py (reuse)
def success_criteria_check(metrics: dict, plan: dict) -> dict:
    """Pass/fail each success criterion. Returns scorecard."""
```
LLM: composes the decision from grounded inputs.

**Input.** Reads `ranked_models`, `critic_verdict`, `critic_findings`, approved model `metrics`, `plan`, `narrative`.

**Output.**
```json
{
  "decision": {
    "winner": "XGBoost",
    "rationale": ["best primary metric among approved", "passed all Critic checks"],
    "recommendation": "GO",
    "confidence": "high",
    "success_scorecard": {"primary_metric_target_met": true},
    "next_steps": ["monitor primary metric monthly", "retrain quarterly"]
  }
}
```

---

## 4. Orchestration & Graph Wiring (`graph.py`)

**Nodes:** the 7 agents + a terminal `FAILED` node.

**Edges (linear with one loop):**
```
research_planner → data_scientist → ml_engineer → experiment_manager → critic
critic ─(reject & retrain_count < MAX_RETRAINS)─────────────────────► ml_engineer   # increment retrain_count
critic ─(reject & retrain_count >= MAX_RETRAINS)──► experiment_manager (refer_alt, try next model once)
critic ─(approve | conditional)──► business_analyst → chief_scientist → END
any node on hard error ──► FAILED (write state["error"])
```

**Conditional routing function** reads `state["critic_verdict"]` and `state["retrain_count"]` (set `MAX_RETRAINS = 2` in `config.py`). On the retrain edge, increment `retrain_count` and pass Critic feedback so the ML Engineer adjusts.

**Human-in-the-loop** (`state["hitl"]`): optional interrupts after `research_planner` (confirm plan) and before `END` at `chief_scientist` (confirm GO/NO-GO). Default on for the product; off for benchmarking.

---

## 5. Build Order & Milestones (map to `PROGRESS.md`)

1. `state.py`, `config.py`, `llm/client.py`, `tools/engine_wrapper.py` scaffold.
2. **Data Scientist** — test on a real CSV; confirm cleaned data + candidate models come out.
3. **ML Engineer** — fit one AutoGluon `TabularPredictor` under a real `time_limit`; confirm per-model test metrics from the leaderboard.
4. **Experiment Manager** — rank real metrics.
5. **Critic + retrain loop** — *milestone tracked separately:* intentionally overfit a model, confirm the Critic catches it and the conditional edge routes back to ML Engineer, and that `MAX_RETRAINS` terminates the loop.
6. **Chief Scientist** — grounded GO/NO-GO.
7. **Research Planner** — build the conversational front-end that reliably produces the Data Scientist's expected input shape.
8. `[V2]` Business Analyst (parameter-driven), forecast/cluster/anomaly families, deployment/serving.

---

## 6. Acceptance Tests (`tests/`)

Write these so the pipeline self-verifies:

- **Family detection:** classification, regression, and (V2) time-series datasets each resolve to the correct `problem_family` and `primary_metric`.
- **Leakage safety:** injecting a target-leaking feature (e.g. an ID correlated with the target) is flagged by `check_leakage`.
- **No fabricated numbers:** with `business_params = None`, the Business Analyst output contains `impact: null` and no dollar figures anywhere.
- **Budget honored:** a tiny `wall_clock_s` still returns a valid best-so-far `ModelResult`.
- **Critic reject→retrain loop (headline test):** feed a deliberately overfit model → Critic verdict `reject` → graph routes to `ml_engineer` → `retrain_count` increments → loop terminates at `MAX_RETRAINS`.
- **End-to-end:** a small classification CSV runs Planner→…→Chief Scientist and produces a `decision` with a winner and GO/NO-GO.

---

### Reminder to Claude Code
Build V1 depth only (V2 items = stubs + TODO). Keep every agent's I/O exactly matching `LabState` field names. All numbers from tools. Fit transforms on train only. Enforce the budget and the retrain cap.
