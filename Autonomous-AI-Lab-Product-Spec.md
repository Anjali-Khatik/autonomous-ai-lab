# Autonomous AI Lab — Product Specification (Refined)

> An agentic AutoML system: a LangGraph pipeline of reasoning agents that plans, builds, validates, and explains ML solutions for **any tabular dataset and problem type** — with a hard rule that every number comes from executed code, never from a language model.

This is a rewrite of the original design doc, restructured around problem-families instead of a single churn demo, with the orchestration, state schema, and LLM/code boundaries made explicit. Read the **Design Principles** and **V1/V2 Scope** sections first — they carry most of the decisions.

---

## 0. Positioning (why this exists, honestly)

The "train 3–5 models + tune + rank" loop is a **commodity**. AutoGluon, FLAML, H2O AutoML, Auto-sklearn, and PyCaret already do it faster and more robustly than a fresh implementation will. So the training engine is **not** the differentiator and should not be rebuilt from scratch.

The differentiator is the **agentic reasoning layer wrapped around** a proven engine:

1. **Planner** that interrogates the problem and dataset in natural language and produces a machine-checkable plan.
2. **Critic** — an explicit quality gate that catches overfitting and data leakage and *routes work back for a retrain*. This is the strongest story beat and the hardest thing for existing AutoML libraries to claim.
3. **Narrative layer** that translates results into plain language grounded strictly in computed evidence.

**Product thesis:** wrap a best-in-class AutoML engine for the modeling grunt work; spend engineering effort on Planner, Critic, and grounded narrative. Compete on *trust, explanation, and quality-gating*, not on raw leaderboard speed.

---

## 1. Design Principles (the cross-cutting fixes)

These apply to every agent and override anything below that seems to contradict them.

### P1 — Numbers come from code, never from the LLM
Any metric, count, dollar figure, or statistic must be produced by a deterministic tool and passed to the LLM as data. LLMs may *interpret* numbers; they may never *compute* or *invent* them. This kills the single biggest failure mode of the original doc (fabricated ROI like "$2.5M retained / 49x").

### P2 — Clear LLM-vs-deterministic boundary
| Layer | Who does it | Examples |
|-------|------------|----------|
| **Reasoning** (LLM) | open model via Fireworks (Llama/Qwen) | Choose candidate models, interpret EDA, write explanations, decide GO/NO-GO framing |
| **Execution** (code) | Python tools on ROCm/CPU | EDA stats, cleaning, training, HPO, metrics, leakage tests, serialization |

Every agent is an **LLM planner that calls trusted tools**. No agent computes results in its own text.

### P3 — Problem-family generality is a contract, not a claim
The pipeline must branch on a detected **problem family** (§2). Metrics, tools, and output schemas differ per family. "Works on any dataset" is only true if each family is actually implemented and tested. V1 implements two families end-to-end; V2 adds more.

### P4 — Wrap, don't reinvent, the training engine
The ML Engineer agent orchestrates an existing AutoML engine — **AutoGluon `TabularPredictor`** (chosen for stronger out-of-box tabular accuracy and built-in stacked ensembling) — rather than hand-rolling GridSearch/Optuna loops. It fits once under a `time_limit` and reads per-model results from the leaderboard. Your value is *around* it.

### P5 — Every run is compute-budget governed
No open-ended HPO. Every run declares a budget (wall-clock seconds, trial cap, and/or cost cap). The engine must stop and return the best-so-far when the budget is hit. This is a first-class product knob, not an afterthought.

### P6 — Deterministic reproducibility
Single global seed threaded through split, engine, and tools. Every artifact (cleaned data hash, feature list, model, metrics) is logged with lineage so a run can be replayed.

---

## 2. Problem Families (the generalization matrix)

The pipeline detects a **family** early and branches on it. This is the core of being a general product.

| Family | Detection signal | Primary metric | Secondary | Notes / gotchas |
|--------|------------------|----------------|-----------|-----------------|
| **Binary classification** | target has 2 classes | F1 (or PR-AUC if imbalanced) | ROC-AUC, recall, precision, MCC | Never use accuracy as primary on imbalance |
| **Multiclass classification** | target has 3–~50 discrete classes | macro-F1 | balanced accuracy, per-class recall | Watch rare classes |
| **Regression** | continuous numeric target | RMSE (or MAE if heavy outliers) | R², MAPE, RMSLE | MAPE breaks on zeros |
| **Time-series forecasting** | datetime index + ordered target | MAE / sMAPE | RMSE, coverage | **Must** use temporal split, never random |
| **Clustering** *(V2)* | no target column | silhouette | Davies-Bouldin, CH index | No ground truth → different validation path |
| **Anomaly detection** *(V2)* | no target / rare-positive framing | PR-AUC (if labels) | precision@k | Often semi-supervised |

**Task detection tool (deterministic + LLM confirm):** infer family from target dtype, cardinality, presence of a datetime index, and class balance; then have the Planner LLM *confirm or override* with a one-line rationale. This is the productized version of your existing `eda_toolkit` auto-inference — promote it to a first-class, tested component, not an assumption.

---

## 3. Shared Pipeline State (LangGraph)

One typed state object flows through the graph. Keep large artifacts on disk and pass **paths/hashes**, not objects, in state (fixes the original doc's non-serializable `<trained_model>`).

```python
class LabState(TypedDict):
    # --- run config ---
    run_id: str
    seed: int
    compute_budget: BudgetSpec          # {wall_clock_s, max_trials, cost_cap}
    business_params: dict | None        # user-supplied ONLY (ARPU, costs...) — never invented

    # --- planner output ---
    problem_family: Literal["binary","multiclass","regression","forecast","cluster","anomaly"]
    primary_metric: str
    plan: dict                          # objective, constraints, success criteria

    # --- data scientist output ---
    dataset_profile: dict               # shapes, dtypes, quality score, EDA summary
    cleaned_data_path: str
    cleaned_data_hash: str
    feature_list: list[str]
    candidate_models: list[dict]        # names + why (LLM), ranges optional

    # --- ml engineer output ---
    trained_models: list[ModelResult]   # each: name, path, metrics{train/val/test}, timings

    # --- experiment manager output ---
    ranked_models: list[str]

    # --- critic loop control ---
    critic_verdict: Literal["approve","reject","conditional","refer_alt"] | None
    critic_findings: list[dict]
    retrain_count: int                  # ENFORCE MAX_RETRAINS

    # --- final ---
    narrative: dict | None              # grounded explanation (V1)
    decision: dict | None               # GO / NO-GO + rationale
```

---

## 4. The Agents (refined)

Same seven agents as your build, kept consistent with the fixed build order (Data Scientist first for real-data testing, Research Planner last). Below each is reframed: **role → LLM vs tools → generalization note**. The V1/V2 column says how deep to build it now.

### 4.1 Research Planner  *(V1 — thin)*
- **Role:** turn a user objective + dataset into a checkable plan; detect problem family; set primary metric and success criteria.
- **LLM:** conducts the conversation, confirms/overrides task detection, writes constraints.
- **Tools:** task-detection tool, dataset quick-profiler.
- **Generalization:** the branch point. Output must set `problem_family` + `primary_metric` correctly for every family in §2.
- **V1 depth:** minimal viable — detect family, set metric, capture constraints. Skip elaborate stakeholder modeling.

### 4.2 Data Scientist  *(V1 — core)*
- **Role:** EDA → cleaning → feature work → candidate model shortlist.
- **LLM:** interprets the EDA report, proposes 3–5 candidate models with rationale.
- **Tools:** EDA (numeric/categorical/datetime), cleaning (impute/encode/scale), leakage-safe feature engineering, selection.
- **Generalization notes:**
  - **Make Feature Extraction (PCA/LDA/autoencoders) conditional, not default.** For the tree models this pipeline favors, dimensionality reduction usually *hurts*. Only trigger it when features are very high-dimensional or highly collinear.
  - Fit all transforms on **train only**; persist the fitted pipeline to avoid leakage.

### 4.3 ML Engineer  *(V1 — core, but wrap an engine)*
- **Role:** train and tune candidate models under the compute budget; return honest per-model metrics.
- **LLM:** almost none — this is an execution agent. Optionally narrates progress.
- **Tools:** **AutoGluon `TabularPredictor` wrapper** (fit-once under `time_limit`, read leaderboard) for training+ensembling under `compute_budget`; hardware profiler for ROCm/CPU tier; independent test-set evaluator; serializer (AutoGluon predictor dir + joblib) writing paths into state.
- **Generalization:** engine + metric set switch on `problem_family`. Same wrapper, different objective.
- **Key rule (P5):** honor the budget and return best-so-far on timeout.

### 4.4 Experiment Manager  *(V1 — thin; V2 — full)*
- **Role:** aggregate metrics, rank by primary metric, package top-N for the Critic, manage the retrain loop.
- **LLM:** none for ranking (deterministic); light narrative for the comparison summary.
- **Tools:** metrics aggregator, ranker (with optional Pareto trade-off), MLflow logging.
- **V1 depth:** rank + hand off + own the retrain counter. Defer fancy visualizations/Pareto to V2.

### 4.5 Critic  *(V1 — core; this is the differentiator)*
- **Role:** quality gate. Detect overfitting and data leakage; validate metric choice and validation strategy; APPROVE / REJECT→retrain / CONDITIONAL / REFER-TO-ALT.
- **LLM:** synthesizes tool findings into a verdict + constructive feedback.
- **Tools:** overfitting detector (train/val/test gap thresholds — keep the original's None<0.03 / Mild / Moderate / Severe>0.10 bands but **calibrate per family**), leakage detector (KS distribution tests, suspicious high-importance ID/date features, temporal integrity), metric-appropriateness checker, validation-strategy checker (stratification, temporal folds, test isolation).
- **Critic → ML Engineer retrain loop:** first-class. On REJECT, route back with specific feedback. **Enforce `MAX_RETRAINS` (e.g. 2)**; on exhaustion, escalate to REFER-TO-ALT or surface to the user. Test this loop by intentionally overfitting a model and confirming the Critic catches it — this is the demo's best moment.

### 4.6 Business / Impact Analyst  *(V2 — rebuilt, parameter-driven)*
- **Role:** translate results into plain-language impact.
- **Hard reframe (P1):** it may compute impact **only** from `business_params` the user supplied. If none are given, it stays **qualitative** ("recall of 0.87 means ~87% of positive cases are caught") and explicitly declines to invent dollar figures. The original doc's fabricated ARPU/ROI/revenue is removed — that's a liability in a product, not a feature.
- **LLM:** writes the narrative; a deterministic Impact Calculator does any arithmetic from user inputs.
- **Why V2:** most fragile, most hallucination-prone, least generalizable part. Ship the product without it first.

### 4.7 Chief Scientist  *(V1 — thin; V2 — full)*
- **Role:** final synthesis — declare the winning model, explain why (from Critic + metrics only), give a GO / GO-WITH-CONDITIONS / NO-GO with a confidence level, and list next steps.
- **LLM:** composes the decision from grounded inputs.
- **Tools:** model comparison summarizer, success-criteria checker.
- **V1 depth:** winner + grounded rationale + GO/NO-GO. Defer deployment roadmaps/resource plans to V2.

---

## 5. Orchestration & Control Flow

### Graph topology
```
Research Planner
      │
Data Scientist ──► ML Engineer ──► Experiment Manager ──► Critic
                        ▲                                    │
                        │        reject (retrain_count<MAX)  │
                        └────────────────────────────────────┤
                                                              │ approve / conditional
                                                              ▼
                                          Business Analyst (V2) ──► Chief Scientist ──► END
```

### Control rules
- **Loop termination (was missing):** `Critic` REJECT increments `retrain_count`; the conditional edge back to `ML Engineer` fires only while `retrain_count < MAX_RETRAINS`. On exhaustion → REFER-TO-ALT (try next-ranked model once) → else surface to user. No infinite loops.
- **Error handling:** every node wraps its tool calls; a hard failure writes an error to state and routes to a terminal `FAILED` node with a readable reason, never a silent hang.
- **Human-in-the-loop:** two defined checkpoints — (a) after Planner (confirm plan/family before spending compute), (b) at Chief Scientist GO/NO-GO. Both optional via a flag; default on for the product, off for automated benchmarking.

---

## 6. Cross-Cutting Subsystems

- **Data ingestion & validation (add — was implicit):** load CSV/Parquet, infer schema/dtypes, and decide a **sampling/out-of-core strategy** for large data (the original silently assumed everything fits in RAM). V1: in-memory with a row cap + sampling warning. V2: chunked/out-of-core.
- **Compute budget governor (P5):** owns the `BudgetSpec`, passes it to the engine, enforces stop conditions.
- **Experiment tracking & lineage (P6):** MLflow for runs; additionally log `cleaned_data_hash`, `feature_list`, and seed so any result is replayable.
- **Deployment/serving (explicitly deferred):** the Chief Scientist decides GO but the product does **not** deploy in V1. V2 adds a serving endpoint + drift monitoring. Don't let "DEPLOY" imply a serving path that doesn't exist yet.

---

## 7. V1 vs V2 Scope (build this order)

**V1 — "Trustworthy AutoML for tabular classification & regression."** The defensible MVP. Goal: end-to-end, genuinely general across two families, with the Critic loop as the headline.

| Component | V1 depth |
|-----------|----------|
| Problem families | **Binary + multiclass classification, and regression** — fully implemented & tested |
| Research Planner | Thin: detect family, set metric, capture constraints |
| Data Scientist | Core: EDA + leakage-safe cleaning + selection; extraction conditional/off |
| ML Engineer | Core: **wrap AutoGluon `TabularPredictor`** (fit-once under `time_limit`, read leaderboard) |
| Experiment Manager | Thin: aggregate + rank + own retrain counter |
| Critic | **Core + retrain loop end-to-end** (the differentiator) |
| Chief Scientist | Thin: winner + grounded rationale + GO/NO-GO |
| Business Analyst | **Excluded** (qualitative one-liners only, if anything) |
| Data ingestion | In-memory + row cap + sampling warning |
| Compute budget | Enforced |
| Tracking/lineage | MLflow + hashes + seed |
| Deployment | Decision only, no serving |

**V2 — "Depth, breadth, and impact."** Adds the differentiated depth and the risky parts once the core is proven.

| Component | V2 additions |
|-----------|--------------|
| Problem families | **Time-series forecasting** (temporal split), then **clustering + anomaly detection** |
| Business Analyst | Rebuilt, **parameter-driven**, grounded-only impact + ROI *from user inputs* |
| Chief Scientist | Full: deployment roadmap, resource plan, phased rollout |
| Experiment Manager | Pareto trade-off analysis + comparison visualizations |
| Data ingestion | Chunked / out-of-core for large datasets |
| Deployment | Serving endpoint + drift monitoring + retrain triggers |
| Modalities | (Stretch) text/image via feature extractors — only if a real use case pulls for it |

---

## 8. What to cut or defer, and why

- **The elaborate ROI/business-case machinery → V2, rebuilt.** It's the part most likely to impress in a demo and embarrass in a product, because it invents numbers. Grounded-only or nothing.
- **Feature extraction as a default stage → make conditional.** Hurts tree models; only justified for very high-dimensional/collinear data.
- **Hand-rolled HPO → replaced by an engine wrapper.** Don't spend V1 effort on a solved commodity.
- **Deployment execution → V2.** V1 stops at a defensible GO/NO-GO decision.
- **Clustering / anomaly / forecasting → phase in.** Forecasting first in V2 (highest demand), unsupervised families after.

---

### One-line summary
Keep the 7-agent skeleton and the Critic retrain loop; make generality real by branching on problem-family; draw a hard line so **all numbers come from code**; wrap an existing AutoML engine instead of rebuilding it; ship V1 as trustworthy tabular classification + regression with the Critic as the headline, and defer the fragile business/ROI and deployment layers to V2.
