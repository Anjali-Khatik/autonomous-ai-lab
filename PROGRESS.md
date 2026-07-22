# Autonomous AI Lab — Build Progress

> Update this at the end of each agent. Build order is fixed (spec §5).
> Each agent is built and tested against the **real** output of the previous one.
> Status: ⬜ not started · 🟡 in progress · ✅ done

_Last updated: 2026-07-20 · Current focus: spec §6 acceptance test suite written as real pytest tests, all 13 passing (fast + slow/LLM). V1 core build is functionally complete. HITL interrupt behavior still untested (live pause/resume) — that's application-layer/UI work._

## Foundation
- ✅ Repo scaffold (spec §0 structure)
- ✅ `state.py` — `LabState`, `BudgetSpec`, `ModelResult` (+ 4 fields added, see deviations log)
- ✅ `config.py` — seed, `FAMILY_METRICS`, `MODEL_REGISTRY`, `MAX_RETRAINS`
- ✅ `llm/client.py` — Fireworks `reason()` helper (model slug fixed, see deviations log)
- ✅ `tools/engine_wrapper.py` — AutoGluon `fit_predictor` + `read_leaderboard`

## Agents (fixed build order)
| # | Agent | Built | Tested vs real upstream | Notes / open issues |
|---|-------|-------|-------------------------|---------------------|
| 1 | Data Scientist | ✅ | ✅ | 3 tools (`run_eda`, `build_clean_pipeline`, `select_features`) + LLM candidate-model step, all wired into `agents/data_scientist.py`. Validated on 3 real datasets spanning all 3 V1 families: `dataset/loan_dataset.csv` (binary), `dataset/Iris.csv` (multiclass), `dataset/Housing.csv` (regression). Fully generalized — no dataset-specific logic anywhere. |
| 2 | ML Engineer (AutoGluon) | ✅ | ✅ | One `TabularPredictor.fit()` + `leaderboard()`-based read, per spec's API note (not a per-model loop). Tested end-to-end against Data Scientist's real output for all 3 datasets/families. Real overfitting gaps visible in the results already (e.g. housing: train r2 0.85–0.95 vs test r2 ~0.6) — good real input for the Critic milestone next. |
| 3 | Experiment Manager | ✅ | ✅ | `aggregate_and_rank` (direction-aware: max for f1/macro_f1, min for rmse) + `log_mlflow` (sqlite backend, see deviations log), wired into `agents/experiment_manager.py`. Tested end-to-end against real ML Engineer output for all 3 families — ranking order verified correct in each (f1/macro_f1 descending, rmse ascending; right model on top each time). LLM narrative step skipped for V1 — spec's own output contract (§3 Agent 4) has no field to put it in. |
| 4 | Critic | ✅ | ✅ | 4 tools (`check_overfitting`, `check_leakage`, `check_metric_choice`, `check_validation`) + LLM synthesis, wired into `agents/critic.py`. Safety-critical routing (severe finding -> reject/refer_alt) is deterministic in Python regardless of LLM output — LLM only adjudicates the approve-vs-conditional call and writes feedback text. Tested against real Experiment Manager output for all 3 datasets: loan -> `reject` (genuine severe overfit, train f1=1.0 vs test f1=0.89), iris -> `conditional` (see leakage fix below), housing -> `reject` (genuine severe r2 gap 0.85->0.62). |
| 5 | **Critic → ML Engineer retrain loop** | ✅ | ✅ | Full milestone now verified through the actual compiled graph's `invoke()` (not manual chaining): housing's `WeightedEnsemble_L2` genuinely, repeatedly overfits on this 545-row dataset — real `reject` fired, `prepare_retry` excluded the flagged learner, ml_engineer retried with a real behavioral change (`fit_weighted_ensemble=False`), and `LinearModel` (no longer competing against an ensemble) was approved on the very next Critic pass. Also separately verified the `refer_alt` path fires correctly when `MAX_RETRAINS` is hit and promotes the next-ranked model. See deviations log for 2 real bugs found and fixed while wiring this. |
| 6 | Business Analyst | ✅ (stub only) | ✅ | `[V2]` qualitative stub, `agents/business_analyst.py`. `tools/impact.py::compute_impact` left as a `[V2]` `NotImplementedError` stub — never called in V1. `impact` is always `null`, verified in BOTH directions: with `business_params=None` (stays qualitative) AND with real `business_params` supplied (still correctly declines, since no compute tool exists — doesn't fabricate just because params were given). Tested against real Critic output (Iris, `conditional` verdict). |
| 7 | Chief Scientist | ✅ | ✅ | `success_criteria_check` added to `tools/ranking.py` (reused per spec) + `agents/chief_scientist.py`. Same pattern as Critic: recommendation (GO/GO-WITH-CONDITIONS/NO-GO) and confidence are decided deterministically in Python from `critic_verdict` + the measured scorecard — LLM only writes grounded rationale/next_steps. 6 deterministic-logic cases verified directly, plus full real pipeline (Iris) end-to-end. |
| 8 | Research Planner | ✅ | ✅ | `detect_problem_family` + `quick_profile` (`tools/task_detection.py`) + `agents/research_planner.py`. LLM confirms/overrides family, constrained to the 3 V1-supported families only. `success_criteria` deliberately NOT LLM-generated (see deviations log) — stays empty unless the user supplies a real target via `constraints`. Family detection verified correct standalone for all 3 real datasets + a hard-stop check for no target column. |

## Graph & tests
- ✅ `graph.py` wired: 7 agent nodes + 2 small deterministic prep nodes (`prepare_retry`,
  `prepare_refer_alt` — needed to implement spec §4's routing, not in spec's own node list) +
  `failed` terminal node. Every node wrapped so a raised exception becomes `state["error"]` and
  routes to `failed` instead of crashing the graph (spec rule: hard error -> FAILED node).
- ✅ HITL checkpoints wired: `build_graph(hitl=True)` compiles with a `MemorySaver` checkpointer
  and `interrupt_after=["research_planner", "chief_scientist"]`. Compiles successfully; actually
  exercising a live interrupt/resume cycle (pause -> external review -> resume) is application-
  layer integration (future UI work), not yet manually tested end-to-end.
- ✅ Acceptance tests (spec §6) as a real `pytest` suite — all 6 required behaviors, all passing:
  - `test_family_detection.py` (5 tests, <1s) — real datasets + a synthetic time-series case
    confirming forecast is never fabricated, since it's unimplemented.
  - `test_leakage_detection.py` (3 tests, <1s) — a synthetic ID-shaped leaking feature IS flagged
    "high" severity; a contrast case (genuinely-informative dominant feature, same shape but not
    ID-named) is correctly NOT escalated — regression-guards the real bug found 2026-07-20.
  - `test_no_fabricated_numbers.py` (2 tests, real LLM, ~45s) — `impact: null` AND a regex scan
    for dollar figures in the narrative text itself, in both directions (no params / params
    supplied but uncomputable).
  - `test_budget_honored.py` (1 test, real fit, ~15s) — `wall_clock_s=5` still returns a valid
    `ModelResult`.
  - `test_end_to_end.py` (1 test, real graph, slow) — iris via `graph.invoke()`, real decision
    with winner + GO/GO-WITH-CONDITIONS/NO-GO.
  - `test_retrain_loop.py` (1 test, real graph, slow) — housing via `graph.invoke()`, asserts
    `retrain_count <= MAX_RETRAINS` (the safety-critical invariant) and that the loop actually
    engaged, not just terminated trivially.
  - Registered `slow`/`llm` pytest markers (`pytest.ini`) so `pytest -m "not slow and not llm"`
    gives a <10s sanity pass (13 tests, ~6 min total for the full suite incl. the 2 heavy ones).
  - Renamed 2 stray `def test_...` functions inside the earlier manual scripts
    (`test_chief_scientist_manual.py`, `test_research_planner_manual.py`) to `check_...` so
    `pytest`'s default `test_*.py` discovery only picks up the dedicated acceptance suite, not
    ad-hoc functions inside exploratory scripts.

## V1 exit criteria
- ✅ End-to-end run on a small **classification** CSV → GO/NO-GO decision — full raw-input-to-decision
  run (`user_objective` + `dataset_path` only) via the compiled graph's real `invoke()`, verified for
  both loan (binary, `GO`) and iris (multiclass, `GO-WITH-CONDITIONS`).
- ✅ End-to-end run on a small **regression** CSV → GO/NO-GO decision — housing now reaches a full
  Chief Scientist decision (`GO`, `LinearModel`) via the graph's real retrain loop, after the ensemble-
  exclusion fix (see below) let a genuinely different retry succeed.
- ✅ Retrain loop verified end-to-end with an intentionally overfit model — via real `graph.invoke()`,
  not manual chaining: housing's `WeightedEnsemble_L2` rejected for genuine severe overfitting,
  `prepare_retry` correctly excluded it, ml_engineer's retry was measurably different
  (`fit_weighted_ensemble=False`), and the retry passed Critic clean. `refer_alt` path (MAX_RETRAINS
  cap -> exclude model -> promote next-ranked) also separately verified.

## Decisions / deviations log
- 2026-07-15: Added `dataset_path`, `user_objective`, `target_column`, `constraints` to
  `LabState` (state.py). Spec §1 omitted them, but §3 Agent 1's own input contract requires
  all four — Research Planner can't run without a place to read them from state. Confirmed
  with user before adding.
- 2026-07-19: `llm/client.py`'s default model slug (`llama-v3p1-70b-instruct`) 404'd — not
  deployed on this Fireworks account. Switched default to `accounts/fireworks/models/kimi-k2p6`,
  the slug validated working in the prior SGEMM project. Override via `FIREWORKS_MODEL` env var.
- 2026-07-19: `cleaned_data_path` (spec §1/§3) is a DIRECTORY, not a single file — contains
  `train.parquet`, `test.parquet` (leakage-safe split, done at Data Scientist stage since that's
  the one place transforms are fit — rule §3), and `pipeline.joblib`. Spec doesn't say where the
  train/test split happens, only that ML Engineer needs "a separate untouched test split ...
  never passed to fit()". Doing it here means every downstream agent reads the same split from
  one well-known path instead of re-deriving it (this was flagged as a real bug source in the
  prior SGEMM project's Critic — see that project's history). Does not change any `LabState`
  field name, so treated as an implementation clarification rather than a spec gap requiring
  sign-off.
- 2026-07-19: `build_clean_pipeline`/`data_scientist_node` take an optional `run_id` param (not
  in the spec's function signature) purely to control output directory naming for repeat test
  runs; defaults to a generated uuid so the spec'd 4-arg call signature still works unchanged.
- 2026-07-19: Removed `CatBoost` from `config.MODEL_REGISTRY`. It's not in `requirements.txt` /
  not installed in this venv — restricting AutoGluon's `hyperparameters` to an uninstalled
  learner's key hard-fails `fit()`. Candidate list now only offers models actually installed
  (XGBoost, LightGBM, RandomForest, Logistic/LinearRegression). Also confirmed `torch` is
  absent from this venv despite the kickoff prompt's claim that "ROCm PyTorch ... are already
  installed" — not true here. Doesn't block V1: the tree/linear models used don't need torch,
  so `hyperparameters` is always restricted away from NN_TORCH/FASTAI/CAT.
- 2026-07-19: `fit_predictor` carves its own train/val split out of Data Scientist's
  `train.parquet` (80/20, stratified for classification) and passes the val split to
  AutoGluon's `fit(tuning_data=...)`. Necessary because AutoGluon doesn't expose which rows
  it validated on if you let it auto-holdout — there'd be no way to honestly report a "val"
  entry in `ModelResult` otherwise. `read_leaderboard` locates this val split (and the
  train-fit split) via a `train_fit.parquet`/`val.parquet` naming convention next to
  `predictor/` under `outputs/ml_engineer/<run_id>/` — not part of the spec's function
  signature, but doesn't require any new `LabState` field either.
- 2026-07-19: **Real bug caught and fixed** — AutoGluon's leaderboard/`.score()` returns
  regression error metrics (rmse, mae, mape, rmsle, mse) NEGATED, so that "higher score =
  better" holds uniformly across its internal ranking regardless of metric direction (verified
  directly: calling the rmse scorer on a toy array returned -0.5477, not +0.5477). r2 is
  unaffected (already higher-is-better). Fixed in `tools/engine_wrapper.py::split_metrics` by
  taking `abs()` of the known negated metric names before returning — first real run without
  this fix reported `rmse: -662301.23` for the housing dataset, which would have silently
  corrupted every downstream ranking/overfitting comparison.
- 2026-07-19: Added `outputs/` to `.gitignore` — AutoGluon predictor directories contain
  per-model binary artifacts (`model.pkl`, `xgb.ubj`, etc.), not meant for git.
- 2026-07-20: Added `comparison_table` to `LabState` (experiment manager section). Same class
  of gap as the Research Planner fields on 2026-07-15 — spec §3 Agent 4's own output example
  includes it, spec §1 doesn't. Same resolution (add now, log here) per established precedent,
  not re-asked since it's the same category of decision already made once.
- 2026-07-20: **Real bug caught and fixed** — `log_mlflow` failed on every run with mlflow
  3.14.0: plain filesystem tracking (`./mlruns`) is deprecated in mlflow 3.x and raises unless
  `MLFLOW_ALLOW_FILE_STORE=true` is set. Fixed by pointing `mlflow.set_tracking_uri` at a local
  sqlite backend (`sqlite:///outputs/mlflow.db`) instead. Verified the fix actually persists
  runs (not just "no exception") via `mlflow.search_runs()` after a real run.
- 2026-07-20: **Real bug caught and fixed** — `check_leakage`'s "suspicious high-importance
  feature" heuristic (share > 0.5) falsely flagged Iris's top model for likely leakage, because
  `PetalLengthCm` legitimately has ~100% importance once feature selection leaves only 3
  features (petal length genuinely separates Iris species almost perfectly — well-known, not a
  data problem). This forced an incorrect `reject`. Spec §3 Agent 5 actually asks this check to
  flag "suspicious high-importance ID/date FEATURES" specifically, not any dominant feature.
  Fixed: `tools/leakage.py` now only escalates to "high" severity (which forces reject) when
  the dominant feature's name also looks like an identifier/date/timestamp; a dominant feature
  that doesn't match is downgraded to "moderate" (surfaced for review via `conditional`, not an
  automatic reject). Re-verified: Iris now correctly gets `conditional` with an honest "likely
  genuine signal, not leakage" note; loan's `Credit_History` (76% importance, also not ID/date-
  shaped) is now correctly treated the same way — loan still rejects, but for the right reason
  (genuine severe overfitting alone), not a false leakage flag.
- 2026-07-20: Critic doesn't persist the LLM's feedback paragraph into `critic_findings` —
  spec §3 Agent 5's own output example has no narrative/feedback field on a finding (only
  type/severity/evidence/recommendation). Same reasoning as Experiment Manager's narrative
  skip. Printed to console for visibility during manual testing only.
- 2026-07-20: `success_criteria_check(metrics, plan, primary_metric)` takes a 3rd arg beyond
  spec §3's `(metrics, plan)` — needed to know which key in `metrics` the
  `success_criteria["primary_metric_target"]` number refers to. Same class of minimal, necessary
  signature addition as `run_id` on `fit_predictor`/`build_clean_pipeline`.
- 2026-07-20: **Real finding, not a bug** — Critic's approve-vs-conditional LLM call is NOT
  perfectly deterministic across separate runs even at `temperature=0`: two runs of the exact
  same Iris pipeline (same real finding: `PetalLengthCm` at 100% importance, confirmed not
  ID/date-shaped) returned `approve` once and `conditional` once. Chief Scientist's own logic
  was correct both times (GO/high vs GO-WITH-CONDITIONS/medium, matching whatever verdict it
  actually received) — the test bug was hardcoding an assumed upstream verdict; fixed the test
  to assert the recommendation-matches-verdict invariant instead of a fixed value. Worth keeping
  in mind for the demo: don't rehearse an exact expected verdict for a borderline case, since it
  can genuinely go either way run-to-run. The severe/high-severity -> reject/refer_alt path
  stays deterministic regardless (that logic lives in Python, not the LLM).
- 2026-07-20: Research Planner's `plan.success_criteria` deliberately never gets an LLM- or
  policy-invented default target (e.g. "f1 >= 0.85"). Considered a config-level default (like
  `OVERFIT_THRESHOLDS`) but rejected it: unlike overfitting gaps, a "good" primary_metric value
  is either dataset-context-dependent (rmse is unbounded/scale-dependent — no universal number
  makes sense) or a genuine business judgment call nobody asked this system to make for
  bounded metrics either. `success_criteria` only gets `primary_metric_target` if the user
  supplies `constraints["success_metric_target"]` explicitly; otherwise it's `{}` and Chief
  Scientist's scorecard has nothing to check, so GO/NO-GO rests on `critic_verdict` alone.
  Verified both paths: iris with no target -> empty scorecard, GO-WITH-CONDITIONS; iris with a
  user-supplied target (0.5) -> scorecard shows `primary_metric_target_met: true`, GO/high.
- 2026-07-20: **First full raw-input pipeline runs** (`user_objective` + `dataset_path` only, no
  manually-supplied `problem_family`) surfaced no new bugs — every prior fix held up under the
  real end-to-end path. Housing's regression run showed a milder (but still severe) r2 gap than
  earlier Critic-stage testing (train r2 0.82 vs 0.65, not 0.86 vs 0.62) because this run let
  Data Scientist's LLM pick candidate models for real instead of using the test harness's fixed
  3-model stub — different candidate models -> different AutoGluon leaderboard -> different
  numbers. Expected, not a bug: confirms the earlier stub was never masking anything.
- 2026-07-20: **Design decision, confirmed with user first** — ML Engineer's retrain adjustment
  (how it actually changes anything given the one-global-seed rule) works by excluding the
  specific AutoGluon learner Critic flagged for overfitting from the next `fit()` call, rather
  than retraining identically and just re-rolling the dice against the same fixed seed. Requires
  2 new `LabState` fields (`excluded_models`, `excluded_learner_keys`) purely for graph-loop
  bookkeeping — not in spec, flagged before adding since this was a genuine design fork with
  real tradeoffs, not a copy-paste contract-completeness gap like earlier field additions.
- 2026-07-20: **Real bug caught and fixed** — the "exclude the flagged learner" mechanism above
  initially did nothing: `graph.py` looked up the Critic-flagged model name (e.g.
  `"WeightedEnsemble_L2"`, `"LinearModel"` — AutoGluon's own LEADERBOARD names) in
  `AUTOGLUON_MODEL_KEY`, which actually maps a DIFFERENT namespace (Data Scientist's candidate
  proposal names like `"XGBoost"`/`"LinearRegression"` -> the hyperparameters key used to
  restrict `fit()`). `.get()` silently returned `None` every time, so `excluded_learner_keys`
  stayed `[]` through a full real run even though the retrain loop still eventually terminated
  correctly (via the separate `refer_alt` path) — the adjustment feature the user explicitly
  asked for was quietly a no-op. Fixed: added `config.AUTOGLUON_LEADERBOARD_PREFIX_TO_KEY` (a
  real reverse mapping, prefix-matched since AutoGluon suffixes names like `_BAG_L1`/`_2`) and
  `leaderboard_name_to_key()`. Caught specifically because this was the first real graph-level
  test — none of the earlier manual per-agent tests exercised this code path at all.
- 2026-07-20: `"WeightedEnsemble_*"` isn't a single learner — it's AutoGluon's own auto-built
  stack combining every base model — so it can't be excluded via `hyperparameters` the way a real
  learner can. Added `ENSEMBLE_SENTINEL` in `config.py`: when the flagged model resolves to it,
  `fit_predictor` now passes `fit_weighted_ensemble=False` to `fit()` instead of a hyperparameters
  exclusion. This was the actual real-world case that fired in testing (housing's flagged model
  WAS the ensemble both times) — verified the fix works: after excluding it, `LinearModel` (no
  longer competing against an ensemble of everything) became the new top-ranked model and passed
  Critic clean on the very next pass, resolving in 1 retry instead of exhausting the full cap.
