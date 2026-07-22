"""Global config: seed, budget defaults, family->metric/model mappings.

Numbers here are pipeline policy (thresholds, defaults), not measured
results — those always come from tools at runtime.
"""

DEFAULT_SEED = 42

MAX_RETRAINS = 2

DEFAULT_BUDGET = {
    "wall_clock_s": 300,
    "max_trials": None,
    "cost_cap_usd": None,
}

# family -> {primary_metric, secondary metrics, validation split strategy}
FAMILY_METRICS = {
    "binary": {
        "primary_metric": "f1",  # pr_auc if imbalanced — decided by check_metric_choice
        "secondary": ["roc_auc", "recall", "precision", "mcc"],
        "validation_split": "stratified_kfold",
    },
    "multiclass": {
        "primary_metric": "macro_f1",
        "secondary": ["balanced_acc", "per_class_recall"],
        "validation_split": "stratified_kfold",
    },
    "regression": {
        "primary_metric": "rmse",  # mae if outlier-heavy
        "secondary": ["r2", "mape", "rmsle"],
        "validation_split": "kfold",
    },
    # [V2]
    "forecast": {
        "primary_metric": "mae",  # or smape
        "secondary": ["rmse", "coverage"],
        "validation_split": "temporal",
    },
    "cluster": {
        "primary_metric": "silhouette",
        "secondary": ["davies_bouldin", "ch_index"],
        "validation_split": None,
    },
    "anomaly": {
        "primary_metric": "pr_auc",  # if labels available
        "secondary": ["precision_at_k"],
        "validation_split": "semi_supervised",
    },
}

# family -> candidate model names the Data Scientist may shortlist from.
# ML Engineer maps these to AutoGluon `hyperparameters` restrictions.
# NOTE: CatBoost intentionally excluded — not in requirements.txt / not installed in this
# venv. Restricting fit() hyperparameters to an uninstalled learner's key would hard-fail
# the run, so the candidate list only offers models actually available. See PROGRESS.md
# deviations log (2026-07-19).
MODEL_REGISTRY = {
    "binary": ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression"],
    "multiclass": ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression"],
    "regression": ["XGBoost", "LightGBM", "RandomForest", "LinearRegression"],
    # [V2]
    "forecast": [],
    "cluster": [],
    "anomaly": [],
}

# candidate_models name -> AutoGluon `hyperparameters` dict key.
# Deliberately excludes NN_TORCH/FASTAI (require torch, not installed — V1 is CPU tree/linear
# models only) and CAT (catboost not installed).
AUTOGLUON_MODEL_KEY = {
    "XGBoost": "XGB",
    "LightGBM": "GBM",
    "RandomForest": "RF",
    "LogisticRegression": "LR",
    "LinearRegression": "LR",
}

# AutoGluon LEADERBOARD model name prefix -> the hyperparameters key that produced it.
# NOT the same mapping as AUTOGLUON_MODEL_KEY above: that maps OUR candidate-model
# proposal names (Data Scientist's "XGBoost"/"LinearRegression") to the key used to
# RESTRICT fit(); this maps AutoGluon's own leaderboard model names (e.g. "LinearModel",
# "WeightedEnsemble_L2" — different strings, sometimes with a "_BAG_L1"/"_2" suffix) back
# to that key, needed when the graph's retrain loop wants to exclude whichever specific
# learner Critic flagged. "WeightedEnsemble" isn't a single learner — it's AutoGluon's
# own auto-built stack of everything else — so it maps to the ENSEMBLE_SENTINEL instead
# of a hyperparameters key; fit_predictor interprets that sentinel as
# fit_weighted_ensemble=False on retry, not a hyperparameters exclusion.
ENSEMBLE_SENTINEL = "ENSEMBLE"
AUTOGLUON_LEADERBOARD_PREFIX_TO_KEY = {
    "XGBoost": "XGB",
    "LightGBM": "GBM",
    "RandomForest": "RF",
    "LinearModel": "LR",
    "WeightedEnsemble": ENSEMBLE_SENTINEL,
}


def leaderboard_name_to_key(model_name: str) -> str | None:
    """Resolve an AutoGluon leaderboard model name to its excludable key
    (a hyperparameters key, or ENSEMBLE_SENTINEL). None if unrecognized.
    """
    for prefix, key in AUTOGLUON_LEADERBOARD_PREFIX_TO_KEY.items():
        if model_name.startswith(prefix):
            return key
    return None

# family -> AutoGluon problem_type string
AUTOGLUON_PROBLEM_TYPE = {
    "binary": "binary",
    "multiclass": "multiclass",
    "regression": "regression",
}

# our primary_metric name -> AutoGluon eval_metric string
AUTOGLUON_METRIC = {
    "f1": "f1",
    "macro_f1": "f1_macro",
    "rmse": "root_mean_squared_error",
    "mae": "mean_absolute_error",
}

# primary_metric name -> ranking direction. Used by Experiment Manager to sort the
# leaderboard correctly regardless of family (higher-is-better vs lower-is-better).
METRIC_DIRECTION = {
    "f1": "max",
    "macro_f1": "max",
    "rmse": "min",
    "mae": "min",
    # [V2]
    "smape": "min",
    "silhouette": "max",
    "pr_auc": "max",
}

# Critic overfitting severity thresholds, on primary-metric train/test gap
# (calibrate per family once real baselines exist; these are starting defaults).
OVERFIT_THRESHOLDS = {
    "none": 0.03,
    "mild": 0.05,
    "moderate": 0.10,
    # >0.10 => severe
}
