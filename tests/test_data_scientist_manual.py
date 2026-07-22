"""Manual end-to-end validation of the Data Scientist agent against real
datasets. Not a pytest acceptance test (those come in §6 once the whole
pipeline exists) - this is the "test against real upstream output" step
required before moving to ML Engineer.

Research Planner (family detection) isn't built yet (built last per spec
build order), so family/target are supplied directly here as test-harness
inputs - not fabricated agent output.
"""

import json
import sys

sys.path.insert(0, ".")

from agents.data_scientist import data_scientist_node

CASES = [
    {"name": "loan (binary)", "dataset_path": "dataset/loan_dataset.csv", "target_column": "Loan_Status", "problem_family": "binary"},
    {"name": "iris (multiclass)", "dataset_path": "dataset/Iris.csv", "target_column": "Species", "problem_family": "multiclass"},
    {"name": "housing (regression)", "dataset_path": "dataset/Housing.csv", "target_column": "price", "problem_family": "regression"},
]


def run_case(case: dict, use_llm: bool) -> None:
    state = {
        "dataset_path": case["dataset_path"],
        "target_column": case["target_column"],
        "problem_family": case["problem_family"],
        "seed": 42,
        "run_id": f"manualtest_{case['name'].split()[0]}",
    }
    print(f"\n{'=' * 70}\n{case['name']}  (llm={use_llm})\n{'=' * 70}")

    if not use_llm:
        import agents.data_scientist as ds
        ds._propose_candidate_models = lambda *a, **k: [{"name": "STUB", "why": "llm skipped for this run"}]

    result = data_scientist_node(state)

    print("dataset_profile.shape:", result["dataset_profile"]["shape"])
    print("dataset_profile.quality_score:", result["dataset_profile"]["quality_score"])
    print("dataset_profile.missing_pct:", result["dataset_profile"]["missing_pct"])
    print("dropped identifier columns:", result["dataset_profile"]["cleaning_report"]["dropped_columns"])
    print("train/test rows:", result["dataset_profile"]["cleaning_report"]["train_rows"],
          result["dataset_profile"]["cleaning_report"]["test_rows"])
    print("cleaned_data_path:", result["cleaned_data_path"])
    print("cleaned_data_hash:", result["cleaned_data_hash"][:24], "...")
    print("feature_list (%d):" % len(result["feature_list"]), result["feature_list"])
    print("feature_selection.dropped:", result["dataset_profile"]["feature_selection"]["dropped"])
    print("candidate_models:")
    print(json.dumps(result["candidate_models"], indent=2))


if __name__ == "__main__":
    use_llm = "--llm" in sys.argv
    for case in CASES:
        run_case(case, use_llm)
