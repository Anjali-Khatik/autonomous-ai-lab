"""Artifact registry for trained AutoGluon predictors."""

import json
import os

REGISTRY_PATH = "outputs/ml_engineer/registry.jsonl"


def register_predictor(predictor_path: str, meta: dict) -> str:
    """Append a record of this run's predictor dir + metadata to the run
    artifact store. Returns predictor_path unchanged (the path is already
    the canonical artifact location; this just records it for lookup).
    """
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    record = {"predictor_path": predictor_path, **meta}
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return predictor_path
