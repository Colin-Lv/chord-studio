"""Versioned HGB artifact helpers for the independent v2 experiment.

Changing representation requires a newly fitted artifact. A v1 joblib or a
contract with different feature semantics is always rejected, even if its
matrix width happens to match.
"""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiment_protocol_v2 import signature, write_new_json
from multisource_features_v2 import SCHEMA as FEATURE_SCHEMA, DIMENSION, FEATURE_NAMES

SCHEMA = "multisource_hgb_maskable_full_semantics_v2"
CONTRACT_SCHEMA = "multisource_hgb_deployment_contract_v2"
HGB_PARAMETERS = dict(max_iter=160, max_leaf_nodes=15, learning_rate=.06,
    min_samples_leaf=30, l2_regularization=3., max_bins=63,
    early_stopping=False, random_state=20260926)
PINNED_SOURCES = ("chord_semantics_v2.py", "multisource_candidates_v2.py", "candidate_targets_v2.py",
    "multisource_features_v2.py", "multisource_model_v2.py", "experiment_protocol_v2.py",
    "harmonic_note_features.py", "nnls_seventh_model.py", "chord_extension_model.py", "chord_recognizer.py",
    "candidate_data_v2.py", "multisource_root_guard.py", "multisource_root_guard_v2.py",
    "evaluate_chords.py", "apply_multisource_candidates_v2.py")


def implementation_identity():
    return {name: signature(Path(__file__).with_name(name))["sha256"] for name in PINNED_SOURCES}


def fit_hgb(features, targets, sample_weights):
    features, targets, weights = np.asarray(features), np.asarray(targets), np.asarray(sample_weights)
    if (features.ndim != 2 or features.shape[1] != DIMENSION or targets.shape != (len(features),)
            or weights.shape != targets.shape or not len(features)
            or not np.isfinite(features).all() or not np.isfinite(targets).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or np.any((targets < 0) | (targets > 1))):
        raise ValueError("Finite v2 feature rows, utilities in [0,1] and positive weights required")
    with threadpool_limits(limits=2):
        return HistGradientBoostingRegressor(**HGB_PARAMETERS).fit(features, targets, sample_weight=weights/weights.mean())


def save_artifact(path, model, *, objective, margin, provenance, nnls_contract,
                  selection_config=None, diagnostic_only=True):
    path = Path(path)
    contract_path = path.with_suffix(".contract.json")
    if path.exists() or contract_path.exists():
        raise FileExistsError("Never overwrite an existing model or contract")
    if getattr(model, "n_features_in_", None) != DIMENSION or not np.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("Model feature width/margin invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(schema=SCHEMA, feature_schema=FEATURE_SCHEMA, feature_dimension=DIMENSION,
        feature_names=list(FEATURE_NAMES), objective=objective, margin=margin,
        implementation=implementation_identity(), training=provenance,
        nnls_contract=nnls_contract, selection_config=selection_config,
        diagnostic_only=diagnostic_only, calibrated=False, selection_policy="legacy_postselection_root_guard",
        versions={name: metadata.version(name) for name in ("scikit-learn", "numpy", "mir_eval", "joblib")})
    # One canonical metadata representation in both joblib and JSON: policies
    # contain tuples, which JSON would otherwise silently turn into lists.
    artifact = json.loads(json.dumps(artifact, ensure_ascii=False, allow_nan=False))
    artifact["model"] = model
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(artifact, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    contract = {key: value for key, value in artifact.items() if key != "model"}
    contract.update(schema=CONTRACT_SCHEMA, artifact_schema=SCHEMA, model_sha256=signature(path)["sha256"])
    write_new_json(contract_path, contract)
    return signature(path), signature(contract_path)


def load_artifact(path):
    path = Path(path)
    contract = json.loads(path.with_suffix(".contract.json").read_text(encoding="utf-8"))
    if (contract.get("schema") != CONTRACT_SCHEMA or contract.get("artifact_schema") != SCHEMA
            or contract.get("feature_schema") != FEATURE_SCHEMA or contract.get("feature_dimension") != DIMENSION
            or contract.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Not a compatible v2 model/feature contract; v1 weights cannot be reused")
    if signature(path)["sha256"] != contract["model_sha256"]:
        raise ValueError("Model bytes changed")
    if contract["implementation"] != implementation_identity():
        raise ValueError("V2 source semantics changed; retrain and issue a new contract")
    for name in ("scikit-learn", "numpy", "mir_eval"):
        if contract["versions"][name] != metadata.version(name):
            raise ValueError(f"Model runtime version mismatch: {name}")
    artifact = joblib.load(path)
    for key, value in contract.items():
        if key not in {"schema", "artifact_schema", "model_sha256"} and artifact.get(key) != value:
            raise ValueError(f"Artifact/contract disagree: {key}")
    if artifact.get("schema") != SCHEMA or getattr(artifact["model"], "n_features_in_", None) != DIMENSION:
        raise ValueError("Artifact is not a fitted v2 HGB model")
    return artifact, contract


def score_candidates(artifact, features):
    if (artifact.get("schema") != SCHEMA or artifact.get("feature_schema") != FEATURE_SCHEMA
            or artifact.get("feature_dimension") != DIMENSION or artifact.get("feature_names") != list(FEATURE_NAMES)
            or features.ndim != 2 or features.shape[1] != DIMENSION):
        raise ValueError("V2 inference cannot accept v1 feature matrices or artifacts")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite v2 features")
    if not len(features):
        return np.empty(0)
    with threadpool_limits(limits=2):
        return np.clip(artifact["model"].predict(features), 0., 1.)
