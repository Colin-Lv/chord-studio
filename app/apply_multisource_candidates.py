"""Apply a frozen three-source candidate model without references or fitting.

Ship the model's .contract.json sidecar with its .joblib file. The contract
pins feature code, NNLS transform/binaries and training audio content identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import harmonic_note_features as notes
from analyze_multisource_candidate_potential import chord_identity
from evaluate_chords import read_reference, validate_timeline
from train_multisource_candidate_reliability import (SCHEMA, DIMENSION,
    automatic_candidate_features, predict_song)
from train_multisource_seventh_gate import signature
from multisource_root_guard import preserve_raw_root

NNLS_CONTRACT_KEYS = ("sample_rate", "block_size", "hop_length", "right_padding_samples",
    "sonic_sha256", "plugin_sha256", "decoder_sha256", "decoder_implementation_sha256",
    "bridge_implementation_sha256", "transform_id", "transform_sha256", "transform")


def contract_path(model_path):
    return Path(model_path).with_suffix(".contract.json")


def build_model_contract(model_path, training_report_path):
    """Packaging helper only: use training provenance, never label files."""
    model_path, training_report_path = Path(model_path), Path(training_report_path)
    artifact = joblib.load(model_path)
    model_signature = signature(model_path)
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if model_signature["sha256"] not in {report.get(key, {}).get("sha256") for key in ("development_model", "refit_model")}:
        raise ValueError("Training report does not identify this frozen model")
    training_ids = set(artifact["training_song_ids"])
    inputs = [item for item in report["inputs"] if item["id"] in training_ids]
    if {item["id"] for item in inputs} != training_ids:
        raise ValueError("Training report lacks fitted song identities")
    expected = None
    for item in inputs:
        cache = Path(item["notes"]["path"])
        if signature(cache)["sha256"] != item["notes"]["sha256"]:
            raise ValueError("Training note cache changed before model packaging")
        with np.load(cache, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        current = {key: metadata["identity"][key] for key in NNLS_CONTRACT_KEYS}
        if expected is not None and current != expected:
            raise ValueError("Training used inconsistent NNLS extraction contracts")
        expected = current
    if expected is None:
        raise ValueError("A deployment evidence contract requires fitted training provenance")
    value = dict(schema="multisource_candidate_deployment_contract_v1", model_sha256=model_signature["sha256"],
        feature_schema=artifact["schema"], feature_dimension=artifact["feature_dimension"],
        method=artifact["method"], margin=artifact["margin"],
        implementation=artifact["protocol"]["implementation"], nnls=expected,
        training_audio=[dict(id=item["id"], **item["audio"]) for item in inputs],
        training_report=signature(training_report_path),
        source_order=["raw_madmom", "chordino", "ISMIR2019_largvocab"],
        external_source_limitation="Original legacy training LAB generation identity remains unverified")
    path = contract_path(model_path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Do not overwrite a different model contract")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_frozen_artifact(model_path):
    model_path = Path(model_path)
    contract = json.loads(contract_path(model_path).read_text(encoding="utf-8"))
    if signature(model_path)["sha256"] != contract.get("model_sha256"):
        raise ValueError("Model bytes do not match deployment contract")
    artifact = joblib.load(model_path)
    if (artifact.get("schema") != SCHEMA or artifact.get("feature_dimension") != DIMENSION
            or contract.get("feature_schema") != SCHEMA or contract.get("feature_dimension") != DIMENSION
            or artifact.get("method") != contract.get("method") or artifact.get("margin") != contract.get("margin")):
        raise ValueError("Frozen model schema/method/margin contract mismatch")
    if artifact["method"] not in {"hgb_regression", "ridge_regression"}:
        raise ValueError("This deployment contract expects a fitted candidate-reliability model")
    if (getattr(artifact["model"], "n_features_in_", None) != DIMENSION
            or not np.isfinite(artifact["margin"]) or not 0 <= artifact["margin"] <= 1):
        raise ValueError("Invalid model feature dimension/margin")
    if contract["implementation"] != artifact["protocol"]["implementation"]:
        raise ValueError("Artifact and sidecar code contracts differ")
    for implementation in contract["implementation"]:
        current = Path(__file__).with_name(Path(implementation["path"]).name)
        if signature(current)["sha256"] != implementation["sha256"]:
            raise ValueError(f"Frozen feature implementation changed: {current.name}")
    if not any(Path(item["path"]).name == "harmonic_note_features.py" for item in contract["implementation"]):
        raise ValueError("Model contract does not pin NNLS feature implementation")
    return artifact, contract


def validate_note_contract(evidence, contract, audio_sha256):
    notes._validate_evidence(evidence)
    identity = evidence.metadata["identity"]
    if {key: identity.get(key) for key in NNLS_CONTRACT_KEYS} != contract["nnls"]:
        raise ValueError("NNLS features differ from the model's frozen extraction contract")
    if identity.get("source_sha256") != audio_sha256 or identity.get("offset_seconds") != 0 or identity.get("requested_duration_seconds") is not None:
        raise ValueError("NNLS evidence must belong to this complete source audio at offset zero")


def validate_root_policy(policy_path, artifact, model_path):
    """Optional published decision record; never read dataset model scores."""
    if policy_path is None:
        return None
    identity = signature(policy_path)
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    decision = policy.get("additional_root_policy", {})
    if (policy.get("selected_model_sha256") != signature(model_path)["sha256"]
            or policy.get("frozen_method") != artifact["method"]
            or policy.get("frozen_margin") != artifact["margin"]
            or decision.get("name") != "preserve_raw_root"
            or decision.get("development_outcome", {}).get("eligible") is not True):
        raise ValueError("Published root-protection policy differs from the frozen model or eligibility")
    return identity


def lab_provenance(path, audio_identity, role=None, expected_duration=None):
    """Check supplied sidecars if present; absence stays explicitly unverified."""
    path = Path(path)
    lab_identity = signature(path)
    result = dict(lab=lab_identity, status="caller_supplied_pairing_without_original_generation_identity")
    sidecars = [candidate for candidate in (path.with_name(path.name + ".provenance.json"),
                                           path.with_suffix(".provenance.json")) if candidate.is_file()]
    if not sidecars:
        return result
    evidence = []
    for sidecar in sidecars:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        source = record.get("audio", record.get("source_audio", record.get("source", {})))
        source_hash = source.get("sha256") if isinstance(source, dict) else None
        source_hash = source_hash or record.get("audio_sha256") or record.get("source_sha256")
        output = record.get("lab", record.get("output_lab", record.get("output", {})))
        output_hash = output.get("sha256") if isinstance(output, dict) else None
        if source_hash and source_hash != audio_identity["sha256"]:
            raise ValueError(f"External LAB belongs to different audio bytes: {path}")
        if output_hash and output_hash != lab_identity["sha256"]:
            raise ValueError(f"External LAB changed since its provenance record: {path}")
        for field in ("offset", "audio_offset", "offset_seconds"):
            if record.get(field) is not None and float(record[field]) != 0:
                raise ValueError(f"External LAB uses a nonzero audio offset: {path}")
        full_window_declared = False
        for field in ("requested_duration", "requested_duration_seconds", "duration_limit"):
            if field in record:
                if record[field] is not None:
                    raise ValueError(f"External LAB was generated from a limited-duration window: {path}")
                full_window_declared = True
        item = dict(sidecar=signature(sidecar), source_sha_verified=bool(source_hash),
                    output_sha_verified=bool(output_hash))
        sample_rate = record.get("sample_rate", 44100)
        if record.get("configuration_sha256"):
            report_path = path.parent.parent / "report.json"
            configuration_report = json.loads(report_path.read_text(encoding="utf-8"))
            configuration = configuration_report["configuration"]
            config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
            if config_hash != record["configuration_sha256"] or config_hash != configuration_report["configuration_sha256"]:
                raise ValueError("Chordino toolchain configuration SHA mismatch")
            if (configuration.get("offset") != 0 or configuration.get("duration_limit", "absent") is not None
                    or configuration.get("sample_rate") != 44100 or configuration.get("channels") != 1):
                raise ValueError("Chordino source was not complete 44100 Hz mono audio at offset zero")
            if configuration.get("transform") != "vamp:nnls-chroma:chordino:simplechord":
                raise ValueError("Unexpected explicit Chordino transform")
            for tool in ("sonic", "plugin", "ffmpeg", "code"):
                expected = configuration[tool]
                if signature(expected["path"])["sha256"] != expected["sha256"]:
                    raise ValueError(f"Recorded Chordino {tool} identity changed")
            if role is not None and role != "chordino":
                raise ValueError("Chordino sidecar supplied in a different source slot")
            item.update(configuration_sha256=config_hash, configuration=configuration,
                        configuration_report=signature(report_path))
            sample_rate = configuration["sample_rate"]
            full_window_declared = True
        duration = None
        if record.get("decoded_sample_count") is not None:
            duration = float(record["decoded_sample_count"]) / sample_rate
        elif record.get("audio_seconds") is not None:
            duration = float(record["audio_seconds"])
        elif record.get("duration") is not None:
            duration = float(record["duration"])
        if duration is not None and expected_duration is not None and abs(duration - expected_duration) > 2 / 44100:
            raise ValueError(f"External LAB decoded duration differs from source audio: {path}")
        coverage = record.get("coverage", {})
        if coverage and coverage.get("full_coverage") is not True:
            raise ValueError("Fresh Chordino sidecar does not establish full audio coverage")
        item.update(full_window_declared=full_window_declared, decoded_duration=duration,
                    duration_verified=duration is not None and expected_duration is not None)
        evidence.append(item)
    complete = all(item["source_sha_verified"] and item["output_sha_verified"] for item in evidence)
    result.update(sidecars=evidence, status="audio_and_lab_sha_verified" if complete else "sidecar_has_incomplete_source_identity")
    return result


def reject_input_overwrite(outputs, inputs):
    sources = [Path(path).resolve() for path in inputs]
    for output in outputs:
        output = Path(output).resolve()
        if output in sources or any(output.exists() and source.exists() and os.path.samefile(output, source) for source in sources):
            raise ValueError(f"Output would overwrite an input: {output}")
        if output.exists():
            raise FileExistsError(f"Output already exists; choose a new output path: {output}")


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_model(audio, raw, chordino, largevocab, model, output, cache, preserve_root=False, policy=None):
    started = time.perf_counter()
    output = Path(output)
    provenance_output = output.with_name(output.name + ".provenance.json")
    inputs = [audio, raw, chordino, largevocab, model, contract_path(model)]
    reject_input_overwrite([output, provenance_output], inputs)
    artifact, contract = load_frozen_artifact(model)
    policy_identity = validate_root_policy(policy, artifact, model)
    audio_identity = signature(audio)
    source_paths = [Path(raw), Path(chordino), Path(largevocab)]
    sources = [lab_provenance(path, audio_identity, role) for path, role in
               zip(source_paths, ("raw_madmom", "chordino", "largevocab"))]
    labs = [validate_timeline(read_reference(path), str(path)) for path in source_paths]
    if any(not rows for rows in labs):
        raise ValueError("All three explicitly supplied LAB files must contain intervals")
    with threadpool_limits(limits=2):
        evidence = notes.extract_note_evidence(audio, cache_dir=cache)
        validate_note_contract(evidence, contract, audio_identity["sha256"])
        sources = [lab_provenance(path, audio_identity, role, evidence.metadata["duration_seconds"]) for path, role in
                   zip(source_paths, ("raw_madmom", "chordino", "largevocab"))]
        features, groups = automatic_candidate_features(labs[0], labs[1:], evidence)
        prediction = predict_song(dict(raw=labs[0], groups=groups, features=features),
                                  artifact["method"], artifact["model"], artifact["margin"])
    if [(row.start, row.end) for row in prediction] != [(row.start, row.end) for row in labs[0]]:
        raise AssertionError("Candidate model changed raw interval support")
    full_head_changed = sum(chord_identity(a.chord) != chord_identity(b.chord) for a, b in zip(labs[0], prediction))
    guard = None
    if preserve_root:
        prediction, guard = preserve_raw_root(labs[0], prediction)
        guard["implementation"] = signature(Path(__file__).with_name("multisource_root_guard.py"))
        guard["policy"] = policy_identity
        guard["policy_sha256"] = policy_identity["sha256"] if policy_identity else None
        guard["development_history"] = "Fixed after user4 root-degradation feedback, before SWD scores; accepted only on fixed dev14"
    if signature(audio) != audio_identity or any(signature(path) != record["lab"] for path, record in zip(source_paths, sources)):
        raise RuntimeError("An input changed during candidate inference")
    text = "".join(f"{row.start:.17g} {row.end:.17g} {row.chord}\n" for row in prediction)
    _atomic_text(output, text)
    record = dict(schema="multisource_candidate_prediction_v1", audio=audio_identity,
        lab=signature(output), sources=dict(zip(contract["source_order"], sources)),
        model=signature(model), model_contract=signature(contract_path(model)), method=artifact["method"],
        margin=artifact["margin"], feature_schema=SCHEMA, feature_dimension=DIMENSION,
        nnls_contract=contract["nnls"], nnls_evidence_identity=evidence.metadata["identity"],
        interval_count=len(prediction), original_interval_support_preserved=True,
        changed_intervals=sum(chord_identity(a.chord) != chord_identity(b.chord) for a, b in zip(labs[0], prediction)),
        preserve_root=bool(preserve_root), root_guard=guard, full_head_changed_intervals=full_head_changed,
        offset=0, duration=evidence.metadata["duration_seconds"], timestamp_origin="decoded_source_audio_start",
        reference_inputs=False, fitting=False, score_kind="predicted_duration_utility_not_calibrated_probability",
        elapsed_seconds=time.perf_counter()-started)
    _atomic_text(provenance_output, json.dumps(record, ensure_ascii=False, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio", "raw", "chordino", "largevocab", "model", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preserve-root", action="store_true", help="Reject full-head root or N-status changes; retain same-root quality corrections")
    parser.add_argument("--policy", type=Path, help="Published frozen root-protection decision JSON; its SHA is retained")
    args = parser.parse_args()
    result = apply_model(args.audio, args.raw, args.chordino, args.largevocab, args.model, args.output, args.cache, preserve_root=args.preserve_root, policy=args.policy)
    print(json.dumps({key: result[key] for key in ("method", "margin", "interval_count", "changed_intervals", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
