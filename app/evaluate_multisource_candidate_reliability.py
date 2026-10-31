"""Evaluate one frozen candidate head; no tuning or fitting.

Loads existing three-source LABs and existing NNLS note caches. Selection and
margin come only from the previously frozen artifact, never this manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_chords import evaluate, read_reference, read_prediction, validate_timeline
from generate_madmom_chords_from_manifest import output_path_for_row
from train_multisource_candidate_reliability import (SCHEMA, DIMENSION,
    automatic_candidate_features, load_existing_evidence, predict_song)
from analyze_multisource_candidate_potential import chord_identity
from train_multisource_seventh_gate import signature, write_json
import harmonic_note_features as notes
from apply_multisource_candidates import (load_frozen_artifact, validate_note_contract,
                                         lab_provenance, reject_input_overwrite, contract_path, validate_root_policy)
from multisource_root_guard import GUARD_RULE, GUARD_SCHEMA, preserve_raw_root


def manifest_items(manifest, madmom_labs=None, chordino_labs=None, largevocab_labs=None):
    manifest = Path(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    generic = "upstream" in fields
    required = {"id", "audio", "upstream", "chordino", "largevocab", "reference", "domain", "reference_status"}
    if generic and not required <= fields:
        raise ValueError(f"Generic frozen manifest lacks {sorted(required - fields)}")
    items = []
    for row in rows:
        if not row.get("id") or not row.get("audio"):
            raise ValueError("Every evaluation row requires id and audio")
        item = dict(id=row["id"], audio=(manifest.parent / row["audio"]).resolve())
        if generic:
            if any(not row.get(name) for name in required):
                raise ValueError("Generic evaluation fields cannot be empty")
            item.update({name: (manifest.parent / row[name]).resolve() for name in ("upstream", "chordino", "largevocab", "reference")})
            item.update(domain=row["domain"], reference_status=row["reference_status"])
        else:
            if not all((madmom_labs, chordino_labs, largevocab_labs)):
                raise ValueError("Legacy manifests require all three --*-labs directories")
            item.update(upstream=output_path_for_row(madmom_labs, row["id"]),
                        chordino=output_path_for_row(chordino_labs, row["id"]),
                        largevocab=output_path_for_row(largevocab_labs, row["id"]),
                        reference=(manifest.parent / row["chord_reference"]).resolve(),
                        domain=row.get("domain", "Beatles_eval17"),
                        reference_status=row.get("reference_status", "existing_annotation_not_project_blind"))
        items.append(item)
    if not items:
        raise ValueError("Evaluation manifest is empty")
    return items


def reject_training_overlap(items, artifact, contract):
    train_ids = set(artifact["training_song_ids"])
    train_paths = {str(Path(path).resolve()).casefold() for path in artifact["training_audio_paths"]}
    train_hashes = {item["sha256"] for item in contract["training_audio"]}
    ids, paths, hashes, output_names = set(), set(), set(), set()
    for item in items:
        path = str(item["audio"].resolve()).casefold()
        audio = signature(item["audio"])
        name = output_path_for_row(Path("."), item["id"]).name.casefold()
        if (item["id"] in train_ids or path in train_paths or audio["sha256"] in train_hashes):
            raise ValueError("Evaluation song/audio overlaps fitted training identities")
        if item["id"] in ids or path in paths or audio["sha256"] in hashes or name in output_names:
            raise ValueError("Evaluation contains duplicate song/audio/output identities")
        ids.add(item["id"])
        paths.add(path)
        hashes.add(audio["sha256"])
        output_names.add(name)
        item["audio_identity"] = audio


def read_evaluation_reference(path):
    """Studio event JSON and timed LAB/CSV share the same scoring contract."""
    path = Path(path)
    rows = read_prediction(path) if path.suffix.lower() == ".json" else read_reference(path)
    rows = validate_timeline(rows, "Evaluation reference")
    if not rows:
        raise ValueError("Evaluation reference must contain timed chords")
    return rows


def validate_resume_protocol(directory, protocol):
    directory = Path(directory)
    old_path = directory / "protocol.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    for key in ("model", "manifest", "model_contract"):
        if old.get(key, {}).get("sha256") != protocol[key]["sha256"]:
            raise ValueError(f"Resume input differs from frozen {key}")
    for key in ("protocol", "frozen_method", "frozen_margin"):
        if old.get(key) != protocol[key]:
            raise ValueError(f"Resume protocol differs at {key}")
    return dict(protocol=signature(old_path), report=signature(directory / "report.json"),
                policy="Reuse existing predictions; infer only missing songs from cached evidence. No fitting or selection.")


def same_interval_support(left, right):
    # Older evaluation LABs used nine decimals. This tolerance permits only
    # their serialization rounding, never reference-aligned retiming.
    return len(left) == len(right) and all(abs(a.start - b.start) <= 5.1e-10 and
        abs(a.end - b.end) <= 5.1e-10 for a, b in zip(left, right))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--madmom-labs", type=Path)
    parser.add_argument("--chordino-labs", type=Path)
    parser.add_argument("--largevocab-labs", type=Path)
    parser.add_argument("--note-cache", type=Path, default=Path(".tmp/nnls_seventh_features/notes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-notes", action="store_true", help="Permit frozen NNLS extraction on a cache miss; never runs external chord recognizers")
    parser.add_argument("--resume-from", type=Path, help="Reuse existing predictions from an identical frozen run, preserving its report")
    parser.add_argument("--policy", type=Path, help="Published frozen root-protection decision JSON; its SHA is retained")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    if report_path.exists() or (args.output / "protocol.json").exists():
        raise ValueError("Use a new directory for a frozen evaluation")
    started = time.perf_counter()
    artifact, contract = load_frozen_artifact(args.model)
    policy_identity = validate_root_policy(args.policy, artifact, args.model)
    rows = manifest_items(args.manifest, args.madmom_labs, args.chordino_labs, args.largevocab_labs)
    reject_training_overlap(rows, artifact, contract)
    protocol = dict(protocol="single_frozen_multisource_candidate_evaluation", model=signature(args.model),
        manifest=signature(args.manifest), frozen_method=artifact["method"], frozen_margin=artifact["margin"],
        score_type="exact-duration chord-only accuracy and WCSR", configuration_search=False, model_fitting=False,
        external_recognizer_inference=False, nnls_cache_miss_extraction_allowed=args.extract_notes and not bool(args.resume_from),
        model_contract=signature(contract_path(args.model)),
        reported_sources=["baseline_raw_madmom", "standalone_chordino", "standalone_largvocab", "selected_frozen_candidate", "root_guarded_frozen_candidate"],
        baseline_protocol="Each frozen standalone source retains its native original intervals; selected alone uses raw intervals. These columns never select or replace the frozen model.",
        selection_note="Original frozen head plus one separately fixed root-only guard; no external-score substitution or threshold search",
        root_guard=dict(schema=GUARD_SCHEMA, rule=GUARD_RULE,
            implementation=signature(Path(__file__).with_name("multisource_root_guard.py")),
            policy=policy_identity, policy_sha256=policy_identity["sha256"] if policy_identity else None,
            development_history="Proposed after user4 diagnostic root degradation, accepted on fixed dev14, frozen before SWD scoring. Root+triad protection had failed dev and is not deployed."),
        limitations=["Independence depends on each row's domain/reference_status; prior project eval17 is not a project-level blind test",
                     "Legacy auxiliary LAB generation audio/model/config identity is unverified",
                     "External pretrained model overlap unknown"],
        implementation=signature(__file__))
    if args.resume_from:
        protocol["resume"] = validate_resume_protocol(args.resume_from, protocol)
    write_json(args.output / "protocol.json", protocol)
    report = dict(protocol=protocol, status="running", songs=[], failures=[], requested=len(rows))
    predictions = args.output / "predictions"
    predictions.mkdir(exist_ok=True)
    guarded_predictions = args.output / "predictions_root_guarded"
    guarded_predictions.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        for row in rows:
            audio = row["audio"]
            song_started = time.perf_counter()
            try:
                lab_paths = [row[name] for name in ("upstream", "chordino", "largevocab")]
                labs = [validate_timeline(read_reference(path), str(path)) if path.is_file() else [] for path in lab_paths]
                if not labs[0]:
                    raise ValueError("Raw LAB required")
                old_prediction = output_path_for_row(args.resume_from / "predictions", row["id"]) if args.resume_from else None
                if old_prediction is not None and old_prediction.is_file():
                    predicted = validate_timeline(read_reference(old_prediction), "Resumed prediction")
                    if not same_interval_support(predicted, labs[0]):
                        raise ValueError("Resumed prediction does not retain the current upstream interval support")
                    cache_identity = dict(reused_prediction=signature(old_prediction), extraction=False,
                        limitation="Original failed run did not checkpoint per-song input hashes; reuse trusts its matching frozen protocol and retained predictions")
                    source_provenance = [lab_provenance(path, row["audio_identity"], role)
                        for path, role in zip(lab_paths, ("raw_madmom", "chordino", "largevocab")) if path.is_file()]
                else:
                    if args.extract_notes and not args.resume_from:
                        evidence = notes.extract_note_evidence(audio, cache_dir=args.note_cache)
                        cache_identity = dict(identity=evidence.metadata["identity"])
                    else:
                        evidence, cache_identity = load_existing_evidence(audio, args.note_cache)
                    validate_note_contract(evidence, contract, row["audio_identity"]["sha256"])
                    source_provenance = [lab_provenance(path, row["audio_identity"], role, evidence.metadata["duration_seconds"])
                        for path, role in zip(lab_paths, ("raw_madmom", "chordino", "largevocab")) if path.is_file()]
                    features, groups = automatic_candidate_features(labs[0], labs[1:], evidence)
                    song = dict(raw=labs[0], features=features, groups=groups)
                    predicted = predict_song(song, artifact["method"], artifact["model"], artifact["margin"] or 0.)
                    if [(r.start, r.end) for r in predicted] != [(r.start, r.end) for r in labs[0]]:
                        raise AssertionError("Frozen model altered original upstream interval boundaries")
                # Reference read only after candidates, features and predictions
                # have been fixed for this song.
                reference_path = row["reference"]
                path = output_path_for_row(predictions, row["id"])
                reject_input_overwrite([path], [audio, *lab_paths, reference_path, args.model, contract_path(args.model), args.manifest])
                if old_prediction is not None and old_prediction.is_file():
                    shutil.copyfile(old_prediction, path)
                else:
                    path.write_text("".join(f"{r.start:.17g} {r.end:.17g} {r.chord}\n" for r in predicted), encoding="utf-8")
                root_guarded, guard_details = preserve_raw_root(labs[0], predicted)
                guard_details["policy"] = policy_identity
                guard_details["policy_sha256"] = policy_identity["sha256"] if policy_identity else None
                guard_details["implementation"] = protocol["root_guard"]["implementation"]
                guarded_path = output_path_for_row(guarded_predictions, row["id"])
                guarded_sidecar = guarded_path.with_name(guarded_path.name + ".provenance.json")
                reject_input_overwrite([guarded_path, guarded_sidecar], [audio, *lab_paths, reference_path, args.model, contract_path(args.model), args.manifest])
                guarded_path.write_text("".join(f"{r.start:.17g} {r.end:.17g} {r.chord}\n" for r in root_guarded), encoding="utf-8")
                write_json(guarded_sidecar, dict(schema="multisource_root_guarded_prediction_v1",
                    audio=row["audio_identity"], lab=signature(guarded_path), model=signature(args.model),
                    model_contract=signature(contract_path(args.model)), method=artifact["method"], margin=artifact["margin"],
                    preserve_root=True, root_guard=guard_details, full_head_prediction=signature(path),
                    sources=source_provenance, original_interval_support_preserved=True, offset=0,
                    reference_inputs=False, fitting=False))
                reference = read_evaluation_reference(reference_path)
                result = dict(id=row["id"], domain=row["domain"], reference_status=row["reference_status"],
                    baseline=evaluate(reference, labs[0]), chordino=evaluate(reference, labs[1]),
                    largevocab=evaluate(reference, labs[2]), selected=evaluate(reference, predicted),
                    root_guarded=evaluate(reference, root_guarded), root_guard=guard_details,
                    changed_intervals=sum(chord_identity(a.chord) != chord_identity(b.chord) for a, b in zip(labs[0], predicted)),
                    original_timing_preserved=True, elapsed_seconds=time.perf_counter()-song_started,
                    inputs=dict(audio=signature(audio), notes=cache_identity, reference=signature(reference_path),
                        labs=[signature(path) if path.is_file() else dict(path=str(path.resolve()), missing=True) for path in lab_paths],
                        source_provenance=source_provenance),
                    prediction=signature(path), root_guarded_prediction=signature(guarded_path), root_guarded_provenance=signature(guarded_sidecar))
                for metric in ("root_accuracy", "root_wcsr", "prediction_coverage"):
                    if abs(result["baseline"][metric] - result["root_guarded"][metric]) > 1e-12:
                        raise AssertionError("Fixed root guard changed root scores or interval support")
                report["songs"].append(result)
                print(f"Evaluated {len(report['songs'])}/{len(rows)}: {row['id']}", flush=True)
            except Exception as exc:
                report["failures"].append(dict(id=row["id"], domain=row["domain"], reference_status=row["reference_status"], error=f"{type(exc).__name__}: {exc}"))
                print(f"FAILED {row['id']}: {exc}", flush=True)
            write_json(report_path, report)
    metrics = ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy",
               "root_wcsr", "triad_wcsr", "tetrad_wcsr", "tetrad_inv_wcsr", "prediction_coverage")
    report["summary"] = {f"{kind}_macro_{metric}": float(np.mean([song[kind][metric] for song in report["songs"]]))
        for kind in ("baseline", "chordino", "largevocab", "selected", "root_guarded") for metric in metrics} if report["songs"] else {}
    report["domains"] = {}
    for domain in sorted({row["domain"] for row in rows}):
        songs = [song for song in report["songs"] if song["domain"] == domain]
        report["domains"][domain] = dict(completed=len(songs),
            failures=sum(item["domain"] == domain for item in report["failures"]),
            reference_statuses=sorted({row["reference_status"] for row in rows if row["domain"] == domain}),
            summary={f"{kind}_macro_{metric}": float(np.mean([song[kind][metric] for song in songs]))
                for kind in ("baseline", "chordino", "largevocab", "selected", "root_guarded") for metric in metrics} if songs else {})
    report.update(status="complete" if not report["failures"] else "complete_with_failures",
                  completed=len(report["songs"]), failure_count=len(report["failures"]),
                  macro_denominator="Successful songs only; all failures recorded explicitly",
                  elapsed_seconds=time.perf_counter()-started)
    write_json(report_path, report)
    print(json.dumps(dict(completed=report["completed"], failures=report["failure_count"], summary=report["summary"]), indent=2), flush=True)
    return 2 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
