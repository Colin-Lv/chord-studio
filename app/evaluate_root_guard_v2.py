"""Cached policy-only v2 diagnostic; no acoustic inference, training or search.

The five reported systems are raw, Chordino, large vocabulary, frozen v1
rank-then-root-guard, and allowed-first strategy v2. Both policies share one
prediction call on the unchanged v1 features and weights for each recording.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time

import joblib
import mir_eval
import numpy as np
from threadpoolctl import threadpool_limits

from analyze_multisource_candidate_potential import chord_identity
from apply_multisource_candidates import NNLS_CONTRACT_KEYS, validate_note_contract
from evaluate_chords import aligned_intervals, evaluate, harte_chord, read_reference, validate_timeline
from evaluate_multisource_candidate_reliability import manifest_items, read_evaluation_reference, reject_training_overlap
from generate_madmom_chords_from_manifest import output_path_for_row
from multisource_root_guard import preserve_raw_root
from multisource_root_guard_v2 import (FIXED_MARGIN, POLICY_RULE, POLICY_SCHEMA,
    frozen_v1_full_head_from_scores, preserve_root_allowed_first)
from train_multisource_candidate_reliability import (DIMENSION, SCHEMA,
    automatic_candidate_features, load_existing_evidence)
from train_multisource_seventh_gate import signature, write_json

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "benchmarks/reports/session_20260926_accuracy2"
MODEL_DIR = SESSION / "multisource_candidate_reliability"
SYSTEMS = ("raw", "chordino", "largevocab", "frozen_v1", "strategy_v2")
METRICS = tuple(name + suffix for suffix in ("_accuracy", "_wcsr")
                for name in ("root", "triad", "tetrad", "tetrad_inv")) + ("prediction_coverage",)


class ResourceMonitor:
    def __init__(self):
        self.peak_rss = None
        self.stop = threading.Event()
        try:
            import psutil
            self.process = psutil.Process(os.getpid())
        except ImportError:
            self.process = None
        self.worker = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            if self.process is not None:
                value = self.process.memory_info().rss
                self.peak_rss = max(self.peak_rss or 0, value)
            self.stop.wait(.1)

    def __enter__(self):
        self.wall, self.cpu = time.perf_counter(), time.process_time()
        self.worker.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.worker.join()

    def report(self):
        return dict(wall_seconds=time.perf_counter()-self.wall, cpu_seconds=time.process_time()-self.cpu,
                    peak_rss_bytes=self.peak_rss, rss_measurement="psutil RSS sampled each 0.1 s" if self.process else "psutil unavailable",
                    numeric_threads=2, acoustic_inference_calls=0)


def load_artifact(development=False):
    report_path = MODEL_DIR / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_path = MODEL_DIR / ("selected_development_head.joblib" if development else "selected_train66_head.joblib")
    record = report["development_model" if development else "refit_model"]
    if signature(model_path)["sha256"] != record["sha256"]:
        raise ValueError("Frozen v1 model differs from its original training report")
    artifact = joblib.load(model_path)
    if (artifact.get("schema") != SCHEMA or artifact.get("feature_dimension") != DIMENSION
            or artifact.get("method") != "hgb_regression" or artifact.get("margin") != FIXED_MARGIN
            or getattr(artifact["model"], "n_features_in_", None) != DIMENSION):
        raise ValueError("Only the original v1 HGB 136-feature model and margin .02 are allowed")
    for item in artifact["protocol"]["implementation"]:
        current = Path(__file__).with_name(Path(item["path"]).name)
        if signature(current)["sha256"] != item["sha256"]:
            raise ValueError("Frozen v1 feature implementation changed: " + current.name)
    contract_path = MODEL_DIR / "selected_train66_head.contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["implementation"] != artifact["protocol"]["implementation"]:
        raise ValueError("v1 note/feature contract differs between development and refit model")
    training_ids = set(artifact["training_song_ids"])
    training_audio = [dict(id=row["id"], **row["audio"]) for row in report["inputs"] if row["id"] in training_ids]
    if {row["id"] for row in training_audio} != training_ids:
        raise ValueError("Original report lacks fitted audio identities")
    return artifact, contract, dict(training_audio=training_audio), dict(model=signature(model_path),
        training_report=signature(report_path), feature_contract=signature(contract_path),
        fitted_song_count=len(training_ids), development=development,
        artifact_feature_schema=SCHEMA, feature_dimension=DIMENSION)


def source_identity(path, audio_identity):
    """Verify audio/LAB pairing; legacy absent sidecars remain unverified."""
    path = Path(path)
    result = dict(lab=signature(path), generation_identity="legacy_pairing_not_verified", sidecars=[])
    for sidecar in (path.with_name(path.name + ".provenance.json"), path.with_suffix(".provenance.json")):
        if not sidecar.is_file():
            continue
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        # Fresh Chordino records a display path in ``audio`` beside its
        # structured ``source_audio`` identity. A path must not shadow the
        # latter. Check every available identity so conflicting aliases fail.
        audio_hashes = [data[key]["sha256"] for key in ("audio", "source_audio")
                        if isinstance(data.get(key), dict) and data[key].get("sha256")]
        lab_hashes = [data[key]["sha256"] for key in ("lab", "output_lab", "output")
                      if isinstance(data.get(key), dict) and data[key].get("sha256")]
        if any(value != audio_identity["sha256"] for value in audio_hashes):
            raise ValueError("Source sidecar audio SHA mismatch")
        if any(value != result["lab"]["sha256"] for value in lab_hashes):
            raise ValueError("Source sidecar LAB SHA mismatch")
        for key in ("audio_offset", "offset", "offset_seconds"):
            if data.get(key) is not None and float(data[key]) != 0:
                raise ValueError("Only source-time offset zero is allowed")
        result["sidecars"].append(dict(identity=signature(sidecar),
            source_sha_verified=bool(audio_hashes), lab_sha_verified=bool(lab_hashes)))
    if result["sidecars"] and all(item["source_sha_verified"] and item["lab_sha_verified"] for item in result["sidecars"]):
        result["generation_identity"] = "audio_and_lab_bytes_verified_from_existing_sidecar"
    return result


def dataset_rows(name, development_ids):
    if name in {"swd24", "user4"}:
        manifest = SESSION / f"candidate_frozen_{name}.csv"
        rows = manifest_items(manifest)
    else:
        manifest = ROOT / ("benchmarks/manifest.beatles_selected6_83_train.csv" if name == "dev14" else "benchmarks/manifest.beatles_selected6_83_eval.csv")
        rows = manifest_items(manifest,
            ROOT / "benchmarks/external_chords/madmom/beatles_selected6",
            ROOT / "benchmarks/external_chords/chordino/beatles_selected6",
            ROOT / "benchmarks/external_chords/largevocab/beatles_selected6_raw")
        if name == "dev14":
            rows = [row for row in rows if row["id"] in development_ids]
            for row in rows:
                row.update(domain="Beatles_dev14", reference_status="Previously used development set; fitted model uses B52 only")
    expected = {"dev14": 14, "swd24": 24, "regression17": 17, "user4": 4}[name]
    if len(rows) != expected:
        raise ValueError(f"Fixed {name} expects {expected} songs")
    return rows, signature(manifest)


def _absolute_bass(label):
    label = harte_chord(label)
    if label in {"N", "X"}:
        return None
    root, _, bass = mir_eval.chord.encode(label)
    return (int(root) + int(bass)) % 12


def change_metrics(reference, raw, prediction):
    if [(r.start, r.end) for r in raw] != [(r.start, r.end) for r in prediction]:
        raise ValueError("A policy changed original raw timing/coverage")
    result = dict(changed_intervals=0, changed_seconds=0., changed_annotated_seconds=0.,
                  harmful_change_seconds=0., beneficial_change_seconds=0.,
                  inversion_metric_regression_seconds=0., bass_regression_seconds=0.)
    for before, after in zip(raw, prediction):
        if chord_identity(before.chord) != chord_identity(after.chord):
            result["changed_intervals"] += 1
            result["changed_seconds"] += before.end - before.start
    index = 0
    for start, end, truth, after in aligned_intervals(reference, prediction):
        while index < len(raw) and raw[index].end <= start:
            index += 1
        if truth is None or after is None or index == len(raw) or raw[index].start > start:
            continue
        before = raw[index]
        ref = harte_chord(truth.chord)
        if ref == "X" or chord_identity(before.chord) == chord_identity(after.chord):
            continue
        seconds = end-start
        result["changed_annotated_seconds"] += seconds
        labels = [harte_chord(before.chord), harte_chord(after.chord)]
        tetrad = mir_eval.chord.tetrads([ref, ref], labels)
        inversion = mir_eval.chord.tetrads_inv([ref, ref], labels)
        result["harmful_change_seconds"] += seconds * int(tetrad[0] == 1 and tetrad[1] != 1)
        result["beneficial_change_seconds"] += seconds * int(tetrad[0] != 1 and tetrad[1] == 1)
        result["inversion_metric_regression_seconds"] += seconds * int(inversion[0] == 1 and inversion[1] != 1)
        expected_bass = _absolute_bass(ref)
        if expected_bass is not None:
            result["bass_regression_seconds"] += seconds * int(_absolute_bass(before.chord) == expected_bass and _absolute_bass(after.chord) != expected_bass)
    return result


def aggregate(songs):
    output = dict(completed=len(songs), denominator="Successful songs only; failures retained separately", systems={})
    for system in SYSTEMS:
        rows = [song["metrics"][system] for song in songs]
        macro, micro = {}, {}
        for metric in METRICS:
            macro[metric] = float(np.mean([row[metric] for row in rows])) if rows else None
            denominator = "non_no_chord_seconds" if metric.endswith("_accuracy") else "evaluated_seconds"
            seconds = sum(row[denominator] for row in rows)
            micro[metric] = sum(row[metric]*row[denominator] for row in rows)/seconds if seconds else None
        output["systems"][system] = dict(macro=macro, micro=micro,
            evaluated_seconds=sum(row["evaluated_seconds"] for row in rows),
            chord_seconds=sum(row["non_no_chord_seconds"] for row in rows))
    output["changes"] = {kind: {key: sum(song["changes"][kind][key] for song in songs)
        for key in (songs[0]["changes"][kind] if songs else [])} for kind in ("frozen_v1", "strategy_v2")}
    return output


def run_dataset(name, output, cache):
    artifact, contract, fitted, model_identity = load_artifact(name == "dev14")
    rows, manifest_identity = dataset_rows(name, set(artifact["protocol"]["development_ids"]))
    reject_training_overlap(rows, artifact, fitted)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(dataset=name, model=model_identity, manifest=manifest_identity, requested=len(rows), songs=[], failures=[])
    write_json(output / "input_protocol.json", dict(model=model_identity, manifest=manifest_identity,
        ids=[row["id"] for row in rows], margin=FIXED_MARGIN, policy=POLICY_SCHEMA,
        fitting=False, acoustic_inference=False, v1_feature_dimension=DIMENSION))
    with ResourceMonitor() as resources, threadpool_limits(limits=2):
        for row in rows:
            started, cpu = time.perf_counter(), time.process_time()
            try:
                paths = [row[key] for key in ("upstream", "chordino", "largevocab")]
                source_records = [source_identity(path, row["audio_identity"]) for path in paths]
                labs = [validate_timeline(read_reference(path), str(path)) for path in paths]
                if any(not data for data in labs):
                    raise ValueError("All three existing source LABs must have intervals")
                evidence, cache_identity = load_existing_evidence(row["audio"], cache)
                validate_note_contract(evidence, contract, row["audio_identity"]["sha256"])
                features, groups = automatic_candidate_features(labs[0], labs[1:], evidence)
                # Exactly one model call per song; both policies see these scores.
                utilities = np.asarray(artifact["model"].predict(features))
                full = frozen_v1_full_head_from_scores(labs[0], groups, utilities)
                frozen, old_details = preserve_raw_root(labs[0], full)
                strategy, new_details = preserve_root_allowed_first(labs[0], groups, utilities)
                predicted = dict(raw=labs[0], chordino=labs[1], largevocab=labs[2], frozen_v1=frozen, strategy_v2=strategy)
                predictions = {}
                for kind in ("frozen_v1", "strategy_v2"):
                    folder = output / kind
                    folder.mkdir(exist_ok=True)
                    target = output_path_for_row(folder, row["id"])
                    if target.exists():
                        raise FileExistsError("Refusing to overwrite diagnostic predictions")
                    target.write_text("".join(f"{item.start:.17g} {item.end:.17g} {item.chord}\n" for item in predicted[kind]), encoding="utf-8")
                    predictions[kind] = signature(target)
                # No reference file is opened before both predictions are fixed.
                reference = read_evaluation_reference(row["reference"])
                metrics = {kind: evaluate(reference, data) for kind, data in predicted.items()}
                for kind in ("frozen_v1", "strategy_v2"):
                    for metric in ("root_accuracy", "root_wcsr", "prediction_coverage"):
                        if abs(metrics[kind][metric]-metrics["raw"][metric]) > 1e-12:
                            raise AssertionError("A root-preserving policy changed raw roots or coverage")
                differences = [dict(start=a.start, end=a.end, raw=r.chord, frozen_v1=a.chord, strategy_v2=b.chord)
                    for r, a, b in zip(labs[0], frozen, strategy) if chord_identity(a.chord) != chord_identity(b.chord)]
                result = dict(id=row["id"], domain=row["domain"], reference_status=row["reference_status"],
                    metrics=metrics, changes={kind: change_metrics(reference, labs[0], predicted[kind]) for kind in ("frozen_v1", "strategy_v2")},
                    policy_differences=differences, v1_guard=old_details,
                    v2_summary={key:value for key,value in new_details.items() if key != "decisions"},
                    candidate_count=len(utilities), model_predict_calls=1,
                    inputs=dict(audio=row["audio_identity"], sources=source_records, notes=cache_identity,
                        reference=signature(row["reference"])), predictions=predictions,
                    wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
                report["songs"].append(result)
                print(f"{name}: {len(report['songs'])}/{len(rows)} {row['id']}", flush=True)
            except Exception as exc:
                report["failures"].append(dict(id=row["id"], error=f"{type(exc).__name__}: {exc}"))
                print(f"FAILED {name} {row['id']}: {exc}", flush=True)
            write_json(output / "report.json", report)
        report.update(resources=resources.report(), summary=aggregate(report["songs"]),
            completed=len(report["songs"]), failure_count=len(report["failures"]),
            status="complete" if not report["failures"] else "complete_with_failures")
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("dev14", "swd24", "regression17", "user4"),
                        default=["dev14", "swd24", "regression17", "user4"])
    parser.add_argument("--note-cache", type=Path, default=ROOT / ".tmp/nnls_seventh_features/notes")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/reports/model_v2_20261003/policy_diagnostic")
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Dataset names must not repeat")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        parser.error("Use a new directory; original protocols/results are immutable")
    protocol = dict(policy=POLICY_SCHEMA, rule=POLICY_RULE, margin=FIXED_MARGIN,
        declared_before_diagnostic_scores=True, datasets=args.datasets, systems=list(SYSTEMS),
        frozen_v1_definition="Global argmax then strict raw margin .02, followed by original v1 root guard",
        strategy_v2_definition="Filter allowed same-root candidates first, then argmax and same strict raw margin .02",
        semantic_model_change=False, v1_feature_schema=SCHEMA, v1_feature_dimension=DIMENSION,
        model_training=False, parameter_search=False, acoustic_inference=False, source_recognizers=False,
        dev14_model="Original B52 fitted development head; never B66 refit on development songs",
        other_models="Original B66 selected/refit head; no refit in this diagnostic",
        single_score_call=True, interval_policy="Original raw boundaries and gaps unchanged",
        wrong_change_definition="Known-reference duration where raw tetrad was correct and policy tetrad is wrong",
        bass_regression_definition="Known chord-reference duration where raw absolute bass matches reference and policy absolute bass does not",
        macro="Equal-song mean", micro="Sum correct seconds / sum eligible seconds; accuracy excludes N/X references, WCSR includes explicit N",
        limitations=["All datasets are previously observed regression/development material, not a new blind test",
            "Original Beatles auxiliary LAB generation identities are missing; pretrained overlaps unknown",
            "User4 stored corrections are mixed automatic/manual and not independent complete gold",
            "Strategy v2 is a policy diagnostic on v1 scores, not the new semantic feature head",
            "Per-song failures are retained; aggregate denominators include only successful songs"],
        implementation=[signature(__file__), signature(Path(__file__).with_name("multisource_root_guard_v2.py")),
            signature(Path(__file__).with_name("multisource_root_guard.py")), signature(Path(__file__).with_name("evaluate_chords.py"))])
    write_json(args.output / "protocol.json", protocol)
    report = dict(protocol=protocol, datasets={}, failures=[])
    for name in args.datasets:
        try:
            data = run_dataset(name, args.output / name, args.note_cache)
            report["datasets"][name] = dict(report=signature(args.output/name/"report.json"),
                completed=data["completed"], failures=data["failure_count"], summary=data["summary"], resources=data["resources"])
        except Exception as exc:
            report["failures"].append(dict(dataset=name, error=f"{type(exc).__name__}: {exc}"))
            print(f"FAILED DATASET {name}: {exc}", flush=True)
        write_json(args.output / "report.json", report)
    failures = len(report["failures"]) + sum(data["failures"] for data in report["datasets"].values())
    report["status"] = "complete" if not failures else "complete_with_failures"
    write_json(args.output / "report.json", report)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
