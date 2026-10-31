"""Run the unhinted madmom pipeline and score the resulting complete analyses.

Reference key, meter, tempo and beat annotations are used only after inference
for evaluation. Existing madmom chord LABs may be reused as explicit pretrained
expert outputs; overlap between its training corpus and this evaluation is
unknown. Every manifest row receives a result or a recorded failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mir_eval.key

import evaluate_beats
import evaluate_chords
from evaluate_song_structure import boundary_metrics, locate_annotations
from evaluate_structure_expert import alignment_diagnostic


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def file_signature(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def global_key_metrics(reference, estimated, mode=None):
    def standard_key(value, mode_hint=None):
        text = str(value or "").strip().replace("♭", "b").replace("♯", "#")
        match = re.fullmatch(r"([A-G][b#]?)(m|:min|:minor|:maj|:major| minor| major)?", text)
        if not match or mode_hint not in {None, "", "major", "minor", "ionian", "aeolian"}:
            return None
        suffix = (match.group(2) or "").strip()
        minor = suffix in {"m", ":min", ":minor", "minor"} or mode_hint in {"minor", "aeolian"}
        return match.group(1) + (" minor" if minor else " major")
    ref, pred = standard_key(reference), standard_key(estimated, mode)
    if not ref:
        return {"available": False, "reference": reference, "estimated": estimated}
    if not pred:
        return {"available": True, "reference": ref, "estimated": estimated, "exact_correct": False, "mirex_weighted_score": 0.0, "note": "Reference global key is available; missing or unsupported prediction counts as incorrect."}
    score = float(mir_eval.key.weighted_score(ref, pred))
    return {"available": True, "reference": ref, "estimated": pred, "exact_correct": score == 1.0, "mirex_weighted_score": score, "note": "Single global key only; not a local modulation evaluation."}


def score_track(row, prediction, root, external_lab=None):
    payload = json.loads(prediction.read_text(encoding="utf-8-sig"))
    chord_ref = evaluate_chords.read_reference(root / row["chord_reference"])
    chord_pred = evaluate_chords.read_prediction(prediction)
    beat_ref = evaluate_beats.read_reference(root / row["beat_reference"])
    beat_pred = evaluate_beats.read_prediction(prediction)
    chords = evaluate_chords.evaluate(chord_ref, chord_pred)
    beats = evaluate_beats.evaluate(beat_ref, beat_pred, tolerance=0.07)
    diagnostics = alignment_diagnostic([item.time for item in beat_ref], [item.time for item in beat_pred], [item.time for item in beat_ref if item.beat == 1], [item.time for item in beat_pred if item.beat == 1])
    annotation = locate_annotations(root, row["id"])
    structure_metrics = {}
    if annotation:
        rows = [line.split(maxsplit=2) for line in annotation.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        boundaries = [float(line[0]) for line in rows[1:]]
        estimated = [float(item["time"]) for item in payload.get("structure", {}).get("boundaries", [])]
        structure_metrics = {str(tolerance): boundary_metrics(boundaries, estimated, tolerance) for tolerance in (0.5, 3.0)}
    result = {"chord_metrics": chords, "beat_metrics": beats, "key_metrics": global_key_metrics(row.get("key"), payload.get("key"), payload.get("mode")), "structure_metrics": structure_metrics, "alignment_diagnostic": diagnostics, "inferred": {key: payload.get(key) for key in ("key", "meter", "bpm", "beat_source", "beat_grid", "analysis_model", "duration")}, "structure_method": payload.get("structure", {}).get("method")}
    result["structure_configuration"] = payload.get("structure", {}).get("config")
    if external_lab:
        result["external_lab_chord_metrics"] = evaluate_chords.evaluate(chord_ref, evaluate_chords.read_reference(external_lab))
        result["fusion_minus_external_lab"] = {key: chords[key] - result["external_lab_chord_metrics"][key] for key in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "triad_wcsr")}
    return result


def summarize(records, total, manifest, key_source="functional", seventh_threshold=0.45, bass_weight=0.075, edge_fill=False, edge_include_n=False, infer_slash=True):
    successful = [item for item in records if item.get("status") in {"complete", "degraded"}]
    summary = {"protocol": "audio + pretrained madmom LAB; automatic functional key, tempo, meter and beats; default fusion and lightweight structure", "key_source": "functional", "reference_hints_used": False, "pretrained_training_overlap": "unknown", "annotation_alignment": "raw reference times; affine diagnostics never applied to scores", "manifest": str(manifest), "manifest_track_count": total, "completed_count": len(successful), "failure_count": sum(item.get("status") == "failed" for item in records), "remaining_count": total - len(records), "degraded_count": sum(item.get("status") == "degraded" for item in records), "tracks": records}
    summary["manifest_signature"] = file_signature(manifest)
    summary["key_source"] = key_source
    summary["dominant_seventh_threshold"] = seventh_threshold
    summary["bass_root_weight"] = bass_weight
    summary["edge_fill"] = edge_fill
    summary["edge_fill_include_no_chord"] = edge_include_n
    summary["infer_slash_bass"] = infer_slash
    summary["protocol"] = f"audio + pretrained madmom LAB; automatic {key_source} key, tempo, meter and beats; fusion seventh threshold {seventh_threshold}; lightweight structure"
    summary["scoring_parameters"] = {"beat_tolerance_seconds": 0.07, "structure_boundary_tolerances_seconds": [0.5, 3.0], "chord_time_integration": "exact", "beat_matching": "maximum one-to-one mir_eval.util.match_events"}
    if successful:
        chord_keys = ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "root_wcsr", "triad_wcsr", "tetrad_wcsr", "tetrad_inv_wcsr", "prediction_coverage")
        summary["macro_chord_metrics"] = {key: float(np.mean([item["chord_metrics"][key] for item in successful if key in item["chord_metrics"]])) for key in chord_keys if any(key in item["chord_metrics"] for item in successful)}
        summary["duration_weighted_chord_metrics"] = {}
        for key in chord_keys:
            pairs = [(item["chord_metrics"][key], item["chord_metrics"].get("non_no_chord_seconds" if key.endswith("_accuracy") else "evaluated_seconds", 0)) for item in successful if key in item["chord_metrics"]]
            if pairs and sum(weight for _, weight in pairs):
                summary["duration_weighted_chord_metrics"][key] = sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)
        external = [item["external_lab_chord_metrics"] for item in successful if item.get("external_lab_chord_metrics")]
        if external:
            summary["external_lab_comparison_track_count"] = len(external)
            summary["macro_external_lab_chord_metrics"] = {key: float(np.mean([item[key] for item in external])) for key in chord_keys}
            summary["duration_weighted_external_lab_chord_metrics"] = {}
            for key in chord_keys:
                weight_key = "non_no_chord_seconds" if key.endswith("_accuracy") else "evaluated_seconds"
                weight = sum(item[weight_key] for item in external)
                if weight:
                    summary["duration_weighted_external_lab_chord_metrics"][key] = sum(item[key] * item[weight_key] for item in external) / weight
            summary["macro_fusion_minus_external_lab"] = {key: float(np.mean([item["fusion_minus_external_lab"][key] for item in successful if "fusion_minus_external_lab" in item])) for key in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "triad_wcsr")}
        summary["macro_beat_metrics"] = {key: float(np.mean([item["beat_metrics"].get(key, 0) for item in successful])) for key in ("beat_f1", "downbeat_f1", "beat_precision", "beat_recall", "downbeat_precision", "downbeat_recall")}
        summary["macro_structure_f1"] = {str(tolerance): float(np.mean([item["structure_metrics"][str(tolerance)]["f1"] for item in successful if str(tolerance) in item["structure_metrics"]])) for tolerance in (0.5, 3.0)}
        keys = [item["key_metrics"] for item in successful if item.get("key_metrics", {}).get("available")]
        if keys:
            summary["global_key_metrics"] = {"track_count": len(keys), "exact_accuracy": float(np.mean([item["exact_correct"] for item in keys])), "mirex_weighted_score": float(np.mean([item["mirex_weighted_score"] for item in keys]))}
    return clean_json(summary)


def run_pipeline(command, environment, log, timeout):
    """Kill only this pipeline's process tree if its time budget expires."""
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=log, stderr=subprocess.STDOUT, check=False)
        else:
            process.kill()
        process.wait()
        raise


def write_score_table(records, path):
    fields = ["id", "status", "reference_key", "estimated_key", "key_exact", "root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "triad_wcsr", "prediction_coverage", "external_lab_triad_accuracy", "external_lab_tetrad_accuracy", "fusion_minus_external_triad", "beat_f1", "downbeat_f1", "structure_f1_05s", "structure_f1_3s", "first_run_seconds", "prediction", "error"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            row = {key: item.get(key, "") for key in ("id", "status", "prediction", "error")}
            row.update({key: item.get("chord_metrics", {}).get(key) for key in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "triad_wcsr", "prediction_coverage")})
            row.update({key: item.get("beat_metrics", {}).get(key) for key in ("beat_f1", "downbeat_f1")})
            row["external_lab_triad_accuracy"] = item.get("external_lab_chord_metrics", {}).get("triad_accuracy")
            row["external_lab_tetrad_accuracy"] = item.get("external_lab_chord_metrics", {}).get("tetrad_accuracy")
            row["fusion_minus_external_triad"] = item.get("fusion_minus_external_lab", {}).get("triad_accuracy")
            keys = item.get("key_metrics", {})
            row.update({"reference_key": keys.get("reference"), "estimated_key": keys.get("estimated"), "key_exact": keys.get("exact_correct"), "first_run_seconds": item.get("first_run_elapsed_seconds", item.get("elapsed_seconds"))})
            row["structure_f1_05s"] = item.get("structure_metrics", {}).get("0.5", {}).get("f1")
            row["structure_f1_3s"] = item.get("structure_metrics", {}).get("3.0", {}).get("f1")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--key-source", choices=["functional", "madmom"], default="functional",
                        help="Explicitly frozen estimator; default reproduces the original session baseline.")
    parser.add_argument("--seventh-threshold", type=float, default=0.45,
                        help="Explicit frozen fusion setting; use 1.01 for the improved default without template seventh inheritance.")
    parser.add_argument("--bass-root-weight", type=float, default=0.075)
    parser.add_argument("--edge-fill", action=argparse.BooleanOptionalAction, default=False,
                        help="Explicit frozen setting; enable for the improved source-preserving edges.")
    parser.add_argument("--edge-fill-include-no-chord", action=argparse.BooleanOptionalAction, default=False,
                        help="Explicitly preserve source N at uncovered edges; disabled by default to reproduce earlier session baselines.")
    parser.add_argument("--infer-slash-bass", action=argparse.BooleanOptionalAction, default=True,
                        help="Legacy inferred bass display; disable for the train-selected source-only display policy.")
    args = parser.parse_args()
    root = Path("benchmarks").resolve()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "evaluation.json"
    previous_records = {}
    if reports.exists():
        previous_records = {item["id"]: item for item in json.loads(reports.read_text(encoding="utf-8")).get("tracks", [])}
    cached_labs = {path.stem: path for path in (root / "external_chords/madmom").rglob("*.lab")}
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[key] = str(max(1, args.threads))
    environment["PYTHONUNBUFFERED"] = "1"
    first_id = "isophonics_all_The_Beatles_06_-_Rubber_Soul_13_-_If_I_Needed_Someone"
    existing_first = output / "if_i_needed_someone_analysis.json"
    records = []
    for row in rows:
        track_id = row["id"]
        audio = root / row["audio"]
        base = output / track_id
        prediction = base.with_name(base.name + "_analysis.json")
        if track_id == first_id and existing_first.exists():
            prediction = existing_first
        record = {"id": track_id, "audio": str(audio), "prediction": str(prediction), "threads": args.threads, "reference_hints_used": False}
        began = time.monotonic()
        try:
            if not audio.is_file():
                raise FileNotFoundError(audio)
            lab = cached_labs.get(track_id)
            command = [sys.executable, str(Path(__file__).resolve().parent / "analyze_with_madmom.py"), str(audio), "-o", str(base), "--beat-source", "madmom", "--structure", "lightweight", "--key-source", args.key_source,
                       "--dominant-seventh-threshold", str(args.seventh_threshold),
                       "--bass-root-weight", str(args.bass_root_weight), "--edge-fill" if args.edge_fill else "--no-edge-fill",
                       "--edge-fill-include-no-chord" if args.edge_fill_include_no_chord else "--no-edge-fill-include-no-chord",
                       "--infer-slash-bass" if args.infer_slash_bass else "--no-infer-slash-bass"]
            if lab:
                command += ["--madmom-lab", str(lab)]
                record["cached_lab"] = file_signature(lab)
            record["command"] = command
            log_path = output / (track_id + ".log")
            record["log"] = str(log_path)
            if prediction.exists():
                record["reused_prediction"] = True
                record["command_is_reproduction_only"] = True
                if track_id == first_id and prediction == existing_first:
                    record["reuse_provenance"] = "Unhinted first-song run produced by root agent during this session; mapped by source audio and manifest ID."
            elif args.score_only:
                raise FileNotFoundError("Missing prediction in score-only mode: " + str(prediction))
            else:
                record["reused_prediction"] = False
                record["command_is_reproduction_only"] = False
                record["original_inference_command"] = command
                print(f"Running {track_id}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    returncode = run_pipeline(command, environment, log, args.timeout)
                record["returncode"] = returncode
                if returncode:
                    raise RuntimeError(f"Pipeline exited with {returncode}; see {log_path}")
                if not prediction.exists():
                    raise FileNotFoundError("Pipeline did not create final JSON")
            predicted_payload = json.loads(prediction.read_text(encoding="utf-8-sig"))
            predicted_source = predicted_payload.get("source")
            if not predicted_source or Path(predicted_source).resolve() != audio.resolve():
                raise ValueError("Prediction source does not match this manifest audio")
            if (predicted_payload.get("key_source") == "madmom_cnn_key") != (args.key_source == "madmom") or predicted_payload.get("fusion_provenance", {}).get("key_source") == "provided":
                raise ValueError("Prediction does not use the requested automatic key estimator")
            actual_threshold = predicted_payload.get("fusion_provenance", {}).get("dominant_seventh_threshold", 0.45)
            if actual_threshold != args.seventh_threshold:
                raise ValueError("Prediction uses a different seventh inheritance configuration")
            actual_fusion = predicted_payload.get("fusion_provenance", {})
            if actual_fusion.get("bass_root_weight", 0.075) != args.bass_root_weight or actual_fusion.get("edge_fill", False) != args.edge_fill:
                raise ValueError("Prediction uses a different bass or edge fusion configuration")
            if args.edge_fill and actual_fusion.get("edge_fill_include_no_chord", False) != args.edge_fill_include_no_chord:
                raise ValueError("Prediction uses a different explicit edge-N policy")
            if actual_fusion.get("infer_slash_bass", True) != args.infer_slash_bass:
                raise ValueError("Prediction uses a different inferred slash display policy")
            if predicted_payload.get("structure", {}).get("method") != "bar_multifeature_novelty":
                raise ValueError("Prediction is missing the requested lightweight structure result")
            record["prediction_signature"] = file_signature(prediction)
            record.update(score_track(row, prediction, root, external_lab=lab))
            record["status"] = "complete" if record["inferred"]["beat_source"] == "madmom" else "degraded"
            if record["status"] == "degraded":
                record["warning"] = "Requested madmom beat tracking fell back to another source; retained and disclosed."
            print(f"{track_id}: triad={record['chord_metrics'].get('triad_accuracy', 0):.3f}, beat={record['beat_metrics'].get('beat_f1', 0):.3f}, downbeat={record['beat_metrics'].get('downbeat_f1', 0):.3f}", flush=True)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {track_id}: {record['error']}", flush=True)
        record["elapsed_seconds"] = round(time.monotonic() - began, 3)
        if record.get("reused_prediction") and track_id in previous_records:
            previous = previous_records[track_id]
            record["first_run_elapsed_seconds"] = previous.get("first_run_elapsed_seconds", None if previous.get("reused_prediction") else previous.get("elapsed_seconds"))
            original_command = previous.get("original_inference_command") or (None if previous.get("reused_prediction") else previous.get("command"))
            record["original_inference_command"] = original_command
            if original_command:
                record["reproduction_command"] = record["command"]
                record["command"] = original_command
                record["original_inference_command"] = record["command"]
                if "--key-source" not in record["command"]:
                    record["key_source_provenance"] = "Original run omitted --key-source while its default was functional; reproduction command freezes that choice explicitly."
        elif record.get("reused_prediction"):
            record["first_run_elapsed_seconds"] = None
            record["original_inference_command"] = None
        records.append(record)
        reports.write_text(json.dumps(summarize(records, len(rows), args.manifest, args.key_source, args.seventh_threshold, args.bass_root_weight, args.edge_fill, args.edge_fill_include_no_chord, args.infer_slash_bass), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        write_score_table(records, output / "per_song.csv")
    print(json.dumps({key: value for key, value in summarize(records, len(rows), args.manifest, args.key_source, args.seventh_threshold, args.bass_root_weight, args.edge_fill, args.edge_fill_include_no_chord, args.infer_slash_bass).items() if key != "tracks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
