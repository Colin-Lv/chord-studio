"""Evaluate the optional pretrained expert on a manifest without oracle boundaries."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from evaluate_song_structure import boundary_metrics, locate_annotations
from evaluate_beats import f_measure
from structure_expert import import_expert_structure, run_expert, write_expert_beats


def role_accuracy(reference, estimated):
    mapping = {"intro": "intro", "outro": "outro", "verse": "verse", "chorus": "chorus", "bridge": "bridge", "silence": "silence", "verse_(instrumental)": "inst", "chorus_(instrumental)": "inst"}
    correct = total = 0.0
    for start, end, role in reference:
        role = mapping.get(role)
        if role is None or role == "silence":
            continue
        total += end - start
        for segment in estimated:
            overlap = max(0.0, min(end, segment["end"]) - max(start, segment["start"]))
            if role == segment["label"]:
                correct += overlap
    return {"duration_weighted_accuracy": correct / total if total else None, "evaluated_seconds": total, "correct_seconds": correct, "label_mapping": mapping, "note": "Only explicitly mapped reference roles are scored; the two datasets use different form vocabularies."}


def alignment_diagnostic(reference_beats, estimated_beats, reference_downbeats=None, estimated_downbeats=None):
    """Flag a possible recording/timebase mismatch; never correct official scores.

    Sequential matching is valid only when the detector has not missed or added
    beats. Report its residual and count difference so this assumption remains
    visible. The affine fit uses test annotations and is strictly diagnostic.
    """
    reference = np.asarray(reference_beats, dtype=float)
    estimated = np.asarray(estimated_beats, dtype=float)
    count = min(len(reference), len(estimated))
    if count < 20:
        return {"available": False}
    scale, offset = np.polyfit(estimated[:count], reference[:count], 1)
    residual = reference[:count] - (scale * estimated[:count] + offset)
    result = {"available": True, "oracle_diagnostic_only": True, "assumption": "Sequential one-to-one beat correspondence; invalid after inserted/missed beats", "count_difference": len(estimated) - len(reference), "reference_equals_scale_times_prediction_plus_offset": {"scale": float(scale), "offset_seconds": float(offset)}, "median_absolute_residual_ms": float(np.median(np.abs(residual)) * 1000), "aligned_beat_f1_70ms": f_measure(list(reference_beats), list(scale * estimated + offset), 0.07)["f1"], "applied_to_metrics": False}
    result["reference_beat_count"] = len(reference)
    result["estimated_beat_count"] = len(estimated)
    result["estimated_reference_count_ratio"] = len(estimated) / len(reference)
    if reference_downbeats is not None and estimated_downbeats is not None:
        result["aligned_downbeat_f1_70ms"] = f_measure(list(reference_downbeats), list(scale * np.asarray(estimated_downbeats) + offset), 0.07)["f1"]
    return result


def beat_metrics(reference, estimated):
    return {key: value if np.isfinite(value) else None for key, value in f_measure(list(reference), list(estimated), 0.07).items()}


def decoded_duration(audio):
    """Count decoded samples; MP3 container duration includes codec padding."""
    from chord_recognizer import find_ffmpeg
    command = [find_ffmpeg(), "-v", "error", "-i", str(audio), "-ar", "22050", "-ac", "1", "-f", "f32le", "-"]
    data = subprocess.run(command, check=True, capture_output=True).stdout
    if not data or len(data) % 4:
        raise ValueError("Invalid decoded audio sample count")
    return len(data) / (4 * 22050)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", default="harmonix-fold0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache", type=Path, default=Path(".tmp/structure_expert"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--automatic-analyses", type=Path, help="Unhinted analysis report for display grids only")
    parser.add_argument("--previous-report", type=Path, help="Preserve first-run timings from an earlier fixed-model subset")
    parser.add_argument("--stop-starting-at", help="Local YYYY-MM-DDTHH:MM:SS deadline for starting another track")
    args = parser.parse_args()
    root = Path("benchmarks").resolve()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]
    records, skipped, failures = [], [], []
    previous_records = {}
    for previous_path in [args.previous_report, args.output]:
        if previous_path is None or not previous_path.exists():
            continue
        previous_report = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_report.get("model") == args.model:
            previous_records.update({item["id"]: item for item in previous_report.get("tracks", [])})
    automatic = {}
    if args.automatic_analyses:
        report = json.loads(args.automatic_analyses.read_text(encoding="utf-8"))
        if report.get("reference_hints_used") is not False:
            raise ValueError("Display analyses must explicitly have no reference hints")
        automatic = {item["id"]: item for item in report["tracks"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(running=None):
        completed = {item["id"] for item in records + skipped + failures}
        summary = {"manifest": str(args.manifest), "model": args.model, "oracle_grid": False,
                   "inference_window_source": "full audio decoded sample count at 22050 Hz; offset=0",
                   "reference_hints_used": False, "pretrained_training_overlap": "unknown",
                   "annotation_alignment": "raw timestamps; affine diagnostic is never applied to metrics",
                   "display_grid_source": str(args.automatic_analyses) if automatic else "reference (display only)",
                   "manifest_track_count": len(rows), "track_count": len(records), "failure_count": len(failures),
                   "remaining": [row["id"] for row in rows if row["id"] not in completed], "running": running,
                   "tracks": records, "skipped": skipped, "failures": failures}
        if records:
            summary.update({"macro_boundary_metrics": {str(tolerance): {metric: float(np.mean([item["boundary_metrics"][str(tolerance)][metric] for item in records])) for metric in ("precision", "recall", "f1")} for tolerance in (0.5, 3.0)}, "macro_beat_f1_70ms": float(np.mean([item["beat_metrics_70ms"]["f1"] for item in records])), "macro_downbeat_f1_70ms": float(np.mean([item["downbeat_metrics_70ms"]["f1"] for item in records])), "macro_role_accuracy": float(np.mean([item["role_metrics"]["duration_weighted_accuracy"] for item in records if item["role_metrics"]["duration_weighted_accuracy"] is not None]))})
        temporary = args.output.with_suffix(".pending.json")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        return summary

    checkpoint()
    for row in rows:
        track_id = row["id"]
        if args.stop_starting_at and datetime.datetime.now() >= datetime.datetime.fromisoformat(args.stop_starting_at):
            print("Stopping before new track at requested deadline", flush=True)
            break
        annotation = locate_annotations(root, track_id)
        reference_path = root / "references/isophonics" / (track_id + "_reference_analysis.json")
        audio = root / row["audio"]
        if annotation is None or not reference_path.exists() or not audio.exists():
            skipped.append({"id": track_id, "reason": "missing audio or reference"})
            checkpoint()
            continue
        print(f"Analyzing {track_id}", flush=True)
        checkpoint(track_id)
        began = time.monotonic()
        try:
            duration = decoded_duration(audio)
            inference_payload = {"source": str(audio.resolve()), "duration": duration, "audio_offset": 0}
            external, activations, provenance = run_expert(inference_payload, args.cache, args.model, audio=audio, threads=args.threads)
        except Exception as exc:
            failures.append({"id": track_id, "error": str(exc), "traceback": traceback.format_exc(), "runtime_seconds": round(time.monotonic() - began, 3)})
            checkpoint()
            print(f"FAILED {track_id}: {exc}", flush=True)
            continue
        # References are first opened after independent full-audio inference.
        display_path = Path(automatic[track_id]["prediction"]) if automatic else reference_path
        payload = json.loads(display_path.read_text(encoding="utf-8-sig"))
        if automatic and (Path(payload["source"]).resolve() != audio.resolve() or abs(float(payload.get("audio_offset", 0))) > 1e-9 or abs(float(payload["duration"]) - duration) > 1 / 22050):
            raise ValueError(f"Automatic display audio/window mismatch: {track_id}")
        provenance["inference_duration_from_decoded_samples"] = duration
        provenance["display_analysis"] = str(display_path)
        provenance["reference_window_used"] = False
        labels = [line.split(maxsplit=2) for line in annotation.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        reference = [(float(start), float(end), label) for start, end, label in labels]
        boundaries = [item[0] for item in reference[1:]]
        estimated = [segment["start"] for segment in external["segments"][1:]]
        beat_rows = [line.split() for line in (root / row["beat_reference"]).read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.startswith("#")]
        reference_beats = [float(line[0]) for line in beat_rows]
        reference_downbeats = [float(line[0]) for line in beat_rows if len(line) > 1 and int(float(line[1])) == 1]
        result = {"id": track_id, "model": args.model, "runtime_seconds": round(time.monotonic() - began, 3), "provenance": provenance, "boundary_metrics": {str(tolerance): boundary_metrics(boundaries, estimated, tolerance) for tolerance in (0.5, 3.0)}, "beat_metrics_70ms": beat_metrics(reference_beats, external["beats"]), "downbeat_metrics_70ms": beat_metrics(reference_downbeats, external["downbeats"]), "role_metrics": role_accuracy(reference, external["segments"]), "alignment_diagnostic": alignment_diagnostic(reference_beats, external["beats"], reference_downbeats, external["downbeats"]), "reference_boundaries": boundaries, "estimated_boundaries": estimated}
        if provenance.get("cache_reused") and track_id in previous_records:
            result["first_run_runtime_seconds"] = previous_records[track_id].get("first_run_runtime_seconds", previous_records[track_id]["runtime_seconds"])
        records.append(result)
        payload["structure"] = import_expert_structure(payload, external, activations, args.model)
        payload["structure"]["provenance"].update(provenance)
        directory = args.output.parent / (args.output.stem + "_payloads") / args.model
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (track_id + ".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_expert_beats(directory / (track_id + ".beats"), external)
        summary = checkpoint()
        print(json.dumps({"completed": len(records), "failed": len(failures), "id": track_id, "cache_reused": provenance.get("cache_reused"), "runtime_seconds": result["runtime_seconds"], "beat_f1": result["beat_metrics_70ms"]["f1"], "downbeat_f1": result["downbeat_metrics_70ms"]["f1"]}), flush=True)
    checkpoint()


if __name__ == "__main__":
    main()
