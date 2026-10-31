#!/usr/bin/env python
"""Run batch benchmark reports for Chord Studio outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_beats import evaluate as evaluate_beats
from evaluate_beats import read_prediction as read_beat_prediction
from evaluate_beats import read_reference as read_beat_reference
from evaluate_beats import greedy_match
from evaluate_chords import evaluate as evaluate_chords
from evaluate_chords import read_prediction as read_chord_prediction
from evaluate_chords import read_reference as read_chord_reference


FIELDNAMES = [
    "id",
    "dataset",
    "split",
    "artist",
    "title",
    "prediction",
    "chord_reference",
    "beat_reference",
    "beat_source",
    "key_source",
    "reference_grid_exact_fraction",
    "evaluation_context",
    "alignment_method",
    "root_accuracy",
    "triad_accuracy",
    "root_wcsr",
    "triad_wcsr",
    "tetrad_wcsr",
    "prediction_coverage",
    "non_no_chord_seconds",
    "evaluated_seconds",
    "unknown_reference_seconds",
    "missing_prediction_seconds",
    "beat_f1",
    "downbeat_f1",
    "beat_median_abs_error_ms",
    "downbeat_median_abs_error_ms",
    "status",
    "message",
]


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def evaluate_row(row: dict[str, str], base: Path, frame: float, tolerance: float) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDNAMES}
    for key in ("id", "dataset", "split", "artist", "title", "prediction", "chord_reference", "beat_reference"):
        result[key] = row.get(key, "")

    prediction = resolve_path(base, row.get("prediction", ""))
    if prediction is None or not prediction.exists():
        result["status"] = "missing_prediction"
        result["message"] = f"Prediction not found: {prediction}"
        return result

    messages: list[str] = []
    try:
        payload = json.loads(prediction.read_text(encoding="utf-8-sig"))
        result["beat_source"] = str(payload.get("beat_source", "unknown"))
        result["key_source"] = str(payload.get("key_source", "unknown"))
        result["alignment_method"] = row.get("alignment_method", "") or (
            "legacy_prediction_alignment_unverified" if row.get("alignment_best_triad") else "")
        context = []
        if result["alignment_method"]:
            context.append("prediction_optimized_reference_alignment")
        declared = payload.get("evaluation_context", {})
        if "reference" in result["beat_source"] or isinstance(declared, dict) and any(
            declared.get(field) for field in ("reference_beat_grid", "reference_grid")):
            context.append("reference_beat_grid")
        if isinstance(declared, dict) and declared.get("reference_key"):
            context.append("reference_key")
        chord_ref = resolve_path(base, row.get("chord_reference", ""))
        if chord_ref and chord_ref.exists():
            chord_metrics = evaluate_chords(
                read_chord_reference(chord_ref),
                read_chord_prediction(prediction),
                frame,
            )
            for metric in chord_metrics:
                if metric in result:
                    result[metric] = chord_metrics[metric]
        elif chord_ref:
            messages.append(f"Chord reference not found: {chord_ref}")

        beat_ref = resolve_path(base, row.get("beat_reference", ""))
        if beat_ref and beat_ref.exists():
            ref_beats = read_beat_reference(beat_ref)
            pred_beats = read_beat_prediction(prediction)
            beat_metrics = evaluate_beats(
                ref_beats,
                pred_beats,
                tolerance,
            )
            exact_hits, _ = greedy_match([b.time for b in ref_beats], [b.time for b in pred_beats], 1e-6)
            exact_fraction = exact_hits / max(len(pred_beats), 1)
            result["reference_grid_exact_fraction"] = exact_fraction
            if len(pred_beats) >= 32 and exact_fraction >= .95 and "reference_beat_grid" not in context:
                context.append("grid_matches_reference_verify_provenance")
            result["beat_f1"] = beat_metrics.get("beat_f1", "")
            result["downbeat_f1"] = beat_metrics.get("downbeat_f1", "")
            result["beat_median_abs_error_ms"] = beat_metrics.get("beat_median_abs_error_ms", "")
            result["downbeat_median_abs_error_ms"] = beat_metrics.get("downbeat_median_abs_error_ms", "")
        elif beat_ref:
            messages.append(f"Beat reference not found: {beat_ref}")
        result["evaluation_context"] = ";".join(context) or "unspecified"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result["status"] = "error"
        result["message"] = str(exc)
        return result

    result["status"] = "ok" if not messages else "partial"
    result["message"] = "; ".join(messages)
    return result


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if row.get("status") in {"ok", "partial"}]
    summary: dict[str, object] = {"items": len(rows), "evaluated": len(ok_rows)}
    for metric in (
        "root_accuracy",
        "triad_accuracy",
        "root_wcsr",
        "triad_wcsr",
        "tetrad_wcsr",
        "prediction_coverage",
        "beat_f1",
        "downbeat_f1",
        "beat_median_abs_error_ms",
        "downbeat_median_abs_error_ms",
    ):
        values = []
        for row in ok_rows:
            value = row.get(metric, "")
            if value == "" or value is None:
                continue
            number = float(value)
            if not math.isnan(number):
                values.append(number)
        if values:
            summary[f"mean_{metric}"] = sum(values) / len(values)
            summary[f"count_{metric}"] = len(values)
    for metric in ("root_accuracy", "triad_accuracy", "root_wcsr", "triad_wcsr", "tetrad_wcsr"):
        weight_key = "non_no_chord_seconds" if metric.endswith("accuracy") else "evaluated_seconds"
        weighted_rows = [row for row in ok_rows if row.get(metric) not in {"", None}
                         and row.get(weight_key) not in {"", None}]
        duration = sum(float(row[weight_key]) for row in weighted_rows)
        if duration:
            summary[f"duration_weighted_{metric}"] = sum(float(row[metric]) * float(row[weight_key])
                                                         for row in weighted_rows) / duration
    summary["evaluation_protocol"] = "full reference support, exact duration; X/gaps excluded; WCSR includes N"
    summary["context_counts"] = {}
    for row in ok_rows:
        for context in str(row.get("evaluation_context") or "unspecified").split(";"):
            summary["context_counts"][context] = summary["context_counts"].get(context, 0) + 1
    if any(key != "unspecified" for key in summary["context_counts"]):
        summary["provenance_note"] = "Controlled/reference-aligned rows must not be described as unassisted end-to-end accuracy. Exact grid matches require provenance review."
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark manifest.")
    parser.add_argument("manifest", type=Path, help="CSV manifest.")
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmarks/reports/latest"))
    parser.add_argument("--frame", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=0.07)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.manifest.parent
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = [evaluate_row(row, base, args.frame, args.tolerance) for row in rows]
    csv_path = args.output.with_suffix(".csv")
    json_path = args.output.with_suffix(".json")
    write_csv(csv_path, results)
    report = {"summary": summarize(results), "items": results}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    for key, value in report["summary"].items():
        if key.startswith("mean_"):
            print(f"{key}: {float(value):.3f}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
