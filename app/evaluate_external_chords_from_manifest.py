#!/usr/bin/env python3
"""Evaluate external chord .lab predictions for manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_chords import evaluate, read_reference
from generate_madmom_chords_from_manifest import output_path_for_row


CHORD_METRICS = ["root_accuracy", "triad_accuracy", "root_wcsr", "triad_wcsr", "tetrad_wcsr",
                 "prediction_coverage", "evaluated_seconds", "non_no_chord_seconds",
                 "unknown_reference_seconds", "missing_prediction_seconds"]
FIELDS = ["id", "prediction", "chord_reference", *CHORD_METRICS, "status", "message"]


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    ok = [row for row in rows if row["status"] == "ok"]
    summary: dict[str, object] = {"items": len(rows), "evaluated": len(ok)}
    for metric in ("root_accuracy", "triad_accuracy", "root_wcsr", "triad_wcsr", "tetrad_wcsr", "prediction_coverage"):
        values = [float(row[metric]) for row in ok if row.get(metric) not in {"", None}]
        if values:
            summary[f"mean_{metric}"] = sum(values) / len(values)
            summary[f"count_{metric}"] = len(values)
            weight = "non_no_chord_seconds" if metric.endswith("accuracy") else "evaluated_seconds"
            total = sum(float(row[weight]) for row in ok)
            summary[f"duration_weighted_{metric}"] = sum(float(row[metric]) * float(row[weight]) for row in ok) / total if total else 0
    summary["evaluation_protocol"] = "full reference support, exact duration; X/gaps excluded; WCSR includes N"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmarks/reports/external_chords"))
    parser.add_argument("--id-contains")
    parser.add_argument("--frame", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.manifest.parent
    prediction_dir = args.prediction_dir.resolve()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for row in manifest_rows:
        row_id = row.get("id", "")
        if args.id_contains and args.id_contains not in row_id:
            continue
        out = {field: "" for field in FIELDS}
        out["id"] = row_id
        ref = resolve_path(base, row.get("chord_reference", ""))
        pred = output_path_for_row(prediction_dir, row_id)
        out["prediction"] = str(pred)
        out["chord_reference"] = str(ref or "")
        if not ref or not ref.exists():
            out["status"] = "missing_reference"
            out["message"] = f"missing reference: {ref}"
        elif not pred.exists():
            out["status"] = "missing_prediction"
            out["message"] = f"missing prediction: {pred}"
        else:
            try:
                metrics = evaluate(read_reference(ref), read_reference(pred), args.frame)
                for metric in CHORD_METRICS:
                    out[metric] = metrics[metric]
                out["status"] = "ok"
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                out["status"] = "error"
                out["message"] = str(exc)
        rows.append(out)

    csv_path = args.output.with_suffix(".csv")
    json_path = args.output.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {"summary": summarize(rows), "items": rows}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    for key, value in report["summary"].items():
        if key.startswith("mean_") and not math.isnan(float(value)):
            print(f"{key}: {float(value):.3f}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
