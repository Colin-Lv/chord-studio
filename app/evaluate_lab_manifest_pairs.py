#!/usr/bin/env python
"""Evaluate paired chord .lab manifests by normalized song title."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from evaluate_chords import evaluate, read_reference


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_title(value: str) -> str:
    text = value.lower()
    text = text.replace("&", "and")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        key = normalize_title(row.get("title", "") or row.get("id", ""))
        if key and key not in output:
            output[key] = row
    return output


def manifest_lab_path(row: dict[str, str]) -> Path:
    value = row.get("chord_reference") or row.get("prediction") or ""
    if not value:
        raise ValueError(f"Manifest row has no chord path: {row.get('id', '')}")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / "benchmarks" / path
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("prediction_manifest", type=Path)
    parser.add_argument("--frame", type=float, default=0.1)
    parser.add_argument("--csv-report", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    reference_manifest = read_manifest(resolve_path(args.reference_manifest))
    prediction_manifest = read_manifest(resolve_path(args.prediction_manifest))
    rows = []
    total_seconds = 0.0
    weighted_root = 0.0
    weighted_triad = 0.0
    for key, reference_row in reference_manifest.items():
        prediction_row = prediction_manifest.get(key)
        if not prediction_row:
            continue
        reference_path = manifest_lab_path(reference_row)
        prediction_path = manifest_lab_path(prediction_row)
        try:
            metrics = evaluate(read_reference(reference_path), read_reference(prediction_path), args.frame)
        except ValueError as exc:
            rows.append(
                {
                    "title": reference_row.get("title", ""),
                    "status": "error",
                    "error": str(exc),
                    "root_accuracy": "",
                    "triad_accuracy": "",
                    "evaluated_seconds": "",
                    "reference": str(reference_path),
                    "prediction": str(prediction_path),
                }
            )
            continue
        # root/triad_accuracy exclude explicit N, so aggregate with the same
        # chord-only denominator rather than the whole annotated timeline.
        seconds = float(metrics["non_no_chord_seconds"])
        total_seconds += seconds
        weighted_root += seconds * float(metrics["root_accuracy"])
        weighted_triad += seconds * float(metrics["triad_accuracy"])
        rows.append(
            {
                "title": reference_row.get("title", ""),
                "status": "ok",
                "error": "",
                "root_accuracy": f"{metrics['root_accuracy']:.4f}",
                "triad_accuracy": f"{metrics['triad_accuracy']:.4f}",
                "evaluated_seconds": f"{seconds:.1f}",
                "reference": str(reference_path),
                "prediction": str(prediction_path),
            }
        )

    csv_report = resolve_path(args.csv_report)
    csv_report.parent.mkdir(parents=True, exist_ok=True)
    with csv_report.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["title", "status", "error", "root_accuracy", "triad_accuracy", "evaluated_seconds", "reference", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    summary = {
        "reference_manifest": str(resolve_path(args.reference_manifest)),
        "prediction_manifest": str(resolve_path(args.prediction_manifest)),
        "matched": len(rows),
        "evaluated": ok_count,
        "weighted_root_accuracy": weighted_root / total_seconds if total_seconds else 0.0,
        "weighted_triad_accuracy": weighted_triad / total_seconds if total_seconds else 0.0,
        "evaluated_seconds": total_seconds,
        "csv_report": str(csv_report),
    }
    json_report = resolve_path(args.json_report)
    json_report.parent.mkdir(parents=True, exist_ok=True)
    json_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
