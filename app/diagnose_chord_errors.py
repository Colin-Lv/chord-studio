#!/usr/bin/env python3
"""Diagnose chord benchmark errors by comparing external ACR and fused output."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from chord_studio import read_external_chord_hints
from evaluate_chords import TimedChord, evaluate, read_prediction, read_reference
from generate_madmom_chords_from_manifest import output_path_for_row


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def external_prediction(path: Path) -> list[TimedChord]:
    return [TimedChord(item.start, item.end, item.chord) for item in read_external_chord_hints(path, "sharps")]


def metric_or_empty(reference: list[TimedChord], prediction: list[TimedChord], frame: float) -> dict[str, float]:
    try:
        return evaluate(reference, prediction, frame)
    except ValueError:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("external_chords_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--frame", type=float, default=0.1)
    parser.add_argument("--limit-worst", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    base = manifest.parent
    external_dir = args.external_chords_dir.resolve()
    rows_out: list[dict[str, object]] = []

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row_id = row.get("id", "")
        reference_path = resolve_path(base, row.get("chord_reference", ""))
        prediction_path = resolve_path(base, row.get("prediction", ""))
        external_path = output_path_for_row(external_dir, row_id)
        if not reference_path or not prediction_path:
            continue
        if not reference_path.exists() or not prediction_path.exists() or not external_path.exists():
            continue
        reference = read_reference(reference_path)
        fused = read_prediction(prediction_path)
        external = external_prediction(external_path)
        fused_metrics = metric_or_empty(reference, fused, args.frame)
        external_metrics = metric_or_empty(reference, external, args.frame)
        if not fused_metrics or not external_metrics:
            continue
        fused_root = fused_metrics["root_accuracy"]
        fused_triad = fused_metrics["triad_accuracy"]
        ext_root = external_metrics["root_accuracy"]
        ext_triad = external_metrics["triad_accuracy"]
        rows_out.append(
            {
                "id": row_id,
                "artist": row.get("artist", ""),
                "title": row.get("title", ""),
                "key": row.get("key", ""),
                "external_root": ext_root,
                "external_triad": ext_triad,
                "fused_root": fused_root,
                "fused_triad": fused_triad,
                "root_delta": fused_root - ext_root,
                "triad_delta": fused_triad - ext_triad,
                "error_type": (
                    "fusion_regression"
                    if fused_triad + 0.02 < ext_triad
                    else "fusion_gain"
                    if fused_triad > ext_triad + 0.02
                    else "candidate_limit"
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "id",
            "artist",
            "title",
            "key",
            "external_root",
            "external_triad",
            "fused_root",
            "fused_triad",
            "root_delta",
            "triad_delta",
            "error_type",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary: dict[str, int] = {}
    for row in rows_out:
        label = str(row["error_type"])
        summary[label] = summary.get(label, 0) + 1
    print(f"Wrote: {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Worst fusion regressions:")
    for row in sorted(rows_out, key=lambda item: float(item["triad_delta"]))[: args.limit_worst]:
        print(
            f"{float(row['triad_delta']):+.3f} "
            f"ext={float(row['external_triad']):.3f} fused={float(row['fused_triad']):.3f} "
            f"{row['id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
