#!/usr/bin/env python
"""Compare two Chord Studio analysis JSON files by bar/span labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def spans_for_bar(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for bar in payload.get("bars", []):
        number = int(bar.get("bar", 0))
        spans = bar.get("spans") or []
        rows[number] = sorted(spans, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    return rows


def label(span: dict[str, Any]) -> str:
    chord = str(span.get("chord_display") or span.get("chord") or "N")
    roman = str(span.get("roman_display") or span.get("roman") or "N")
    key = str(span.get("local_key") or "")
    return f"{chord}|{roman}|{key}"


def compact(spans: list[dict[str, Any]]) -> str:
    return " ; ".join(label(span) for span in spans)


def has_dominant_seventh_delta(reference: str, prediction: str) -> bool:
    return "7" in reference and reference.replace("7", "") == prediction.replace("7", "")


def compare(reference: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    ref_by_bar = spans_for_bar(reference)
    pred_by_bar = spans_for_bar(prediction)
    rows = []
    for bar in sorted(set(ref_by_bar).union(pred_by_bar)):
        ref_spans = ref_by_bar.get(bar, [])
        pred_spans = pred_by_bar.get(bar, [])
        ref_label = compact(ref_spans)
        pred_label = compact(pred_spans)
        if ref_label == pred_label:
            continue
        ref_chords = [str(span.get("chord_display") or span.get("chord") or "N") for span in ref_spans]
        pred_chords = [str(span.get("chord_display") or span.get("chord") or "N") for span in pred_spans]
        ref_romans = [str(span.get("roman_display") or span.get("roman") or "N") for span in ref_spans]
        pred_romans = [str(span.get("roman_display") or span.get("roman") or "N") for span in pred_spans]
        category = "label"
        if len(ref_chords) != len(pred_chords):
            category = "segmentation"
        elif ref_chords != pred_chords:
            category = "chord"
        elif ref_romans != pred_romans:
            category = "roman_or_key"
        if ref_chords != pred_chords and len(ref_chords) == len(pred_chords) and any(
            has_dominant_seventh_delta(ref, pred) for ref, pred in zip(ref_chords, pred_chords)
        ):
            category = "missing_or_wrong_seventh"
        rows.append(
            {
                "bar": bar,
                "category": category,
                "reference": ref_label,
                "prediction": pred_label,
                "reference_chords": " ".join(ref_chords),
                "prediction_chords": " ".join(pred_chords),
                "reference_romans": " ".join(ref_romans),
                "prediction_romans": " ".join(pred_romans),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = compare(load(args.reference), load(args.prediction))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "bar",
        "category",
        "reference_chords",
        "prediction_chords",
        "reference_romans",
        "prediction_romans",
        "reference",
        "prediction",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.output}")
    print(f"changed bars: {len(rows)}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{key}: {count}")
    for row in rows[: args.limit]:
        print(f"bar {row['bar']:03d} [{row['category']}] ref={row['reference_chords']} pred={row['prediction_chords']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
