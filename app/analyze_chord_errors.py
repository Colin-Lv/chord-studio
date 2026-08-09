#!/usr/bin/env python3
"""Summarize chord confusion patterns for benchmark predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate_chords import (
    active_at,
    canonical_chord,
    convert_annotation_chord,
    evaluate,
    read_prediction,
    read_reference,
)


PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def canonical_label(label: str) -> str:
    root, family = canonical_chord(label)
    if root is None:
        return "N"
    return f"{PC_NAMES[root]}{family}"


def analyze_pair(reference_path: Path, prediction_path: Path, frame: float) -> dict[str, object]:
    reference = read_reference(reference_path)
    prediction = read_prediction(prediction_path)
    metrics = evaluate(reference, prediction, frame)

    start = max(min(item.start for item in reference), min(item.start for item in prediction))
    end = min(max(item.end for item in reference), max(item.end for item in prediction))
    times = np.arange(start + frame / 2.0, end, frame)
    ref_labels = active_at(reference, times)
    pred_labels = active_at(prediction, times)

    confusion: Counter[tuple[str, str]] = Counter()
    root_confusion: Counter[tuple[str, str]] = Counter()
    quality_confusion: Counter[tuple[str, str]] = Counter()
    ref_distribution: Counter[str] = Counter()

    for ref_raw, pred_raw in zip(ref_labels, pred_labels):
        ref_root, ref_family = canonical_chord(ref_raw)
        if ref_root is None:
            continue
        pred_root, pred_family = canonical_chord(pred_raw)
        ref = canonical_label(ref_raw)
        pred = canonical_label(pred_raw)
        ref_distribution[ref] += 1
        if ref != pred:
            confusion[(ref, pred)] += 1
        root_name = PC_NAMES[ref_root]
        pred_root_name = "N" if pred_root is None else PC_NAMES[pred_root]
        if ref_root != pred_root:
            root_confusion[(root_name, pred_root_name)] += 1
        if ref_root == pred_root and ref_family != pred_family:
            quality_confusion[(ref_family or "maj", pred_family or "maj")] += 1

    return {
        "metrics": metrics,
        "confusion": confusion,
        "root_confusion": root_confusion,
        "quality_confusion": quality_confusion,
        "ref_distribution": ref_distribution,
    }


def counter_rows(counter: Counter[tuple[str, str]], total: int, limit: int) -> list[dict[str, object]]:
    rows = []
    for (ref, pred), count in counter.most_common(limit):
        rows.append(
            {
                "reference": ref,
                "prediction": pred,
                "frames": count,
                "seconds": round(count / 10.0, 3),
                "share": count / max(total, 1),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmarks/reports/chord_errors"))
    parser.add_argument("--frame", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.manifest.parent
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    global_confusion: Counter[tuple[str, str]] = Counter()
    global_root_confusion: Counter[tuple[str, str]] = Counter()
    global_quality_confusion: Counter[tuple[str, str]] = Counter()
    per_song: list[dict[str, object]] = []
    evaluated = 0

    for row in rows:
        ref_path = resolve_path(base, row.get("chord_reference", ""))
        pred_path = resolve_path(base, row.get("prediction", ""))
        if not ref_path or not pred_path or not ref_path.exists() or not pred_path.exists():
            continue
        result = analyze_pair(ref_path, pred_path, args.frame)
        evaluated += 1
        metrics = result["metrics"]
        confusion = result["confusion"]
        root_confusion = result["root_confusion"]
        quality_confusion = result["quality_confusion"]
        global_confusion.update(confusion)
        global_root_confusion.update(root_confusion)
        global_quality_confusion.update(quality_confusion)
        worst = confusion.most_common(1)
        per_song.append(
            {
                "id": row.get("id", ""),
                "root_accuracy": metrics["root_accuracy"],
                "triad_accuracy": metrics["triad_accuracy"],
                "top_error": f"{worst[0][0][0]}->{worst[0][0][1]}" if worst else "",
                "top_error_seconds": round(worst[0][1] * args.frame, 3) if worst else 0.0,
            }
        )

    total_errors = sum(global_confusion.values())
    report = {
        "evaluated": evaluated,
        "top_confusions": counter_rows(global_confusion, total_errors, args.limit),
        "top_root_confusions": counter_rows(global_root_confusion, sum(global_root_confusion.values()), args.limit),
        "top_quality_confusions": counter_rows(global_quality_confusion, sum(global_quality_confusion.values()), args.limit),
        "per_song": per_song,
    }

    json_path = args.output.with_suffix(".json")
    csv_path = args.output.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        csv_path,
        report["top_confusions"],
        ["reference", "prediction", "frames", "seconds", "share"],
    )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"evaluated: {evaluated}")
    for item in report["top_confusions"][:10]:
        print(
            f"{item['reference']} -> {item['prediction']}: "
            f"{item['seconds']:.1f}s ({item['share']:.1%} of errors)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
