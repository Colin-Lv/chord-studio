#!/usr/bin/env python
"""Evaluate chord predictions by musical attributes instead of flat labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from chord_studio import parse_chord_name
from evaluate_chords import TimedChord, aligned_intervals, convert_annotation_chord, harte_chord
import mir_eval
from train_candidate_reranker import resolve_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for bar in payload.get("bars", []) or []:
        for span in bar.get("spans") or []:
            item = dict(span)
            item["bar"] = int(bar.get("bar") or item.get("bar") or 0)
            spans.append(item)
    return sorted(spans, key=lambda row: (float(row.get("start") or 0.0), float(row.get("end") or 0.0)))


def label(span: dict[str, Any]) -> str:
    return str(span.get("chord_display") or span.get("chord") or "N")


def overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(left.get("end") or 0.0), float(right.get("end") or 0.0))
        - max(float(left.get("start") or 0.0), float(right.get("start") or 0.0)),
    )


def best_match(span: dict[str, Any], references: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_overlap = 0.0
    for reference in references:
        amount = overlap(span, reference)
        if amount > best_overlap:
            best = reference
            best_overlap = amount
    return best, best_overlap


def bass_root(chord: str) -> int | None:
    if "/" not in chord:
        return None
    bass = chord.split("/", 1)[1]
    pc, _ = parse_chord_name(bass)
    return pc


def quality(suffix: str) -> str:
    low = suffix.lower()
    if low in {"", "maj7", "6", "add9"}:
        return "major"
    if low in {"m", "m7", "m6", "m9", "madd9", "mmaj7"}:
        return "minor"
    if low in {"dim", "dim7", "m7b5"}:
        return "diminished"
    if low.startswith("sus"):
        return "sus"
    if low in {"7", "9"}:
        return "dominant"
    if low == "aug":
        return "augmented"
    return low or "major"


def seventh(suffix: str) -> str:
    low = suffix.lower()
    if low == "dim7":
        return "bb7"
    if "m7b5" in low or low in {"7", "9", "11", "13", "m7", "m9", "m11", "m13", "7sus4", "9sus4"}:
        return "b7"
    if "maj7" in low or "mmaj7" in low:
        return "maj7"
    return "none"


def extensions(suffix: str) -> set[str]:
    low = suffix.lower()
    values = set()
    if "add9" in low or re.search(r"(^|[^a-z])9", low):
        values.add("9")
    if "6" in low:
        values.add("6")
    if "sus2" in low:
        values.add("sus2")
    if "sus4" in low or low == "sus":
        values.add("sus4")
    if "m7b5" in low:
        values.add("b5")
    return values


def attrs(chord: str) -> dict[str, object]:
    chord = convert_annotation_chord(str(chord or "N"))
    body = chord.split("/", 1)[0]
    root, suffix = parse_chord_name(body)
    if root is None or chord == "N":
        unknown = harte_chord(chord) == "X"
        return {
            "root": -1 if unknown else None,
            "quality": "X" if unknown else "N",
            "seventh": "X" if unknown else "N",
            "extensions": frozenset(),
            "bass": None,
            "has_slash": False,
        }
    return {
        "root": root,
        "quality": quality(suffix),
        "seventh": seventh(suffix),
        "extensions": frozenset(extensions(suffix)),
        "bass": bass_root(chord),
        "has_slash": "/" in chord,
    }


def compare_attrs(reference: str, prediction: str) -> dict[str, int]:
    ref = attrs(reference)
    pred = attrs(prediction)
    ref_encoded = mir_eval.chord.encode(harte_chord(reference), reduce_extended_chords=True)
    pred_encoded = mir_eval.chord.encode(harte_chord(prediction), reduce_extended_chords=True)
    exact = (harte_chord(reference) != "X" and ref_encoded[0] == pred_encoded[0]
             and (ref_encoded[1] == pred_encoded[1]).all() and ref_encoded[2] == pred_encoded[2])
    return {
        "exact": int(exact),
        "root": int(ref["root"] == pred["root"]),
        "quality": int(ref["quality"] == pred["quality"]),
        "seventh": int(ref["seventh"] == pred["seventh"]),
        "extensions": int(ref["extensions"] == pred["extensions"]),
        "bass": int(ref["bass"] == pred["bass"]),
        "slash": int(ref["has_slash"] == pred["has_slash"]),
    }


def evaluate(reference_path: Path, prediction_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    references = iter_spans(load_json(reference_path))
    predictions = iter_spans(load_json(prediction_path))
    ref_timeline = [TimedChord(float(item["start"]), float(item["end"]), label(item)) for item in references]
    pred_timeline = [TimedChord(float(item["start"]), float(item["end"]), label(item)) for item in predictions]
    ref_metadata = {(float(item["start"]), label(item)): item for item in references}
    rows = []
    totals: dict[str, float] = defaultdict(float)
    denominators: dict[str, float] = defaultdict(float)
    evaluated_seconds = missing_seconds = unknown_seconds = 0.0
    for start, end, ref, pred in aligned_intervals(ref_timeline, pred_timeline):
        amount = end - start
        if ref is None or harte_chord(ref.chord) == "X":
            unknown_seconds += amount
            continue
        ref_label = ref.chord
        pred_label = pred.chord if pred else "N"
        evaluated_seconds += amount
        if pred is None:
            missing_seconds += amount
        scores = compare_attrs(ref_label, pred_label)
        metadata = ref_metadata[(ref.start, ref.chord)]
        root_only = bool(metadata.get("root_locked")) or metadata.get("label_strength") == "root_only"
        if root_only:
            scores = {key: value if key == "root" else None for key, value in scores.items()}
        for key, value in scores.items():
            if value is not None:
                totals[key] += amount * value
                denominators[key] += amount
        rows.append(
            {
                "bar": next((item.get("bar") for item in predictions if float(item["start"]) <= start < float(item["end"])), ""),
                "start": start,
                "end": end,
                "reference": ref_label,
                "prediction": pred_label,
                "overlap": f"{amount:.6f}",
                **scores,
            }
        )
    summary = {
        "reference": str(reference_path),
        "prediction": str(prediction_path),
        "segments": len(rows),
        "evaluation_protocol": "exact duration-weighted intersection partitions over full reference; missing predictions=N; X excluded",
        "evaluated_seconds": evaluated_seconds,
        "missing_prediction_seconds": missing_seconds,
        "unknown_reference_seconds": unknown_seconds,
        "metric_denominator_seconds": dict(denominators),
        "metrics": {
            key: round(value / denominators[key], 4) if denominators[key] else 0.0
            for key, value in sorted(totals.items())
        },
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, summary = evaluate(resolve_path(args.reference), resolve_path(args.prediction))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.report:
        report_path = resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
