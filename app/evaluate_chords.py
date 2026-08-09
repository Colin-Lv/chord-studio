#!/usr/bin/env python
"""Evaluate Chord Studio output against chord annotations.

Supported reference formats:
- .lab/.txt: start end chord
- .csv: start,end,chord columns

Prediction input is the *_analysis.json produced by chord_studio.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chord_studio import chord_quality_family, parse_chord_name


@dataclass(frozen=True)
class TimedChord:
    start: float
    end: float
    chord: str


def parse_seconds(value: object) -> float:
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def convert_annotation_chord(label: str) -> str:
    label = label.strip()
    if not label or label in {"N", "X", "n", "x"}:
        return "N"
    label = label.split("/", 1)[0]
    label = label.replace(":maj", "")
    label = label.replace(":min", "m")
    label = label.replace(":dim", "dim")
    label = label.replace(":aug", "aug")
    label = label.replace(":sus4", "sus4")
    label = label.replace(":sus2", "sus2")
    label = label.replace(":7", "7")
    label = label.replace(":maj7", "maj7")
    label = label.replace(":min7", "m7")
    label = label.replace(":", "")
    return label


def canonical_chord(label: str) -> tuple[int | None, str]:
    label = convert_annotation_chord(label)
    root, suffix = parse_chord_name(label)
    if root is None:
        return None, "N"
    family = chord_quality_family(suffix)
    if family == "dom":
        family = ""
    if family not in {"", "m", "dim", "aug", "sus"}:
        family = "" if not suffix.startswith("m") else "m"
    return root, family


def is_reference_chord_label(label: str) -> bool:
    converted = convert_annotation_chord(label)
    if converted == "N":
        return True
    root, _ = parse_chord_name(converted)
    return root is not None


def read_reference(path: Path) -> list[TimedChord]:
    if path.suffix.lower() == ".csv":
        rows: list[TimedChord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                chord = row.get("chord") or row.get("label") or row.get("name")
                if chord is None:
                    continue
                if not is_reference_chord_label(str(chord)):
                    continue
                rows.append(
                    TimedChord(
                        parse_seconds(row["start"]),
                        parse_seconds(row["end"]),
                        str(chord),
                    )
                )
        return rows

    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if not is_reference_chord_label(parts[2]):
            continue
        rows.append(TimedChord(parse_seconds(parts[0]), parse_seconds(parts[1]), parts[2]))
    return sorted(rows, key=lambda item: (item.start, item.end))


def read_prediction(path: Path) -> list[TimedChord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events", [])
    rows = []
    for event in events:
        rows.append(
            TimedChord(
                float(event["start"]),
                float(event["end"]),
                str(event.get("chord_display") or event.get("chord") or "N"),
            )
        )
    return rows


def active_at(rows: list[TimedChord], times: np.ndarray) -> list[str]:
    labels = []
    index = 0
    for time in times:
        while index + 1 < len(rows) and time >= rows[index].end:
            index += 1
        if not rows or time < rows[index].start or time >= rows[index].end:
            labels.append("N")
        else:
            labels.append(rows[index].chord)
    return labels


def evaluate(reference: list[TimedChord], prediction: list[TimedChord], frame: float) -> dict[str, float]:
    if not reference or not prediction:
        raise ValueError("Both reference and prediction must contain timed chords.")

    start = max(min(item.start for item in reference), min(item.start for item in prediction))
    end = min(max(item.end for item in reference), max(item.end for item in prediction))
    if end <= start:
        raise ValueError("Reference and prediction do not overlap in time.")

    times = np.arange(start + frame / 2.0, end, frame)
    if times.size == 0:
        raise ValueError("No evaluation frames were produced; use a smaller --frame.")

    ref_labels = active_at(reference, times)
    pred_labels = active_at(prediction, times)
    root_hits = 0
    chord_hits = 0
    non_no_chord = 0
    for ref_label, pred_label in zip(ref_labels, pred_labels):
        ref_root, ref_family = canonical_chord(ref_label)
        pred_root, pred_family = canonical_chord(pred_label)
        if ref_root is None:
            continue
        non_no_chord += 1
        if pred_root == ref_root:
            root_hits += 1
            if pred_family == ref_family:
                chord_hits += 1

    denominator = max(non_no_chord, 1)
    return {
        "frames": float(times.size),
        "evaluated_seconds": float(times.size * frame),
        "non_no_chord_frames": float(non_no_chord),
        "root_accuracy": root_hits / denominator,
        "triad_accuracy": chord_hits / denominator,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chord Studio JSON against chord annotations.")
    parser.add_argument("reference", type=Path, help="Reference .lab/.txt/.csv chord annotations.")
    parser.add_argument("prediction", type=Path, help="Chord Studio *_analysis.json file.")
    parser.add_argument("--frame", type=float, default=0.1, help="Evaluation frame size in seconds.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.frame <= 0 or not math.isfinite(args.frame):
        print("--frame must be a positive number.", file=sys.stderr)
        return 2
    reference = read_reference(args.reference)
    prediction = read_prediction(args.prediction)
    try:
        metrics = evaluate(reference, prediction, args.frame)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for key, value in metrics.items():
        if key.endswith("accuracy"):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
