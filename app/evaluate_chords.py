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
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import mir_eval

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
    """Convert Harte shorthand to display notation without losing quality/bass.

    X is an unknown annotation, not silence.  In particular, replacing ``:maj``
    before ``:maj7`` silently changes a major seventh into a dominant seventh.
    """
    label = label.strip()
    if not label or label.upper() == "N":
        return "N"
    if label.upper() == "X":
        return "X"
    body, slash, bass = label.partition("/")
    if ":" not in body:
        return label
    root, quality = body.split(":", 1)
    aliases = {"maj": "", "min": "m", "min7": "m7", "min6": "m6",
               "maj6": "6", "min9": "m9", "min11": "m11", "min13": "m13",
               "hdim7": "m7b5", "minmaj7": "mMaj7"}
    quality = aliases.get(quality, quality)
    if slash and not re.fullmatch(r"[A-G][b#]*", bass):
        try:
            root_pc = mir_eval.chord.pitch_class_to_semitone(root)
            bass_pc = (root_pc + mir_eval.chord.scale_degree_to_semitone(bass)) % 12
            bass = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")[bass_pc]
        except mir_eval.chord.InvalidChordException:
            pass
    return root + quality + ("/" + bass if slash else "")


@lru_cache(maxsize=4096)
def harte_chord(label: str) -> str:
    """Normalize project display labels for the official mir_eval metrics."""
    label = str(label).strip().replace("♭", "b").replace("♯", "#")
    if not label or label.upper() == "N":
        return "N"
    if label.upper() == "X":
        return "X"
    body, slash, bass = label.partition("/")
    match = re.fullmatch(r"([A-G][b#]*)(.*)", body)
    if not match:
        return "X"
    root, suffix = match.groups()
    aliases = {"": "maj", "m": "min", "m7": "min7", "m6": "min6", "6": "maj6",
               "m9": "min9", "m11": "min11", "m13": "min13", "m7b5": "hdim7",
               "mMaj7": "minmaj7", "mmaj7": "minmaj7", "add9": "maj(9)",
               "madd9": "min(9)", "sus": "sus4", "7sus4": "sus4(b7)",
               "sus47": "sus4(b7)", "9sus4": "sus4(b7,9)", "sus49": "sus4(b7,9)",
               "7sus2": "sus2(b7)", "maj11": "maj9(11)"}
    quality = suffix[1:] if suffix.startswith(":") else aliases.get(suffix, suffix)
    if slash and re.fullmatch(r"[A-G][b#]*", bass):
        interval = (mir_eval.chord.pitch_class_to_semitone(bass)
                    - mir_eval.chord.pitch_class_to_semitone(root)) % 12
        bass = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")[interval]
    normalized = root + ":" + quality + ("/" + bass if slash else "")
    try:
        mir_eval.chord.encode(normalized)
    except (mir_eval.chord.InvalidChordException, KeyError):
        return "X"
    return normalized


def canonical_chord(label: str) -> tuple[int | None, str]:
    normalized = harte_chord(label)
    if normalized in {"N", "X"}:
        return None, normalized
    root, bitmap, _ = mir_eval.chord.encode(normalized)
    signature = tuple(int(value) for value in bitmap[:9])
    known = {(0, 4, 7): "", (0, 3, 7): "m", (0, 3, 6): "dim",
             (0, 4, 8): "aug", (0, 2, 7): "sus2", (0, 5, 7): "sus4"}
    intervals = tuple(index for index, value in enumerate(signature) if value)
    return int(root), known.get(intervals, str(signature))


def is_reference_chord_label(label: str) -> bool:
    return label.strip().upper() in {"N", "X"} or bool(re.match(r"^[A-G][b#]*(?::|m|dim|aug|sus|add|[0-9]|/|\(|$)", label.strip()))


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
        return sorted(rows, key=lambda item: (item.start, item.end))

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
    return sorted(rows, key=lambda item: (item.start, item.end))


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


def validate_timeline(rows: list[TimedChord], name: str) -> list[TimedChord]:
    ordered = sorted(rows, key=lambda item: (item.start, item.end))
    previous_end = -math.inf
    for index, row in enumerate(ordered):
        if not math.isfinite(row.start) or not math.isfinite(row.end) or row.end <= row.start:
            raise ValueError(f"{name} has invalid interval: {row}")
        if row.start < previous_end - 1e-6:
            # Isophonics TTL stores onset and duration independently at 1 ms;
            # adding then rounding them creates 1–2 ms overlaps. Resolve only
            # this precision artifact at the next onset; reject real overlaps.
            previous = ordered[index - 1]
            if previous_end - row.start <= 0.002001 and row.start > previous.start:
                ordered[index - 1] = TimedChord(previous.start, row.start, previous.chord)
            else:
                raise ValueError(f"{name} has overlapping intervals near {row.start:.6f}s")
        previous_end = row.end
    return ordered


def aligned_intervals(reference: list[TimedChord], prediction: list[TimedChord]):
    """Yield exact partitions of annotated time; omitted predictions stay missing.

    Annotation gaps are unknown (excluded), while explicit N intervals are scored.
    Reference support is never truncated to prediction support.
    """
    reference = validate_timeline(reference, "Reference")
    prediction = validate_timeline(prediction, "Prediction")
    if not reference:
        raise ValueError("Reference must contain timed chords.")
    start, end = reference[0].start, reference[-1].end
    boundaries = sorted({start, end, *[x for row in reference + prediction
                                     for x in (row.start, row.end) if start <= x <= end]})
    ri = pi = 0
    for left, right in zip(boundaries, boundaries[1:]):
        middle = (left + right) * 0.5
        while ri < len(reference) and reference[ri].end <= middle:
            ri += 1
        while pi < len(prediction) and prediction[pi].end <= middle:
            pi += 1
        ref = reference[ri] if ri < len(reference) and reference[ri].start <= middle else None
        pred = prediction[pi] if pi < len(prediction) and prediction[pi].start <= middle else None
        yield left, right, ref, pred


def evaluate(reference: list[TimedChord], prediction: list[TimedChord], frame: float = 0.1) -> dict[str, float]:
    """Exact duration metrics. ``frame`` only controls legacy equivalent counts.

    root/triad_accuracy retain the project's chord-only denominator; *_wcsr
    additionally score explicit N regions using the standard mir_eval rules.
    """
    if frame <= 0 or not math.isfinite(frame):
        raise ValueError("frame must be finite and positive")
    aligned = list(aligned_intervals(reference, prediction))
    weights = np.asarray([right - left for left, right, _, _ in aligned])
    refs = [harte_chord(ref.chord) if ref else "X" for _, _, ref, _ in aligned]
    preds = [harte_chord(pred.chord) if pred else "N" for _, _, _, pred in aligned]
    known = np.asarray([value != "X" for value in refs])
    chord = np.asarray([value not in {"N", "X"} for value in refs])
    present = np.asarray([pred is not None for _, _, _, pred in aligned])
    seconds = float(weights[known].sum())
    chord_seconds = float(weights[chord].sum())
    result = {
        "frames": seconds / frame,
        "evaluated_seconds": seconds,
        "reference_span_seconds": float(weights.sum()),
        "unknown_reference_seconds": float(weights[~known].sum()),
        "non_no_chord_frames": chord_seconds / frame,
        "non_no_chord_seconds": chord_seconds,
        "prediction_coverage": float(weights[known & present].sum() / seconds) if seconds else 0.0,
        "missing_prediction_seconds": float(weights[known & ~present].sum()),
        "unknown_prediction_seconds": float(sum(w for w, label in zip(weights, preds) if label == "X")),
    }
    for name, rows in (("reference", reference), ("prediction", prediction)):
        ordered = sorted(rows, key=lambda row: (row.start, row.end))
        result[name + "_rounding_overlap_seconds"] = sum(max(0, left.end - right.start)
                                                         for left, right in zip(ordered, ordered[1:]))
    for name, comparator in (("root", mir_eval.chord.root), ("triad", mir_eval.chord.triads),
                             ("tetrad", mir_eval.chord.tetrads), ("tetrad_inv", mir_eval.chord.tetrads_inv)):
        # Missing output is distinct from an explicit N prediction, including
        # over reference silence. The synthetic N is only a parser placeholder.
        scores = np.where(present, comparator(refs, preds), 0.0)
        result[name + "_accuracy"] = float(np.sum(weights[chord] * scores[chord]) / chord_seconds) if chord_seconds else 0.0
        result[name + "_wcsr"] = float(np.sum(weights[known] * scores[known]) / seconds) if seconds else 0.0
    no_chord = np.asarray([value == "N" for value in refs])
    no_seconds = float(weights[no_chord].sum())
    result["no_chord_seconds"] = no_seconds
    result["no_chord_accuracy"] = float(sum(w for w, ref, pred, supplied in zip(weights, refs, preds, present)
                                            if supplied and ref == pred == "N") / no_seconds) if no_seconds else 0.0
    return result


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
