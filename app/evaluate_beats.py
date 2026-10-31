#!/usr/bin/env python
"""Evaluate beat and downbeat timing against an analysis JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mir_eval


@dataclass(frozen=True)
class BeatMark:
    time: float
    beat: int | None = None


def parse_seconds(value: object) -> float:
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds


def read_reference(path: Path) -> list[BeatMark]:
    rows: list[BeatMark] = []
    raw = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
    if isinstance(raw, dict):
        raw = raw.get("beats", raw.get("downbeats", []))
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                rows.append(BeatMark(float(item["time"]), int(item["beat"]) if "beat" in item else None))
            else:
                rows.append(BeatMark(float(item[0]), int(item[1]) if len(item) > 1 else None))
        return sorted(rows, key=lambda item: item.time)

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        beat = int(float(parts[1])) if len(parts) > 1 else None
        rows.append(BeatMark(parse_seconds(parts[0]), beat))
    return sorted(rows, key=lambda item: item.time)


def read_prediction(path: Path) -> list[BeatMark]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for bar in data.get("bars", []):
        for beat in bar.get("beats", []):
            rows.append(BeatMark(float(beat["start"]), int(beat.get("beat", 0)) or None))
    return sorted(rows, key=lambda item: item.time)


def greedy_match(reference: list[float], prediction: list[float], tolerance: float) -> tuple[int, list[float]]:
    """Maximum one-to-one matching (legacy public name kept for callers).

    Nearest-first greedy matching can consume the only match available to the
    next reference beat and undercount correct predictions.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative.")
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(pred)) or np.any(ref < 0) or np.any(pred < 0):
        raise ValueError("Beat times must be finite and nonnegative.")
    if ref.size == 0 or pred.size == 0:
        return 0, []
    matching = mir_eval.util.match_events(ref, pred, tolerance)
    return len(matching), [float(pred[right] - ref[left]) for left, right in matching]


def f_measure(reference: list[float], prediction: list[float], tolerance: float) -> dict[str, float]:
    hits, errors = greedy_match(reference, prediction, tolerance)
    precision = hits / max(len(prediction), 1)
    recall = hits / max(len(reference), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    abs_errors = [abs(value) for value in errors]
    return {
        "reference_count": float(len(reference)),
        "prediction_count": float(len(prediction)),
        "hits": float(hits),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_abs_error_ms": float(np.median(abs_errors) * 1000.0) if abs_errors else math.nan,
    }


def evaluate(reference: list[BeatMark], prediction: list[BeatMark], tolerance: float) -> dict[str, float]:
    ref_beats = [item.time for item in reference]
    pred_beats = [item.time for item in prediction]
    metrics = {f"beat_{key}": value for key, value in f_measure(ref_beats, pred_beats, tolerance).items()}

    ref_downbeats = [item.time for item in reference if item.beat == 1]
    pred_downbeats = [item.time for item in prediction if item.beat == 1]
    if ref_downbeats:
        metrics.update(
            {
                f"downbeat_{key}": value
                for key, value in f_measure(ref_downbeats, pred_downbeats, tolerance).items()
            }
        )
    return metrics


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate beat/downbeat timing.")
    parser.add_argument("reference", type=Path, help=".beats/.txt/.json reference beat annotations")
    parser.add_argument("prediction", type=Path, help="Chord Studio *_analysis.json")
    parser.add_argument("--tolerance", type=float, default=0.07, help="Hit tolerance in seconds.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        print("--tolerance must be finite and nonnegative.", file=sys.stderr)
        return 2
    reference = read_reference(args.reference)
    prediction = read_prediction(args.prediction)
    if not reference or not prediction:
        print("Both reference and prediction must contain beats.", file=sys.stderr)
        return 2
    metrics = evaluate(reference, prediction, args.tolerance)
    for key, value in metrics.items():
        if key.endswith("f1") or key.endswith("precision") or key.endswith("recall"):
            print(f"{key}: {value:.3f}")
        elif math.isnan(value):
            print(f"{key}: nan")
        else:
            print(f"{key}: {value:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
