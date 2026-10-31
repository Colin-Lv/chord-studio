#!/usr/bin/env python
"""Evaluate chord-boundary timing between two Chord Studio analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from mir_eval.util import match_events


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_label(event: dict[str, Any]) -> str:
    return str(event.get("chord_display") or event.get("chord") or "N")


def boundaries(payload: dict[str, Any], include_same_label: bool = False) -> list[tuple[float, int, str]]:
    events = sorted(payload.get("events", []), key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    rows: list[tuple[float, int, str]] = []
    previous_label = None
    for index, event in enumerate(events):
        start = float(event.get("start", 0.0))
        label = event_label(event)
        if index == 0 or include_same_label or label != previous_label:
            rows.append((start, int(event.get("bar", 0)), label))
        previous_label = label
    return rows


def match_boundaries(
    reference: list[tuple[float, int, str]],
    prediction: list[tuple[float, int, str]],
    tolerance: float,
) -> tuple[int, list[tuple[float, int, str]], list[tuple[float, int, str]]]:
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Boundary tolerance must be finite and nonnegative.")
    # Nearest-first greedy matching can consume the only candidate for the next
    # reference boundary. Maximum bipartite matching gives the correct F1 count.
    matches = match_events(np.asarray([item[0] for item in reference]),
                           np.asarray([item[0] for item in prediction]), tolerance)
    matched_ref = {left for left, _ in matches}
    matched_pred = {right for _, right in matches}
    missed = [item for index, item in enumerate(reference) if index not in matched_ref]
    extra = [item for index, item in enumerate(prediction) if index not in matched_pred]
    return len(matches), missed, extra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = boundaries(load(args.reference))
    prediction = boundaries(load(args.prediction))
    hits, missed, extra = match_boundaries(reference, prediction, args.tolerance)
    precision = hits / max(len(prediction), 1)
    recall = hits / max(len(reference), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"reference_boundaries: {len(reference)}")
    print(f"prediction_boundaries: {len(prediction)}")
    print(f"hits: {hits}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1: {f1:.3f}")
    print("missed:")
    for time, bar, label in missed[: args.limit]:
        print(f"  {time:.3f}s bar {bar}: {label}")
    print("extra:")
    for time, bar, label in extra[: args.limit]:
        print(f"  {time:.3f}s bar {bar}: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
