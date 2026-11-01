"""Segment-level scoring against the golden timeline (majority label per golden segment)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WORK = Path(r"D:\ChordStudio\research\separation\stem-chord")
GOLDEN = Path(r"D:\ChordStudio\Dialogue\research\golden\reference_golden.lab")


def read_golden():
    events = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            events.append((float(parts[0]), float(parts[1]), parts[2]))
    return events


def segment_accuracy(labels, times, events):
    """For every golden segment, take the model's majority label inside it."""
    exact = root = weight_exact = 0.0
    total = weight = 0.0
    per = []
    for start, end, truth in events:
        inner = [lab for t, lab in zip(times, labels) if start <= t < end]
        if not inner:
            continue
        values, counts = np.unique(inner, return_counts=True)
        guess = values[counts.argmax()]
        duration = end - start
        total += 1
        weight += duration
        ok = guess == truth
        ok_root = guess.split(":")[0] == truth.split(":")[0]
        exact += ok
        root += ok_root
        weight_exact += ok * duration
        per.append((start, end, truth, guess, ok, ok_root))
    return (exact / max(1, total), root / max(1, total),
            weight_exact / max(1, weight), per)


if __name__ == "__main__":
    data = json.loads((WORK / "fusion-result.json").read_text(encoding="utf-8"))
    print(data)
