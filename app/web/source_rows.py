"""Shared LAB parsing for result creation and editor provenance checks."""
import math
from pathlib import Path


def lab_rows(path):
    rows, previous = [], 0.0
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        left, right, label = line.split(maxsplit=2)
        start, end = float(left), float(right)
        if not math.isfinite(start + end) or start < previous or end <= start or start < 0:
            raise ValueError(f"Invalid LAB timeline: {path}")
        rows.append(dict(id=len(rows), start=start, end=end, start_text=left, end_text=right, label=label))
        previous = end
    if not rows:
        raise ValueError(f"Empty LAB: {path}")
    return rows
