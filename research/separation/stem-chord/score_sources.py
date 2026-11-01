"""Score every chord source in an analysis against the golden timeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

GOLDEN = Path(r"D:\ChordStudio\Dialogue\research\golden\reference_golden.lab")
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def read_golden():
    events = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            events.append((float(parts[0]), float(parts[1]), parts[2]))
    return events


def frames(rows, step=0.25):
    out = []
    for row in rows:
        t = row["start"]
        while t < row["end"] - 1e-9:
            out.append((t, row["label"]))
            t += step
    return out


def score(rows, events):
    grid = frames(rows)
    total = hit = root = 0
    for t, label in grid:
        truth = next((lab for start, end, lab in events if start <= t < end), None)
        if truth is None:
            continue
        total += 1
        hit += label == truth
        root += label.split(":")[0] == truth.split(":")[0]
    # segment majority
    seg_total = seg_hit = 0
    for start, end, truth in events:
        inner = [lab for t, lab in grid if start <= t < end]
        if not inner:
            continue
        values = {}
        for lab in inner:
            values[lab] = values.get(lab, 0) + 1
        guess = max(values, key=values.get)
        seg_total += 1
        seg_hit += guess == truth
    return hit / max(1, total), root / max(1, total), seg_hit / max(1, seg_total), total


def main():
    run_dir = RUN
    result = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    events = read_golden()
    print(f"{'source':>18s}  frame  root | segment")
    for source in result["sources"]:
        fe, fr, se, total = score(source["rows"], events)
        print(f"{source['id']:>18s} {fe * 100:5.2f} {fr * 100:5.2f} | {se * 100:5.2f}  ({total} frames)")
    print("default:", result.get("default_source"))


if __name__ == "__main__":
    main()
