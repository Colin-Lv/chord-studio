"""Sweep the fusion weight with the full decoder and report accuracy against the golden."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND))

from app.web.fusion_engine import fuse_project  # noqa: E402
from app.web.storage import Store  # noqa: E402

PROJECT = "aa5a41797bb343e590abcf4d8b27ae54"
GOLDEN = Path(r"D:\ChordStudio\Dialogue\research\golden\reference_golden.lab")


def golden():
    events = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            events.append((float(parts[0]), float(parts[1]), parts[2]))
    return events


def score(rows, events):
    grid = []
    for row in rows:
        t = row["start"]
        while t < row["end"] - 1e-9:
            grid.append((t, row["label"]))
            t += 0.25
    total = hit = root = 0
    seg_total = seg_hit = 0
    for t, label in grid:
        truth = next((lab for start, end, lab in events if start <= t < end), None)
        if truth is None:
            continue
        total += 1
        hit += label == truth
        root += label.split(":")[0] == truth.split(":")[0]
    for start, end, truth in events:
        inner = [lab for t, lab in grid if start <= t < end]
        if not inner:
            continue
        counts = {}
        for lab in inner:
            counts[lab] = counts.get(lab, 0) + 1
        seg_total += 1
        seg_hit += max(counts, key=counts.get) == truth
    return hit / max(1, total), root / max(1, total), seg_hit / max(1, seg_total)


def main():
    store = Store(Path(r"D:\ChordStudio\data\projects"))
    project = store.project(PROJECT)
    events = golden()
    rows = []
    for lam in (0.0, 0.15, 0.25, 0.35, 0.5, 0.65):
        meta = fuse_project(store, project, "bs_roformer_sw", lam=lam, make_default=False)
        run_dir = store.run_dir(PROJECT, project["current_run"])
        result = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
        source = next(item for item in result["sources"] if item["id"] == "stem_fusion")
        fe, fr, se = score(source["rows"], events)
        rows.append((lam, fe, fr, se, meta.get("decoder"), meta.get("segments")))
        print(f"lambda={lam:.2f}  frame {fe * 100:5.2f}  root {fr * 100:5.2f}  segment {se * 100:5.2f}  "
              f"decoder={meta.get('decoder')} segments={meta.get('segments')}")
    best = max(rows, key=lambda r: r[1])
    print("best by frame:", best)
    Path(r"D:\ChordStudio\research\separation\stem-chord\sweep-full.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
