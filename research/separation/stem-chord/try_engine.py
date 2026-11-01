"""Try the installed fusion engine on the project that has an accepted golden timeline."""
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


def main():
    store = Store(Path(r"D:\ChordStudio\data\projects"))
    project = store.project(PROJECT)
    meta = fuse_project(store, project, "bs_roformer_sw", make_default=False)
    print("fusion meta:", json.dumps(meta, ensure_ascii=False))

    run_dir = store.run_dir(PROJECT, project["current_run"])
    result = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    source = next(item for item in result["sources"] if item["id"] == "stem_fusion")

    events = golden()
    total = hit = root_hit = 0
    for row in source["rows"]:
        inner = [lab for start, end, lab in events if row["start"] <= (start + end) / 2 < row["end"]]
        if not inner:
            continue
        truth = inner[0]
        total += 1
        hit += row["label"] == truth
        root_hit += row["label"].split(":")[0] == truth.split(":")[0]
    print(f"fused rows {len(source['rows'])}  rows matched {total}  exact {hit}/{total}")


if __name__ == "__main__":
    main()
