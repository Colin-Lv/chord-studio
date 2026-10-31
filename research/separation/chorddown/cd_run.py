"""Upload + analyze every condition, one at a time, and keep the raw results.

Condition 1 is uploaded twice (as 01_mix and 05_mix_repeat) so the two runs give a
run-to-run noise estimate on the very same audio.

Usage:  python cd_run.py [name ...]      (default: all five)
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from cd_common import WORK, analyze_project, key_analysis, rows_of, timeline, upload

COND = WORK / "cond"
RESULTS = WORK / "results"

RUNS = [
    ("01_mix", COND / "01_mix.wav"),
    ("02_truth_harm", COND / "02_truth_harm.wav"),
    ("03_sep_harm", COND / "03_sep_harm.wav"),
    ("04_sep_harm4", COND / "04_sep_harm4.wav"),
    ("05_mix_repeat", COND / "01_mix.wav"),
    ("06_sep_harm_natural", COND / "06_sep_harm_natural.wav"),
    ("07_truth_harm4", COND / "07_truth_harm4.wav"),
]


def one(name: str, path: Path) -> bool:
    print(f"== {name}  ({path.name}, {path.stat().st_size} bytes)", flush=True)
    started = time.time()
    try:
        project = upload(path)
        print(f"  uploaded project {project.get('id')} "
              f"({time.time() - started:.0f}s)", flush=True)
        result = analyze_project(project["id"])
    except Exception:
        traceback.print_exc()
        return False
    if not result:
        print(f"  FAILED {name}", flush=True)
        return False
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2),
                                          encoding="utf-8")
    info = result.get("info") or {}
    duration = float(info.get("duration") or info.get("duration_sec") or 37.326)
    rows = rows_of(result)
    grid = timeline(result, duration)
    labels = sorted({v for v in grid.values()})
    print(f"  duration={duration:.3f} rows={len(rows)} top_keys={list(result)[:12]}")
    print(f"  key_analysis={json.dumps(key_analysis(result))[:300]}")
    print(f"  labels used: {labels}")
    print(f"  timeline: {' '.join(grid[k] for k in sorted(grid))[:600]}", flush=True)
    return True


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]
    runs = [r for r in RUNS if not wanted or r[0] in wanted]
    ok = []
    for name, path in runs:
        ok.append((name, one(name, path)))
    print("\n== summary")
    for name, flag in ok:
        print(f"  {name}: {'ok' if flag else 'FAILED'}")
    return 0 if all(f for _, f in ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
