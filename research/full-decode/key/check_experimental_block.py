"""Offline check of the experimental key block against the stage-3 reference numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND / "app" / "web"))
sys.path.insert(0, str(BACKEND / "app"))

PROJECTS = {
    "虚拟": "aa5a41797bb343e590abcf4d8b27ae54",
    "坏女孩": "0bc0304e01ed40d8baac6b76575b25b6",
    "沧海一粟": "e18b69d0edef4c5e91f6594520a816cb",
}


def latest_run(project):
    runs = sorted((Path(r"D:\ChordStudio\data\projects") / project / "runs").iterdir(),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1]


def main():
    from source_rows import lab_rows
    from key_experimental import evaluate

    for title, project in PROJECTS.items():
        run = latest_run(project)
        rows = lab_rows(run / "retained10_stable.lab")
        key = json.loads((run / "key.json").read_text(encoding="utf-8"))
        analysis = json.loads((run / "analysis.json").read_text(encoding="utf-8"))
        published = (analysis.get("key_analysis") or {}).get("key")
        block = evaluate(rows, key["candidates"], published_key=published)
        top = block["ranking"][:3]
        print(f"{title} (run {run.name[:8]})")
        print(f"  published {published}  acoustic {block['acoustic_key']}  "
              f"experimental {block['key']}  gap {block['gap']}")
        print(f"  rows {len(rows)} parsed {block['parse']['tokens']}->{block['chords']} "
              f"N {block['parse']['N']} X {block['parse']['X']} missing {block['parse']['missing']}")
        for row in top:
            print(f"    {row['key']:>3} score {row['score']:+.4f} "
                  f"(ac {row['acoustic_term']:+.3f} harm {row['harmonic_term']:+.3f}) "
                  f"native {row['native_share']:.3f} tonic {row['tonic_share']:.3f} "
                  f"arr {row['authentic_arrivals']} ret {row['predominant_returns']}")


if __name__ == "__main__":
    main()
