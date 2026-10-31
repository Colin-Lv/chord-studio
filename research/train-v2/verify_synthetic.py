"""Acceptance check for the synthetic-domain goal: held-out joint >= 0.95, 7th family >= 0.85.

Reads the frozen evaluation written by eval_full.py (5 folds split by song, pitch-rotation
TTA). No re-training, no narration: the numbers come from the saved fold checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main(tag="fix60"):
    cv = json.loads((RUNS / f"cv_{tag}.json").read_text(encoding="utf-8"))
    ev = json.loads((RUNS / f"eval_{tag}.json").read_text(encoding="utf-8"))
    joint = ev["overall"]["joint"]
    folds = [r["joint"] for r in cv["results"]]
    seventh = {row["quality"]: row["joint"] for row in ev["per_quality"]
               if row["quality"] in ("7", "maj7", "min7")}
    print(f"tag {tag}: folds {[round(f, 4) for f in folds]}")
    print(f"  single-pass mean joint {sum(folds) / len(folds):.4f}")
    print(f"  TTA joint {joint:.4f}  root {ev['overall']['root']:.4f} "
          f"quality {ev['overall']['quality']:.4f}  n={ev['overall']['n']}")
    print(f"  seventh family {seventh}")
    ok_joint = joint >= 0.95
    ok_seventh = all(v >= 0.85 for v in seventh.values())
    print(f"  joint >= 0.95      : {'PASS' if ok_joint else 'FAIL'}")
    print(f"  7th family >= 0.85 : {'PASS' if ok_seventh else 'FAIL'}")
    print("  overall:", "PASS" if (ok_joint and ok_seventh) else "FAIL")
    out = dict(tag=tag, folds=folds, single_mean=sum(folds) / len(folds), tta_joint=joint,
               root=ev["overall"]["root"], quality=ev["overall"]["quality"],
               n=ev["overall"]["n"], seventh=seventh, pass_joint=ok_joint,
               pass_seventh=ok_seventh)
    (HERE / "verify_synthetic.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0 if (ok_joint and ok_seventh) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "fix60"))
