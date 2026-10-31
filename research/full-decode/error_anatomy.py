"""Error anatomy of the current span output against the golden reference.

Separates label errors (root / quality) from boundary errors, so the full decoder is aimed
at the part that actually costs accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode_lib as D  # noqa: E402


def at(rows, t):
    for a, b, label in rows:
        if a <= t < b:
            return label
    return "N"


def main():
    run = D.latest_xuni_run()
    art = D.load_run(run)
    ref = D.load_lab(D.GOLDEN)
    pred = D.load_lab(run / "retained10_span.lab")
    print(f"run {run.name}: ref {len(ref)} segments, span {len(pred)} segments")

    same_boundary = 0
    label_ok = 0
    root_ok = 0
    fam_ok = 0
    for i, (a, b, label) in enumerate(ref[1:], 1):
        mid = (a + b) / 2
        got = at(pred, mid)
        gr, _, gq = label.partition(":")
        pr, _, pq = got.partition(":")
        label_ok += got == label
        root_ok += gr == pr
        fam_ok += (gr == pr) and (gq[:3] == pq[:3])
        if abs(a - ref[i][0]) < 1e-9 and i < len(ref):
            pass
        if got != label:
            print(f"  {a:7.2f} {label:>9} -> {got:>9}")
    n = len(ref) - 1
    print(f"\nmidpoint label exact {label_ok}/{n} = {label_ok / n:.3f}  "
          f"root {root_ok / n:.3f}  family {fam_ok / n:.3f}")
    print(f"frame exact {D.frame_acc(pred, ref)[0]:.3f}  family {D.frame_acc(pred, ref)[1]:.3f}")

    # boundary hits / misses
    tb = [a for a, _, _ in ref[1:]]
    pb = [a for a, _, _ in pred[1:]]
    hits = [t for t in tb if any(abs(t - p) <= D.TOL for p in pb)]
    missing = [t for t in tb if not any(abs(t - p) <= D.TOL for p in pb)]
    print(f"boundaries: true {len(tb)}  pred {len(pb)}  hit {len(hits)}  miss {len(missing)}")
    print("  missed:", [round(t, 2) for t in missing])


if __name__ == "__main__":
    main()
