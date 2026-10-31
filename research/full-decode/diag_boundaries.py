"""Where do false boundaries come from? Dump the decoder output against the golden reference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode_lib as D  # noqa: E402
import full_decode as F  # noqa: E402


def at(rows, t):
    for a, b, label in rows:
        if a <= t < b:
            return label
    return "N"


def main():
    art = F._prepare(D.load_run(D.latest_xuni_run()))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    rows = F.decode(art, model, "barsplit", thr_half=0.5, delta=10.0)
    m = D.score(rows, ref)
    print(f"segments {m['segments']} exact {m['exact']:.3f} family {m['family']:.3f} "
          f"P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f}\n")
    tb = [a for a, _, _ in ref[1:]]
    print(f"{'start':>7} {'label':>9}  golden-at-midpoint")
    for i, (a, b, label) in enumerate(rows):
        mid = (a + b) / 2
        truth = at(ref, mid)
        if i == 0:
            mark = "start"
        elif any(abs(a - t) <= D.TOL for t in tb):
            mark = "TRUE"
        else:
            mark = "FALSE"
        print(f"{a:7.2f} {label:>9}  {truth:>9}  {mark}")
    print("\nmissed true boundaries:")
    pb = [a for a, _, _ in rows[1:]]
    for t in tb:
        if not any(abs(t - p) <= D.TOL for p in pb):
            print(f"  {t:7.2f}  {at(ref, t + 0.01)}")


if __name__ == "__main__":
    main()
