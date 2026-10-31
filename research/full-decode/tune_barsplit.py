"""Sweep the bar-split decoder: half-split threshold and the boundary-gain penalty."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode_lib as D  # noqa: E402
import full_decode as F  # noqa: E402


def main():
    art = F._prepare(D.load_run(D.latest_xuni_run()))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    rows = []
    for thr in (0.2, 0.4, 0.5, 0.6, 0.8):
        for delta in (None, 0.0, 5.0, 10.0, 20.0, 40.0):
            segs = F.decode(art, model, "barsplit", thr_half=thr, delta=delta)
            m = D.score(segs, ref)
            rows.append(dict(mode="barsplit", thr=thr, delta=delta, **m))
            print(f"thr {thr:4.2f} delta {str(delta):>5s}: segments {m['segments']:3d} "
                  f"exact {m['exact']:.3f} family {m['family']:.3f} P {m['precision']:.2f} "
                  f"R {m['recall']:.2f} F1 {m['f1']:.2f}")
    (HERE / "sweep_barsplit.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
