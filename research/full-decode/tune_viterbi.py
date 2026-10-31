"""Sweep the Viterbi segmentation: switch costs and the boundary-classifier discount."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode_lib as D  # noqa: E402
import full_decode as F  # noqa: E402


def main():
    art = F._prepare(D.load_run(D.latest_xuni_run()))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    rows = []
    best = None
    for beta in (2.0, 3.0, 4.0, 5.0, 6.0):
        for kappa in (8.0, 12.0, 16.0, 20.0, 25.0, 30.0):
            for kappa_root in (kappa + 30.0,):
                segs = F.decode_viterbi(art, model, kappa=kappa, kappa_root=kappa_root, beta=beta)
                m = D.score(segs, ref)
                rows.append(dict(mode="viterbi", kappa=kappa, kappa_root=kappa_root, beta=beta, **m))
                tag = f"beta {beta:.0f} kappa {kappa:4.0f}"
                print(f"{tag:18s} segments {m['segments']:3d} exact {m['exact']:.3f} "
                      f"family {m['family']:.3f} P {m['precision']:.2f} R {m['recall']:.2f} "
                      f"F1 {m['f1']:.2f}")
                key = (m["f1"], m["exact"])
                if best is None or key > best[0]:
                    best = (key, tag)
    print("best:", best[1], best[0])
    (HERE / "sweep_viterbi.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
