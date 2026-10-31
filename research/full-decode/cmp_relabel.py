"""Compare Viterbi labels with pooled relabelling of the Viterbi segments."""
from __future__ import annotations

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
    for beta in (3.0, 4.0, 5.0):
        for kappa in (12.0, 16.0, 20.0):
            for relabel in (False, True):
                segs = F.decode_viterbi(art, model, kappa=kappa, kappa_root=kappa + 30.0,
                                        beta=beta, relabel=relabel)
                m = D.score(segs, ref)
                print(f"beta {beta} kappa {kappa:4.0f} relabel {str(relabel):5s}: "
                      f"seg {m['segments']:3d} exact {m['exact']:.3f} family {m['family']:.3f} "
                      f"P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f}")


if __name__ == "__main__":
    main()
