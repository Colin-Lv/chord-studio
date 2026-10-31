"""Test a hard rule: a boundary between two spans with the same root is dropped.

Rationale: in real recordings a same-root quality flip (maj <-> maj7 <-> maj9) is the model
hedging, not a new chord; merging it can only help precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode_lib as D  # noqa: E402
import full_decode as F  # noqa: E402


def drop_same_root(art, rows, lam_bass=1.5, mu=0.7, tau=0.3):
    out = []
    for row in rows:
        out.append(row)
        while len(out) >= 2 and out[-2][2].split(":")[0] == out[-1][2].split(":")[0]:
            a, b = out[-2][0], out[-1][1]
            label, _cells = F._span_label(art, a, b, lam_bass, mu, tau)
            out[-2:] = [(a, b, label)]
    return out


def main():
    art = F._prepare(D.load_run(D.latest_xuni_run()))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    for beta in (2.0, 3.0, 4.0):
        for base in (12.0, 16.0, 20.0):
            for mu in (0.5, 0.7):
                segs = F.decode_viterbi(art, model, kappa=base + 10.0, kappa_root=base,
                                        beta=beta, relabel=True, gamma=2.0, mu=mu, tau=0.28)
                plain = D.score(segs, ref)
                merged = D.score(drop_same_root(art, segs, mu=mu, tau=0.28), ref)
                print(f"b{beta} base{base:.0f} mu{mu}: plain F1 {plain['f1']:.3f} "
                      f"ex {plain['exact']:.3f} | drop-same-root F1 {merged['f1']:.3f} "
                      f"ex {merged['exact']:.3f} seg {merged['segments']}")


if __name__ == "__main__":
    main()
