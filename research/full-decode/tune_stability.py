"""Stability check around the semi-Markov configuration that reached the target."""
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
    cells = []
    for lam in (16.0, 18.0, 20.0, 22.0, 24.0):
        for beta in (6.0, 8.0, 10.0):
            for center in (7.0, 8.0, 9.0):
                for sigma in (0.30, 0.35, 0.40):
                    segs = F.decode_semimarkov(art, model, lam=lam, beta=beta, mu=0.6, tau=0.28,
                                               dur_center=center, dur_sigma=sigma)
                    m = D.score(segs, ref)
                    cells.append((lam, beta, center, sigma, m))
    pass_f1 = [(c[4]["f1"], c[4]["exact"]) for c in cells if c[4]["f1"] >= 0.90]
    pass_both = [c for c in cells if c[4]["f1"] >= 0.90 and c[4]["exact"] >= 0.50]
    print(f"{len(cells)} configurations; F1>=0.90 in {len(pass_f1)}; both targets in {len(pass_both)}")
    print("F1 distribution:", sorted(round(c[4]["f1"], 3) for c in cells)[-12:])
    for lam, beta, center, sigma, m in sorted(cells, key=lambda c: -c[4]["f1"])[:12]:
        print(f"  lam{lam:.0f} b{beta:.0f} c{center:.0f} s{sigma:.2f}: seg {m['segments']:3d} "
              f"exact {m['exact']:.3f} fam {m['family']:.3f} P {m['precision']:.3f} "
              f"R {m['recall']:.3f} F1 {m['f1']:.3f}")
    worst = [c for c in cells if c[4]["f1"] < 0.85]
    print(f"configs below F1 0.85: {len(worst)}")


if __name__ == "__main__":
    main()
