"""Local sweep around F1 = 0.894 to close the last points."""
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
    rows = []
    for extra in (16.0, 20.0, 24.0):
        for tau in (0.25, 0.28, 0.30, 0.32, 0.35):
            for lam_bass in (1.2, 1.5, 1.8):
                for gamma in (1.0, 2.0, 3.0):
                    segs = F.decode_viterbi(art, model, kappa=16.0 + extra, kappa_root=16.0,
                                            beta=3.0, relabel=True, gamma=gamma, mu=0.5,
                                            tau=tau, lam_bass=lam_bass)
                    m = D.score(segs, ref)
                    rows.append(dict(extra=extra, tau=tau, lam_bass=lam_bass, gamma=gamma, **m))
    for row in sorted(rows, key=lambda r: -r["f1"])[:10]:
        print(f"  extra{row['extra']:.0f} tau{row['tau']} bass{row['lam_bass']} g{row['gamma']:.0f}: "
              f"seg {row['segments']:3d} exact {row['exact']:.3f} fam {row['family']:.3f} "
              f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f}")
    print()
    for row in sorted(rows, key=lambda r: -(0.5 * r["f1"] + 0.5 * r["exact"]))[:10]:
        print(f"  extra{row['extra']:.0f} tau{row['tau']} bass{row['lam_bass']} g{row['gamma']:.0f}: "
              f"seg {row['segments']:3d} exact {row['exact']:.3f} fam {row['family']:.3f} "
              f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f}")


if __name__ == "__main__":
    main()
