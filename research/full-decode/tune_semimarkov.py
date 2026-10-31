"""Tune the semi-Markov decoder (duration prior + boundary evidence)."""
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
    for lam in (0.0, 10.0, 20.0, 40.0):
        for beta in (0.0, 2.0, 4.0, 8.0):
            for center in (6.0, 8.0, 10.0):
                for sigma in (0.35, 0.5):
                    segs = F.decode_semimarkov(art, model, lam=lam, beta=beta, mu=0.6,
                                               tau=0.28, dur_center=center,
                                               dur_sigma=sigma, gamma=0.0)
                    m = D.score(segs, ref)
                    rows.append(dict(lam=lam, beta=beta, center=center, sigma=sigma, **m))
    print("top 8 by F1:")
    for row in sorted(rows, key=lambda r: -r["f1"])[:8]:
        print(f"  lam{row['lam']:.0f} beta{row['beta']:.0f} c{row['center']:.0f} "
              f"s{row['sigma']}: seg {row['segments']:3d} exact {row['exact']:.3f} "
              f"fam {row['family']:.3f} P {row['precision']:.3f} R {row['recall']:.3f} "
              f"F1 {row['f1']:.3f}")
    print("top 5 by exact:")
    for row in sorted(rows, key=lambda r: -r["exact"])[:5]:
        print(f"  lam{row['lam']:.0f} beta{row['beta']:.0f} c{row['center']:.0f} "
              f"s{row['sigma']}: seg {row['segments']:3d} exact {row['exact']:.3f} "
              f"fam {row['family']:.3f} P {row['precision']:.3f} R {row['recall']:.3f} "
              f"F1 {row['f1']:.3f}")
    (HERE / "sweep_semimarkov.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
