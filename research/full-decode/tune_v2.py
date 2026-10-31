"""Re-tune the decoder against the improved boundary classifier (18 NNLS features)."""
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
    for beta in (2.0, 3.0, 4.0):
        for base in (12.0, 16.0, 20.0):
            for extra in (10.0, 20.0, 30.0):
                for mu in (0.5, 0.7):
                    for tau in (0.28, 0.30):
                        segs = F.decode_viterbi(art, model, kappa=base + extra, kappa_root=base,
                                                beta=beta, relabel=True, gamma=2.0, mu=mu,
                                                tau=tau)
                        m = D.score(segs, ref)
                        rows.append(dict(beta=beta, base=base, extra=extra, mu=mu, tau=tau, **m))
    print("top 8 by F1:")
    for row in sorted(rows, key=lambda r: -r["f1"])[:8]:
        print(f"  b{row['beta']} base{row['base']:.0f} extra{row['extra']:.0f} "
              f"mu{row['mu']} tau{row['tau']}: seg {row['segments']:3d} exact {row['exact']:.3f} "
              f"fam {row['family']:.3f} P {row['precision']:.3f} R {row['recall']:.3f} "
              f"F1 {row['f1']:.3f}")
    print("top 8 by exact:")
    for row in sorted(rows, key=lambda r: -r["exact"])[:8]:
        print(f"  b{row['beta']} base{row['base']:.0f} extra{row['extra']:.0f} "
              f"mu{row['mu']} tau{row['tau']}: seg {row['segments']:3d} exact {row['exact']:.3f} "
              f"fam {row['family']:.3f} P {row['precision']:.3f} R {row['recall']:.3f} "
              f"F1 {row['f1']:.3f}")
    print("top 8 by 0.5*(F1+exact):")
    for row in sorted(rows, key=lambda r: -(r["f1"] + r["exact"]))[:8]:
        print(f"  b{row['beta']} base{row['base']:.0f} extra{row['extra']:.0f} "
              f"mu{row['mu']} tau{row['tau']}: seg {row['segments']:3d} exact {row['exact']:.3f} "
              f"fam {row['family']:.3f} P {row['precision']:.3f} R {row['recall']:.3f} "
              f"F1 {row['f1']:.3f}")
    (HERE / "sweep_v2.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
