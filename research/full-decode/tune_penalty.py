"""Root changes are common, quality-only changes are rare: flip the switch penalties.

In this corpus the harmonic grid changes root at nearly every boundary, so a same-root
quality switch should cost more than a root switch, not less.
"""
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
        for base in (12.0, 16.0, 20.0, 24.0):
            for extra in (0.0, 10.0, 20.0, 30.0, 50.0):
                for mu in (0.5, 0.7):
                    segs = F.decode_viterbi(art, model, kappa=base + extra, kappa_root=base,
                                            beta=beta, relabel=True, gamma=2.0, mu=mu, tau=0.3)
                    m = D.score(segs, ref)
                    rows.append(dict(beta=beta, base=base, extra=extra, mu=mu, **m))
                    if m["f1"] >= 0.88 or m["exact"] >= 0.66:
                        print(f"b{beta} base{base:.0f} extra{extra:.0f} mu{mu}: "
                              f"seg {m['segments']:3d} exact {m['exact']:.3f} "
                              f"family {m['family']:.3f} P {m['precision']:.3f} "
                              f"R {m['recall']:.3f} F1 {m['f1']:.3f}", flush=True)
    print("\ntop 6 by F1:")
    for row in sorted(rows, key=lambda r: -r["f1"])[:6]:
        print(f"  b{row['beta']} base{row['base']:.0f} extra{row['extra']:.0f} mu{row['mu']}: "
              f"seg {row['segments']:3d} exact {row['exact']:.3f} family {row['family']:.3f} "
              f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f}")
    print("top 6 by exact:")
    for row in sorted(rows, key=lambda r: -r["exact"])[:6]:
        print(f"  b{row['beta']} base{row['base']:.0f} extra{row['extra']:.0f} mu{row['mu']}: "
              f"seg {row['segments']:3d} exact {row['exact']:.3f} family {row['family']:.3f} "
              f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f}")
    (HERE / "sweep_penalty.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
