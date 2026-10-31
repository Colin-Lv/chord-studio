"""Fine sweep around the best decoder configuration, including label-term weights."""
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
    for beta in (3.0, 4.0, 5.0):
        for kappa in (20.0, 24.0, 28.0):
            for mu in (0.5, 0.7, 0.9):
                for tau in (0.25, 0.30, 0.35):
                    segs = F.decode_viterbi(art, model, kappa=kappa, kappa_root=kappa + 30.0,
                                            beta=beta, relabel=True, gamma=2.0, mu=mu, tau=tau)
                    m = D.score(segs, ref)
                    rows.append(dict(beta=beta, kappa=kappa, mu=mu, tau=tau, **m))
                    if best is None or (m["f1"], m["exact"]) > best[0]:
                        best = ((m["f1"], m["exact"]),
                                dict(beta=beta, kappa=kappa, mu=mu, tau=tau, **m))
    print("top 8 by F1:")
    for row in sorted(rows, key=lambda r: -r["f1"])[:8]:
        print(f"  b{row['beta']} k{row['kappa']:.0f} mu{row['mu']} tau{row['tau']}: "
              f"seg {row['segments']:3d} exact {row['exact']:.3f} family {row['family']:.3f} "
              f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f}")
    print("top 5 by exact:")
    for row in sorted(rows, key=lambda r: -r["exact"])[:5]:
        print(f"  b{row['beta']} k{row['kappa']:.0f} mu{row['mu']} tau{row['tau']}: "
              f"seg {row['segments']:3d} exact {row['exact']:.3f} family {row['family']:.3f} "
              f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f}")
    b = best[1]
    print("\nbest by (F1, exact):", {k: round(v, 3) for k, v in b.items()})
    for delta in (0.0, 5.0, 10.0, 20.0):
        segs = F.decode_viterbi(art, model, kappa=b["kappa"], kappa_root=b["kappa"] + 30.0,
                                beta=b["beta"], relabel=True, gamma=2.0, mu=b["mu"],
                                tau=b["tau"], prune_delta=delta)
        m = D.score(segs, ref)
        print(f"  + prune {delta:5.1f}: seg {m['segments']:3d} exact {m['exact']:.3f} "
              f"family {m['family']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"F1 {m['f1']:.3f}")
    (HERE / "sweep_fine.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
