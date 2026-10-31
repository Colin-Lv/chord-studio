"""Final tuning pass for the full decoder (bar-split Viterbi + pooled relabel)."""
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
    for beta in (4.0, 5.0, 6.0, 8.0):
        for kappa in (16.0, 20.0, 24.0):
            for root_extra in (10.0, 30.0, 60.0):
                for gamma in (0.0, 2.0):
                    segs = F.decode_viterbi(art, model, kappa=kappa,
                                            kappa_root=kappa + root_extra, beta=beta,
                                            relabel=True, gamma=gamma)
                    m = D.score(segs, ref)
                    rows.append(dict(beta=beta, kappa=kappa, root_extra=root_extra,
                                     gamma=gamma, **m))
                    tag = f"b{beta} k{kappa:.0f} r+{root_extra:.0f} g{gamma:.0f}"
                    print(f"{tag:24s} seg {m['segments']:3d} exact {m['exact']:.3f} "
                          f"family {m['family']:.3f} P {m['precision']:.2f} "
                          f"R {m['recall']:.2f} F1 {m['f1']:.2f}", flush=True)
                    if best is None or (m["f1"], m["exact"]) > best[0]:
                        best = ((m["f1"], m["exact"]), dict(tag=tag, **m))
    print("\nbest:", best[1]["tag"], {k: round(v, 3) if isinstance(v, float) else v
                                      for k, v in best[1].items() if k != "tag"})
    # prune pass on top of the best configuration
    cfg = best[1]
    for delta in (0.0, 5.0, 10.0, 20.0):
        segs = F.decode_viterbi(art, model, kappa=cfg["kappa"],
                                kappa_root=cfg["kappa"] + cfg["root_extra"], beta=cfg["beta"],
                                relabel=True, gamma=cfg["gamma"], prune_delta=delta)
        m = D.score(segs, ref)
        print(f"  + prune {delta:5.1f}: seg {m['segments']:3d} exact {m['exact']:.3f} "
              f"family {m['family']:.3f} P {m['precision']:.2f} R {m['recall']:.2f} "
              f"F1 {m['f1']:.2f}")
    (HERE / "sweep_final.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
