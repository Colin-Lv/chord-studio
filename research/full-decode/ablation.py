"""Reproduce the ablation table of REVIEW.md section 3.4 on the 虚拟 golden reference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the research module *before* decode_lib: decode_lib puts the app's web directory on
# sys.path, which also contains a full_decode.py (the shipped decoder). Order matters here.
import full_decode as F  # noqa: E402
import decode_lib as D  # noqa: E402
assert hasattr(F, "_prepare"), "wrong full_decode module imported"


def main():
    run = D.latest_xuni_run()
    art = F._prepare(D.load_run(run))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    rows = []

    def add(name, segs):
        m = D.score(segs, ref)
        rows.append(dict(stage=name, **m))
        print(f"{name:44s} seg {m['segments']:3d} exact {m['exact']:.3f} "
              f"family {m['family']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"F1 {m['f1']:.3f}")

    add("cell argmax (120)", D.merge_rows([
        (a, b, f"{D.NAMES[i // 10]}:{D.QUALITIES[i % 10]}")
        for (a, b), i in zip(zip(art["starts"], art["ends"]), art["prob"].argmax(axis=1))]))
    add("published retained10_stable", D.load_lab(run / "retained10_stable.lab"))
    add("published retained10_span", D.load_lab(run / "retained10_span.lab"))

    # bar pooling with head+tone labels (no half-bar split)
    sc = F.score_candidates(art, model)
    bars = sc["bars"]
    segs = F._assign(art, list(zip(bars, bars[1:])), 1.5, 0.6, 0.28)
    add("head+tone, fixed bar spans", segs)

    # barsplit + gain pruning
    segs = F.decode(art, model, "barsplit", thr_half=0.5, delta=10.0, mu=0.6, tau=0.28)
    add("+ half-bar split + gain prune", segs)

    # CRF with flipped switch penalties
    segs = F.decode_viterbi(art, model, kappa=16.0 + 20.0, kappa_root=16.0, beta=3.0,
                            relabel=True, gamma=2.0, mu=0.5, tau=0.28)
    add("+ half-bar CRF", segs)

    # semi-Markov with the duration prior (shipped configuration)
    segs = F.decode_semimarkov(art, model)
    add("+ semi-Markov duration prior (shipped)", segs)

    (HERE / "ablation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

