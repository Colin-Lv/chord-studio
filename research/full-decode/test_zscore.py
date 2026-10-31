"""Standardise the boundary logits per song before the DP discount.

The GBDT score scale is song-dependent; a fixed `beta * logit` therefore moves the operating
point between songs. z-scoring the logits makes the discount scale-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import boundary_lib as B  # noqa: E402
import decode_lib as D  # noqa: E402
import full_decode as F  # noqa: E402


def main():
    art = F._prepare(D.load_run(D.latest_xuni_run()))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    sc = F.score_candidates(art, model)
    bars = sc["bars"]
    spans = []
    for a, b in zip(bars, bars[1:]):
        mid = (a + b) / 2.0
        spans.append((float(a), float(mid)))
        spans.append((float(mid), float(b)))
    emission = F.emission_matrix(art, spans, 1.5, 0.6, 0.28)
    score_at = {round(float(t), 3): float(l) for t, l in zip(sc["times"], sc["logit"])}
    raw = np.zeros(len(spans) - 1)
    for i in range(1, len(spans)):
        raw[i - 1] = score_at.get(round(float(spans[i][0]), 3), 0.0)
    z = (raw - raw.mean()) / (raw.std() + 1e-9)
    print(f"logit mean {raw.mean():.2f} std {raw.std():.2f} min {raw.min():.2f} max {raw.max():.2f}")
    best = None
    for scale in ("raw", "z"):
        edge = raw if scale == "raw" else z
        for beta in ((2.0, 3.0, 4.0) if scale == "raw" else (0.5, 1.0, 1.5, 2.0, 3.0)):
            for base in (12.0, 16.0, 20.0):
                for extra in (0.0, 10.0, 20.0):
                    path = F.viterbi_spans(art, spans, emission, edge, base + extra, base, beta)
                    rows = D.merge_rows([(a, b, f"{D.NAMES[s // 10]}:{D.QUALITIES[s % 10]}")
                                         for (a, b), s in zip(spans, path)])
                    out = []
                    for a, b, _old in rows:
                        label, _cells = F._span_label(art, a, b, 1.5, 0.6, 0.28)
                        if label is not None:
                            out.append((a, b, label))
                    segs = D.merge_rows(out)
                    m = D.score(segs, ref)
                    tag = f"{scale:4s} beta {beta:4.1f} base {base:4.0f} extra {extra:4.0f}"
                    if best is None or m["f1"] > best[0]:
                        best = (m["f1"], tag, m)
                        print(f"{tag}: seg {m['segments']:3d} exact {m['exact']:.3f} "
                              f"fam {m['family']:.3f} P {m['precision']:.3f} "
                              f"R {m['recall']:.3f} F1 {m['f1']:.3f}  <- best")
    print("\nbest:", best[1], {k: round(v, 3) for k, v in best[2].items()})


if __name__ == "__main__":
    main()
