"""Split-half check: choose the fusion weight on one half of the song, score the other.

One song is one song, so the only honest thing we can do is show that the weight is not
being fitted on the same segments that are reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import prototype_fusion as P

WORK = Path(r"D:\ChordStudio\research\separation\stem-chord")


def half_score(labels, times, events, lo, hi):
    subset = [(s, e, lab) for s, e, lab in events if s >= lo and e <= hi]
    return P.segment_accuracy(labels, times, subset)


def main():
    folder = P.run_frontend(P.build_harmonic())
    p_h, n_h = P.load_posterior(folder)
    p_mix, n_mix = P.load_posterior(P.MIX_RUN)
    count = min(len(p_mix), len(p_h))
    p_mix, p_h, n_mix, n_h = p_mix[:count], p_h[:count], n_mix[:count], n_h[:count]
    events = P.read_golden()
    times = np.array([(i + 0.5) * P.DURATION / count for i in range(count)])
    chroma = P.bass_chroma(count)

    split = 116.0
    halves = {"A": (events[0][0], split), "B": (split, events[-1][1])}
    grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    table = {}
    for lam in grid:
        fused = P.fuse(p_mix, p_h, lam, 0.0, chroma)
        labels = P.decode_labels(fused, np.maximum(n_mix, n_h))
        row = {}
        for name, (lo, hi) in halves.items():
            exact, root, weighted = half_score(labels, times, events, lo, hi)
            row[name] = exact
        table[lam] = row
        print(f"lambda={lam:.1f}  A {row['A'] * 100:5.2f}%  B {row['B'] * 100:5.2f}%  "
              f"mean {(row['A'] + row['B']) / 2 * 100:5.2f}%")

    best_a = max(grid, key=lambda x: table[x]["A"])
    best_b = max(grid, key=lambda x: table[x]["B"])
    print(f"\nbest on A: lambda={best_a}  -> B {table[best_a]['B'] * 100:.2f}%")
    print(f"best on B: lambda={best_b}  -> A {table[best_b]['A'] * 100:.2f}%")
    print(f"mix only:  A {table[0.0]['A'] * 100:.2f}%  B {table[0.0]['B'] * 100:.2f}%")
    (WORK / "split-half.json").write_text(json.dumps(
        {"grid": {str(k): v for k, v in table.items()}, "best_a": best_a, "best_b": best_b},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
