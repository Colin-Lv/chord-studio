"""Can sustained NNLS note support re-score the frozen model's quality head?

For each golden segment we pool the 120-class posterior, then ask whether the true label
wins when the score adds sustained note support of the chord tones. This is the part the
frozen CNN systematically under-predicts (maj7 -> maj, min9 -> min7, maj9 -> maj).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode_lib as D  # noqa: E402

INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
             "min9": (0, 3, 7, 10, 14), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}
MIDI = np.arange(21, 105)
REG = {"low": (21, 48), "mid": (48, 72), "high": (72, 105)}


def profiles(act, times, a, b):
    norm = act / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    sel = (times >= a) & (times < b)
    if not sel.any():
        sel = np.zeros(len(times), dtype=bool)
        sel[int(np.clip(np.searchsorted(times, (a + b) / 2), 0, len(times) - 1))] = True
    out = {}
    for name, (lo, hi) in REG.items():
        mask = (MIDI >= lo) & (MIDI < hi)
        pc = MIDI[mask] % 12
        prof = np.zeros((12, int(sel.sum())), dtype=np.float32)
        np.add.at(prof, pc, norm[mask][:, sel])
        out[name] = prof
    return out


def pc_scores(prof, threshold=0.02):
    """Mean profile and sustained-support fraction per pc."""
    mean = prof.mean(axis=1)
    return mean, (prof >= threshold).mean(axis=1)


def label_pcs(label):
    root, _, q = label.partition(":")
    # tolerate enharmonics used by the golden file
    flats = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}
    if root in flats:
        r = flats[root]
    elif root in D.NAMES:
        r = D.NAMES.index(root)
    else:
        return None, None
    if q in INTERVALS:
        return r, {((r + iv) % 12) for iv in INTERVALS[q]}
    if q.startswith("add9"):
        return r, {(r + 0) % 12, (r + 4) % 12, (r + 7) % 12, (r + 2) % 12}
    return None, None


def main():
    run = D.latest_xuni_run()
    art = D.load_run(run)
    ref = D.load_lab(D.GOLDEN)
    rows = []
    for a, b, label in ref:
        r, pcs = label_pcs(label)
        if pcs is None:
            continue
        prof = profiles(art["act"], art["note_times"], a, b)
        cells = [i for i, (x, y) in enumerate(zip(art["starts"], art["ends"])) if x < b and y > a]
        p = np.maximum(art["prob"][cells], 1e-9).mean(axis=0).reshape(12, 10)
        roots = p.sum(axis=1)
        bass = D.span_bass(art["act"], art["note_times"], a, b)
        mid_mean, mid_sup = pc_scores(prof["mid"])
        high_mean, high_sup = pc_scores(prof["high"])
        upper = mid_mean + high_mean
        sup = mid_sup + high_sup
        tone = np.mean([sup[i] for i in pcs])
        non = np.mean([sup[i] for i in range(12) if i not in pcs])
        rows.append(dict(a=a, b=b, label=label, root=r, pcs=pcs, p=p, roots=roots, bass=bass,
                         upper=upper, sup=sup, tone=tone, non=non))
    print(f"{len(rows)} segments")
    print(f"{'start':>7} {'ref':>9} {'model':>9} {'tone':>5} {'non':>5} {'gap':>5}")
    for r in rows:
        top = int(np.argmax(r["p"]))
        model = f"{D.NAMES[top // 10]}:{D.QUALITIES[top % 10]}"
        print(f"{r['a']:7.2f} {r['label']:>9} {model:>9} {r['tone']:5.2f} {r['non']:5.2f} "
              f"{r['tone'] - r['non']:+5.2f}")

    np.savez(HERE / "golden_pooled.npz",
             **{k: np.array([r[k] for r in rows], dtype=object) for k in ("label", "root", "pcs")},
             p=np.stack([r["p"] for r in rows]), roots=np.stack([r["roots"] for r in rows]),
             bass=np.stack([r["bass"] for r in rows]),
             upper=np.stack([r["upper"] for r in rows]), sup=np.stack([r["sup"] for r in rows]),
             starts=np.array([r["a"] for r in rows]))


if __name__ == "__main__":
    main()
