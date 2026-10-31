"""Chord-tone recovery per source on 虚拟's golden: does the timeline keep the notes?

For every golden event, take the predicted label over the event's midpoint and compare pitch
class sets. Tone recall = how many of the reference's chord tones the prediction contains;
tone precision = how many of the prediction's tones the reference contains.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode_lib as D  # noqa: E402

INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
             "min9": (0, 3, 7, 10, 14), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
             "add9": (0, 2, 4, 7), "6": (0, 4, 7, 9)}
FLATS = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}


def pcs(label):
    root, _, quality = str(label).partition(":")
    if not quality:
        return set()
    if root in FLATS:
        pc = FLATS[root]
    elif root in D.NAMES:
        pc = D.NAMES.index(root)
    else:
        return set()
    if quality in INTERVALS:
        return {(pc + iv) % 12 for iv in INTERVALS[quality]}
    if quality.startswith("add9"):
        return {(pc + iv) % 12 for iv in INTERVALS["add9"]}
    return set()


def at(rows, t):
    for a, b, label in rows:
        if a <= t < b:
            return label
    return "N"


def main():
    run = D.latest_xuni_run()
    ref = D.load_lab(D.GOLDEN)
    sources = ["retained10", "retained10_stable", "retained10_span", "retained10_full"]
    print(f"run {run.name}  golden events {len(ref)}")
    print(f"{'source':>18} {'rows':>5} {'toneRecall':>11} {'toneF1':>8} {'extShare':>9} {'exact@mid':>10}")
    for name in sources:
        path = run / f"{name}.lab"
        if not path.is_file():
            continue
        rows = D.load_lab(path)
        recall = precision = 0.0
        exact = 0
        ext = 0
        n = 0
        for a, b, label in ref:
            mid = (a + b) / 2
            pred = at(rows, mid)
            g = pcs(label)
            p = pcs(pred)
            if not g:
                continue
            n += 1
            recall += len(g & p) / len(g)
            precision += (len(g & p) / len(p)) if p else 0.0
            exact += int(pred == label)
            ext += int(bool(p) and not (p == {(list(g)[0])}) and len(p) > 3)
        recall /= n
        precision /= n
        f1 = 2 * recall * precision / max(1e-9, recall + precision)
        ext_share = sum(1 for _, _, label in rows
                        if pcs(label) and len(pcs(label)) > 3) / len(rows)
        print(f"{name:>18} {len(rows):5d} {recall:11.3f} {f1:8.3f} {ext_share:9.3f} "
              f"{exact / n:10.3f}")
    ref_ext = sum(1 for _, _, label in ref if len(pcs(label)) > 3) / len(ref)
    print(f"{'golden':>18} {len(ref):5d} {'':11} {'':8} {ref_ext:9.3f} {1.0:10.3f}")


if __name__ == "__main__":
    main()
