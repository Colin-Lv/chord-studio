"""Measure the shipped backend decoder at the current and a stronger tone weight."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\ChordStudio")
BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND / "app" / "web"))
sys.path.insert(0, str(BACKEND / "app"))

GOLDEN = ROOT / "research/key-v3/chord_seg/reference_golden.lab"
INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
             "min9": (0, 3, 7, 10, 14), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
             "add9": (0, 2, 4, 7)}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}


def pcs(label):
    root, _, quality = str(label).partition(":")
    if not quality:
        return set()
    pc = FLATS.get(root, NAMES.index(root) if root in NAMES else None)
    if pc is None:
        return set()
    for name, ivs in INTERVALS.items():
        if quality == name or (name == "add9" and quality.startswith("add9")):
            return {(pc + iv) % 12 for iv in ivs}
    return set()


def load_lab(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b, label = line.split(maxsplit=2)
            rows.append((float(a), float(b), label.strip()))
    return rows


def frame_acc(pred, ref, step=0.02):
    end = min(pred[-1][1], ref[-1][1])
    grid = np.arange(ref[0][0], end, step)

    def at(table, t):
        for row in table:
            if row[0] <= t < row[1]:
                return row[2]
        return "N"

    exact = family = 0
    for t in grid:
        p, q = at(pred, t), at(ref, t)
        exact += p == q
        pr, _, pq = p.partition(":")
        qr, _, qq = q.partition(":")
        family += (pr == qr) and (pq[:3] == qq[:3])
    return exact / len(grid), family / len(grid)


def boundary_f1(pred, truth, tol=0.06):
    pb = [a for a, _, _ in pred[1:]]
    tb = [a for a, _, _ in truth[1:]]
    hp = sum(1 for p in pb if any(abs(p - t) <= tol for t in tb))
    ht = sum(1 for t in tb if any(abs(p - t) <= tol for p in pb))
    prec, rec = hp / max(1, len(pb)), ht / max(1, len(tb))
    return 2 * prec * rec / max(1e-9, prec + rec)


def main():
    import full_decode as FD
    run = sorted((ROOT / "data/projects/aa5a41797bb343e590abcf4d8b27ae54/runs").iterdir(),
                 key=lambda p: p.stat().st_mtime)[-1]
    pred = json.loads((run / "retained10/prediction.json").read_text(encoding="utf-8"))
    cache = Path(pred["cache_directory"])
    post = np.load(run / "retained10/posterior.npz")
    cells = json.loads((cache / "cells.json").read_text(encoding="utf-8"))
    deep = np.load(cache / "deep.npz")
    notes = np.load(cache / "notes.npz")
    beats = json.loads((run / "beats.json").read_text(encoding="utf-8"))
    ref = load_lab(GOLDEN)
    print(f"{'mu':>4} {'segments':>9} {'exact':>7} {'family':>7} {'F1':>6} {'toneRecall':>11}")
    for mu in (0.55, 0.6, 0.65, 0.7, 0.75):
        rows = FD.decode(
            posterior=post["chord"],
            starts=np.array([c["start"] for c in cells], dtype=float),
            ends=np.array([c["end"] for c in cells], dtype=float),
            chroma=deep["chroma"], act=notes["activations"], note_times=notes["times"],
            beats=np.array([float(b["time"]) for b in beats]),
            beat_pos=np.array([int(b.get("beat", 0)) for b in beats]),
            params=dict(mu=mu))
        ex, fam = frame_acc(rows, ref)
        f1 = boundary_f1(rows, ref)
        recall = 0.0
        n = 0
        for a, b, label in ref:
            mid = (a + b) / 2
            got = next((lab for x, y, lab in rows if x <= mid < y), "N")
            g, p = pcs(label), pcs(got)
            if g:
                n += 1
                recall += len(g & p) / len(g)
        print(f"{mu:4.2f} {len(rows):9d} {ex:7.3f} {fam:7.3f} {f1:6.3f} {recall / n:11.3f}")


if __name__ == "__main__":
    main()
