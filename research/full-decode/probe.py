"""Probe the frozen-model artifacts of one run: posterior, cells, NNLS notes, chroma, beats.

Prints the baseline metrics of several decoders against the golden reference so we know
where the full decoder starts from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\ChordStudio")
GOLDEN = ROOT / "research/key-v3/chord_seg/reference_golden.lab"
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
TOL = 0.06


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
            a, b = row[0], row[1]
            if a <= t < b:
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


def boundary_f1(pred, truth, tol=TOL):
    pb = [a for a, _, _ in pred[1:]]
    tb = [a for a, _, _ in truth[1:]]
    hp = sum(1 for p in pb if any(abs(p - t) <= tol for t in tb))
    ht = sum(1 for t in tb if any(abs(p - t) <= tol for p in pb))
    prec, rec = hp / max(1, len(pb)), ht / max(1, len(tb))
    return prec, rec, 2 * prec * rec / max(1e-9, prec + rec)


def merge(rows):
    out = []
    for a, b, label in rows:
        if out and out[-1][2] == label and abs(out[-1][1] - a) < 1e-6:
            out[-1] = (out[-1][0], b, label)
        else:
            out.append((float(a), float(b), label))
    return out


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if run is None:
        runs = sorted((ROOT / "data/projects/aa5a41797bb343e590abcf4d8b27ae54/runs").iterdir(),
                      key=lambda p: p.stat().st_mtime)
        run = runs[-1]
    pred = json.loads((run / "retained10/prediction.json").read_text(encoding="utf-8"))
    cache = Path(pred["cache_directory"])
    print("run", run.name)
    print("cache", cache)
    post = np.load(run / "retained10/posterior.npz")
    chord, N = post["chord"], post["N"]
    cells = json.loads((cache / "cells.json").read_text(encoding="utf-8"))
    starts = np.array([c["start"] for c in cells])
    ends = np.array([c["end"] for c in cells])
    print("cells", chord.shape, "N mean", float(N.mean()))
    deep = np.load(cache / "deep.npz")
    print("chroma", deep["chroma"].shape, "fps", float(deep["fps"]))
    notes = np.load(cache / "notes.npz")
    act, ntimes = notes["activations"], notes["times"]
    print("notes", act.shape, "frames", len(ntimes), "fps", 1 / np.median(np.diff(ntimes)))
    beats = json.loads((run / "beats.json").read_text(encoding="utf-8"))
    bt = np.array([float(b["time"]) for b in beats])
    print("beats", len(bt), "first", bt[:4], "bar marks", [int(b.get("beat", 0)) for b in beats[:8]])

    ref = load_lab(GOLDEN)
    print("ref segments", len(ref))

    # A: plain 120-way argmax
    labels = [f"{NAMES[i // 10]}:{QUALITIES[i % 10]}" for i in chord.argmax(axis=1)]
    a = merge(list(zip(starts, ends, labels)))
    print("A argmax       :", len(a), *[round(x, 3) for x in frame_acc(a, ref)], *[round(x, 2) for x in boundary_f1(a, ref)])

    # B: shipped span output
    span = load_lab(run / "retained10_span.lab")
    print("B span_pool    :", len(span), *[round(x, 3) for x in frame_acc(span, ref)], *[round(x, 2) for x in boundary_f1(span, ref)])

    # C: heads + bass from NNLS low register, per cell
    bass = bass_profile(act, ntimes, starts, ends)
    print("bass profile", bass.shape, "top pc per cell", np.bincount(bass.argmax(axis=1), minlength=12))
    labels = head_labels(chord, bass)
    c = merge(list(zip(starts, ends, labels)))
    print("C heads+bass   :", len(c), *[round(x, 3) for x in frame_acc(c, ref)], *[round(x, 2) for x in boundary_f1(c, ref)])


def bass_profile(act, times, starts, ends, lo=21, hi=48):
    midi = np.arange(21, 105)
    sel = (midi >= lo) & (midi < hi)
    low = act[sel].sum(axis=0)                     # T
    pc = midi[sel] % 12
    prof = np.zeros((12, len(times)), dtype=np.float32)
    np.add.at(prof, pc, low)
    prof /= (prof.sum(axis=0, keepdims=True) + 1e-9)
    out = np.zeros((len(starts), 12), dtype=np.float32)
    for i, (x, y) in enumerate(zip(starts, ends)):
        s = (times >= x) & (times < y)
        if s.any():
            out[i] = prof[:, s].mean(axis=1)
    out /= (out.sum(axis=1, keepdims=True) + 1e-9)
    return out


def head_labels(prob, bass, lam=1.5):
    p = np.maximum(prob, 1e-9)
    roots = p.reshape(len(p), 12, 10).sum(axis=2)
    score = np.log(roots) + lam * np.log(bass + 1e-3)
    root = score.argmax(axis=1)
    quality = np.array([p[i].reshape(12, 10)[root[i]].argmax() for i in range(len(p))])
    return [f"{NAMES[r]}:{QUALITIES[q]}" for r, q in zip(root, quality)]


if __name__ == "__main__":
    main()
