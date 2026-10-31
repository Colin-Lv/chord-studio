"""How well does NNLS note evidence give (a) the bass root and (b) the full chord-tone set?

Compares per golden segment: bass argmax from the low register (MIDI 21-47) vs the golden
root, and chord-tone support of the golden label vs the frozen model's argmax label.
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
INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
             "min9": (0, 3, 7, 10, 14), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}
MIDI = np.arange(21, 105)


def load_lab(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b, label = line.split(maxsplit=2)
            rows.append((float(a), float(b), label.strip()))
    return rows


def pcs_of(label):
    root, _, q = label.partition(":")
    if root not in NAMES:
        return None, None
    r = NAMES.index(root)
    if q not in INTERVALS:
        return None, None
    return r, {(r + iv) % 12 for iv in INTERVALS[q]}


def main():
    run = sorted((ROOT / "data/projects/aa5a41797bb343e590abcf4d8b27ae54/runs").iterdir(),
                 key=lambda p: p.stat().st_mtime)[-1]
    pred = json.loads((run / "retained10/prediction.json").read_text(encoding="utf-8"))
    cache = Path(pred["cache_directory"])
    notes = np.load(cache / "notes.npz")
    act, times = notes["activations"], notes["times"]
    norm = act / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)

    def profile(lo, hi, a, b):
        mask = (MIDI >= lo) & (MIDI < hi)
        pc = MIDI[mask] % 12
        out = np.zeros((12, len(times)), dtype=np.float32)
        np.add.at(out, pc, norm[mask])
        sel = (times >= a) & (times < b)
        if not sel.any():
            return np.zeros(12, dtype=np.float32)
        return out[:, sel].mean(axis=1)

    ref = load_lab(GOLDEN)
    hit = 0
    total = 0
    print(f"{'start':>7} {'ref':>9} {'bass_pc':>8} {'ok':>3}  {'tones':>6} {'non':>6}")
    for a, b, label in ref:
        r, pcs = pcs_of(label)
        if r is None:
            continue
        low = profile(21, 48, a, b)
        bass_pc = int(low.argmax())
        ok = bass_pc == r
        hit += ok
        total += 1
        allp = profile(21, 105, a, b)
        tone = np.mean([allp[p] for p in pcs])
        non = np.mean([allp[p] for p in range(12) if p not in pcs])
        print(f"{a:7.2f} {label:>9} {NAMES[bass_pc]:>8} {str(ok):>3}  {tone:6.3f} {non:6.3f}")
    print(f"\nbass-root accuracy on golden segments: {hit}/{total} = {hit / total:.3f}")


if __name__ == "__main__":
    main()
