"""Are all labelled chord tones actually sounding inside each chord span?

Reads the rendered MIDI and the label file. If a chord tone never sounds in its own span,
no audio model can recover that label from the span alone.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
EXT = Path(r"D:\ChordStudio\research\extension-dataset")


def check(base, entry, limit=None):
    import pretty_midi

    labels = json.loads((base / entry["labels"]).read_text(encoding="utf-8"))
    pm = pretty_midi.PrettyMIDI(str(base / entry["midi"]))
    notes = [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes]
    missing = []
    for chord in labels["chords"]:
        pcs = set(int(p) % 12 for p in chord["pcs"])
        heard = set()
        for start, end, pitch in notes:
            if start < chord["end"] - 1e-6 and end > chord["start"] + 1e-6:
                heard.add(pitch % 12)
        absent = pcs - heard
        if absent:
            missing.append((round(chord["start"], 2), chord["label"], sorted(absent),
                            len(pcs), round(chord["end"] - chord["start"], 2)))
    return missing, len(labels["chords"])


def main():
    for base in (DATA, EXT):
        meta = json.loads((base / "dataset.json").read_text(encoding="utf-8"))["samples"]
        by_texture = Counter()
        total = Counter()
        examples = {}
        for entry in meta:
            texture = entry["sample_id"].split("-")[2]
            missing, n = check(base, entry)
            total[texture] += n
            by_texture[texture] += len(missing)
            if missing and texture not in examples:
                examples[texture] = missing[:4]
        print(f"== {base.name}")
        for texture in total:
            print(f"  {texture:9s} {by_texture[texture]:5d}/{total[texture]:5d} chords have an absent tone")
            if texture in examples:
                for row in examples[texture]:
                    print("     e.g.", row)


if __name__ == "__main__":
    main()
