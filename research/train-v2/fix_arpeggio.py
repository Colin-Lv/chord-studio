"""Repair the arpeggio renderer's missing chord tones (data-fidelity fix).

The boundary dataset's arpeggio branch plays one upper chord tone per beat:
``tone = upper[(beat - start) % len(upper)]``. For a 2-beat chord only the first two upper
tones ever sound, so 4061 of 9108 arpeggio chords have a labelled tone that is *absent* from
the audio. No audio model can recover those labels.

This script adds the missing chord tones to the arpeggio MIDI in the same arpeggio style
(16th-note steps), writes them under ``midi_fixed/`` and leaves the originals untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
OUT = DATA / "midi_fixed"
FIXED = DATA / "arpeggio_fix.json"


def choose_pitch(pc, lo=55, hi=79, target=64):
    cands = [p for p in range(lo, hi + 1) if p % 12 == pc]
    if not cands:
        return pc + 60
    return min(cands, key=lambda p: abs(p - target))


def main():
    import pretty_midi

    OUT.mkdir(exist_ok=True)
    meta = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))["samples"]
    report = []
    for entry in meta:
        sample_id = entry["sample_id"]
        if "-arpeggio" not in sample_id:
            continue
        labels = json.loads((DATA / entry["labels"]).read_text(encoding="utf-8"))
        pm = pretty_midi.PrettyMIDI(str(DATA / entry["midi"]))
        chords = labels["chords"]
        added = 0
        for chord in chords:
            pcs = sorted({int(p) % 12 for p in chord["pcs"]})
            heard = set()
            for inst in pm.instruments:
                for note in inst.notes:
                    if note.start < chord["end"] - 1e-6 and note.end > chord["start"] + 1e-6:
                        heard.add(note.pitch % 12)
            absent = [pc for pc in pcs if pc not in heard]
            if not absent:
                continue
            target_inst = pm.instruments[0]
            span = float(chord["end"]) - float(chord["start"])
            slots = max(len(absent), 1)
            for k, pc in enumerate(absent):
                offset = span * (k + 1) / (slots + 1)
                start = float(chord["start"]) + offset
                dur = min(0.22, max(0.08, span / (slots + 1) * 0.8))
                target_inst.notes.append(pretty_midi.Note(
                    velocity=76, pitch=choose_pitch(pc), start=start, end=start + dur))
                added += 1
        out = OUT / f"{sample_id}.mid"
        pm.write(str(out))
        report.append(dict(sample_id=sample_id, midi=str(out), added=added))
    FIXED.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"rewrote {len(report)} arpeggio MIDI files, {sum(r['added'] for r in report)} notes added")


if __name__ == "__main__":
    main()

