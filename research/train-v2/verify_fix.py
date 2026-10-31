"""Verify that every labelled chord tone sounds inside its span, for a given MIDI folder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pretty_midi

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")


def main(folder="midi_fixed", texture_filter="arpeggio"):
    meta = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))["samples"]
    total = missing = 0
    for entry in meta:
        if texture_filter not in entry["sample_id"]:
            continue
        path = DATA / folder / f"{entry['sample_id']}.mid"
        if not path.is_file():
            continue
        labels = json.loads((DATA / entry["labels"]).read_text(encoding="utf-8"))
        pm = pretty_midi.PrettyMIDI(str(path))
        notes = [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes]
        for chord in labels["chords"]:
            pcs = {int(p) % 12 for p in chord["pcs"]}
            heard = {p % 12 for s, e, p in notes
                     if s < chord["end"] - 1e-6 and e > chord["start"] + 1e-6}
            total += 1
            missing += int(bool(pcs - heard))
    print(f"{folder}: {missing}/{total} chords still have an absent tone")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "midi_fixed",
         sys.argv[2] if len(sys.argv) > 2 else "arpeggio")
