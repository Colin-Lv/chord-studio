"""Split the CHORDS track into COMP + BASS so stems can be rendered without regenerating.

The generator writes bass notes inside the CHORDS track. This script moves the low notes
(pitch <= 47, i.e. below C3) to a new program-33 bass track, keeping every other track
(melody) untouched, then writes a new MIDI next to the original.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi

BASS_MAX_PITCH = 47


def split(path: Path, out: Path | None = None) -> Path:
    pm = pretty_midi.PrettyMIDI(str(path))
    out_pm = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        low = [n for n in inst.notes if n.pitch <= BASS_MAX_PITCH]
        high = [n for n in inst.notes if n.pitch > BASS_MAX_PITCH]
        if low and high:                       # this is the chord+bass track
            comp = pretty_midi.Instrument(program=inst.program, name="COMP")
            comp.notes = high
            bass = pretty_midi.Instrument(program=33, name="BASS")
            bass.notes = low
            out_pm.instruments.extend([comp, bass])
        else:
            inst.name = inst.name or "TRACK"
            out_pm.instruments.append(inst)
    target = out or path.with_suffix(".split.mid")
    out_pm.write(str(target))
    return target


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = split(src)
    pm = pretty_midi.PrettyMIDI(str(dst))
    print("wrote", dst)
    for inst in pm.instruments:
        print(f"  {inst.name:6s} program {inst.program:2d} notes {len(inst.notes)}")
