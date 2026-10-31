"""Batch: split bass out of every MIDI and render BASS / COMP / MELODY stems."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pretty_midi

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
sys.path.insert(0, str(DATA))
import render as R  # noqa: E402
import split_bass as SB  # noqa: E402
import soundfile as sf  # noqa: E402

SR = 48000
OUT = DATA / "stems"


def track_name(inst):
    if inst.program == 33:
        return "BASS"
    if 24 <= inst.program <= 31:
        return "COMP"
    if inst.program in (0, 1, 2, 3, 4, 5, 6, 7):
        return "COMP" if inst.name.upper() in ("CHORDS", "COMP") else "MELODY"
    return "OTHER"


def main():
    OUT.mkdir(exist_ok=True)
    sfx = R.make_sampler_class(R.load_sf2_module())(str(R.find_soundfont(None)))
    midis = sorted((DATA / "midi").glob("*.mid"))
    midis = [m for m in midis if not m.name.endswith(".split.mid")]
    print(f"{len(midis)} midi files", flush=True)
    done = 0
    t0 = time.time()
    for midi in midis:
        target = OUT / midi.stem
        if target.exists() and any(target.glob("stem_*.wav")):
            continue
        split_path = midi.with_suffix(".split.mid")
        if not split_path.exists():
            SB.split(midi, split_path)
        pm = pretty_midi.PrettyMIDI(str(split_path))
        target.mkdir(parents=True, exist_ok=True)
        groups = {}
        for inst in pm.instruments:
            if not inst.notes:
                continue
            solo = pretty_midi.PrettyMIDI()
            solo.instruments.append(inst)
            audio = R.render_pm(solo, sfx, sr=SR)
            name = track_name(inst)
            if name in groups:
                n = min(len(groups[name]), len(audio))
                groups[name] = groups[name][:n] + audio[:n]
            else:
                groups[name] = audio
        for name, audio in groups.items():
            sf.write(target / f"stem_{name}.wav", audio.astype("float32"), SR)
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(midis)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {done}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
