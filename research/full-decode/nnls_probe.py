"""Quick check: can NNLS note evidence separate chord tones from melody on synthetic audio?"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app")
import harmonic_note_features as H  # noqa: E402

SONIC = r"D:\ChordStudio\Dialogue\tools\sonic-annotator-1.7-win64\sonic-annotator-win64\sonic-annotator.exe"
PLUGIN = r"C:\Program Files\Vamp Plugins\nnls-chroma.dll"
DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
CACHE = Path(r"D:\ChordStudio\research\full-decode\nnls_cache")
MIDI = np.arange(21, 105)


def main(sample_ids):
    for sid in sample_ids:
        t0 = time.time()
        ev = H.extract_note_evidence(DATA / "audio" / f"{sid}.wav", cache_dir=CACHE,
                                     sonic_annotator=SONIC, plugin_path=PLUGIN)
        dt = time.time() - t0
        labels = json.loads((DATA / "labels" / f"{sid}.events.json").read_text(encoding="utf-8"))
        print(f"== {sid}  {ev.activations.shape} in {dt:.1f}s")
        for chord in labels["chords"][:5]:
            sel = (ev.times >= chord["start"]) & (ev.times < chord["end"])
            prof = ev.activations[:, sel].mean(axis=1)
            tone = np.isin(MIDI % 12, chord["pcs"])
            top = np.argsort(prof)[::-1][:7]
            print(f"  root {chord['root_pc']:2d} {chord['quality']:5s} pcs {sorted(chord['pcs'])} "
                  f"tone {prof[tone].mean():.3f} non {prof[~tone].mean():.3f} "
                  f"top {[(int(m % 12), round(float(prof[i]), 2)) for i, m in [(i, MIDI[i]) for i in top]]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pop909-004-arpeggio-mel", "pop909-004-block"])
