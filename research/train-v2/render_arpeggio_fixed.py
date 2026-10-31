"""Render the repaired arpeggio MIDI to audio_fixed/ (originals untouched)."""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
OUT = DATA / "audio_fixed"
SONIC = None


def process(job):
    import soundfile as sf

    sys.path.insert(0, str(DATA))
    import render as R

    if not hasattr(process, "_sfx"):
        process._sfx = R.make_sampler_class(R.load_sf2_module())(str(R.find_soundfont(None)))
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(job["midi"])
    audio = R.render_pm(pm, process._sfx, sr=48000)
    sf.write(job["out"], audio.astype("float32"), 48000, subtype="PCM_16")
    return job["sample_id"], round(len(audio) / 48000, 2)


def main(workers: int = 8):
    OUT.mkdir(exist_ok=True)
    report = json.loads((DATA / "arpeggio_fix.json").read_text(encoding="utf-8"))
    jobs = [dict(sample_id=r["sample_id"], midi=r["midi"],
                 out=str(OUT / f"{r['sample_id']}.wav")) for r in report]
    print(f"{len(jobs)} files, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (sid, dur) in enumerate(pool.imap_unordered(process, jobs, chunksize=2), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} rendered ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main(8)
