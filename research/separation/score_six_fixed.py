"""Score a six-stem model correctly: vocals vs truth vocals, everything else vs the backing.

MUSDB labels guitar/piano inside 'other', so the accompaniment estimate is
drums + bass + other + guitar + piano.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")


def load(path: str):
    data, rate = sf.read(path, always_2d=False, dtype="float32")
    return data, rate


def main():
    song = Path(sys.argv[1])
    srcdir = Path(sys.argv[2])
    tag = sys.argv[3]
    stems = {}
    for path in srcdir.glob("*.wav"):
        name = path.name.lower()
        for key in ("vocals", "drums", "bass", "other", "guitar", "piano"):
            if f"({key})" in name:
                stems[key] = str(path)
    missing = [key for key in ("vocals", "drums", "bass", "other") if key not in stems]
    if missing:
        raise SystemExit(f"missing {missing}")

    est = song / f"est_{tag}"
    est.mkdir(exist_ok=True)
    vocals, rate = sf.read(stems["vocals"], always_2d=True, dtype="float32")
    backing = None
    for key in ("drums", "bass", "other", "guitar", "piano"):
        if key not in stems:
            continue
        data, _ = sf.read(stems[key], always_2d=True, dtype="float32")
        backing = data if backing is None else backing + data
    length = min(len(vocals), len(backing))
    sf.write(str(est / "vocals.wav"), vocals[:length], rate, subtype="FLOAT")
    sf.write(str(est / "instrumental.wav"), backing[:length], rate, subtype="FLOAT")

    out = subprocess.run([sys.executable, str(EVAL), "--truth", str(song / "truth_two"),
                          "--stems", str(est)], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr[-1200:])
    data = json.loads(out.stdout)
    print(f"{tag}: vocals {data['stems']['vocals']['si_sdr_db']:.2f}  "
          f"instr {data['stems']['instrumental']['si_sdr_db']:.2f}  "
          f"mean {data['mean_si_sdr_db']:.2f}")


if __name__ == "__main__":
    main()
