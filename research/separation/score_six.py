"""Score a six-stem model (vocals/drums/bass/other/guitar/piano) against MUSDB's four stems.

Guitar and piano are folded into "other" for the comparison, since MUSDB labels them that way.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")


def load(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), always_2d=True, dtype="float32")
    return data


def main():
    song = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    tag = sys.argv[3]
    truth = song / "truth_two"          # vocals + instrumental
    est = song / f"est_{tag}"
    est.mkdir(exist_ok=True)

    stems = {}
    for path in outdir.glob("*.wav"):
        name = path.name.lower()
        for key in ("vocals", "drums", "bass", "other", "guitar", "piano"):
            if f"({key})" in name:
                stems[key] = path
    print("found stems:", sorted(stems))
    if "vocals" not in stems:
        raise SystemExit("no vocals stem")

    other = sum(load(stems[key]) for key in ("other", "guitar", "piano") if key in stems)
    length = min(len(load(stems["vocals"])), len(other))
    sf.write(str(est / "vocals.wav"), load(stems["vocals"])[:length], 44100, subtype="FLOAT")
    sf.write(str(est / "instrumental.wav"), other[:length], 44100, subtype="FLOAT")

    result = subprocess.run([sys.executable, str(EVAL), "--truth", str(truth), "--stems", str(est)],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-1500:])
    data = json.loads(result.stdout)
    print(f"{tag}: vocals {data['stems']['vocals']['si_sdr_db']:.2f}  "
          f"instr(other+gtr+pno) {data['stems']['instrumental']['si_sdr_db']:.2f}  "
          f"mean {data['mean_si_sdr_db']:.2f}")
    for key in ("drums", "bass"):
        if key in stems:
            print(f"  {key}: {20 * np.log10(np.sqrt((load(stems[key]).astype(np.float64) ** 2).mean()) + 1e-12):.1f} dBFS")


if __name__ == "__main__":
    main()
