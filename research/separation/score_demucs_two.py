"""Score demucs' four-stem output as a two-stem system (vocals vs drums+bass+other)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SONG = Path(r"D:\ChordStudio\datasets\musdb_song0")
DEMUCS = SONG / "demucs" / "htdemucs" / "mixture"
EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")


def load(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), always_2d=True, dtype="float32")
    return data


def main():
    estimate = SONG / "est_two_demucs"
    estimate.mkdir(exist_ok=True)
    vocals = load(DEMUCS / "vocals.wav")
    bed = sum(load(DEMUCS / f"{name}.wav") for name in ("drums", "bass", "other"))
    length = min(len(vocals), len(bed))
    sf.write(str(estimate / "vocals.wav"), vocals[:length], 44100, subtype="FLOAT")
    sf.write(str(estimate / "instrumental.wav"), bed[:length], 44100, subtype="FLOAT")
    result = subprocess.run([sys.executable, str(EVAL), "--truth", str(SONG / "truth_two"),
                             "--stems", str(estimate)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr[-2000:])
        return 1
    data = json.loads(result.stdout)
    print("demucs as two-stem: mean SI-SDR %.2f dB, mean SDR %.2f dB"
          % (data["mean_si_sdr_db"], data["mean_sdr_db"]))
    for name, values in data["stems"].items():
        print("  %-13s SI-SDR %6.2f  SDR %6.2f" % (name, values["si_sdr_db"], values["sdr_db"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
