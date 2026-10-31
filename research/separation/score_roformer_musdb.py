"""Score the BS-RoFormer two-stem output against MUSDB ground truth.

Vocals are scored directly; the instrumental is compared with drums+bass+other summed from
the ground truth, which is the definition the model was trained against.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SONG = Path(r"D:\ChordStudio\datasets\musdb_song0")
ROFORMER = SONG / "roformer"
PY = sys.executable
EVAL = Path(r"D:\ChordStudio\research\separation\eval\stem_eval.py")


def load(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), always_2d=True, dtype="float32")
    return data


def main():
    truth = SONG / "truth_two"
    truth.mkdir(exist_ok=True)
    vocals = load(SONG / "vocals.wav")
    instrumental = sum(load(SONG / f"{name}.wav") for name in ("drums", "bass", "other"))
    rate = 44100
    sf.write(str(truth / "vocals.wav"), vocals, rate, subtype="FLOAT")
    sf.write(str(truth / "instrumental.wav"), instrumental, rate, subtype="FLOAT")

    estimate = SONG / "est_two"
    estimate.mkdir(exist_ok=True)
    for source, target in ((ROFORMER.glob("*Vocals*.wav"), "vocals.wav"),
                           (ROFORMER.glob("*Instrumental*.wav"), "instrumental.wav")):
        path = next(iter(source))
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        sf.write(str(estimate / target), data, sr, subtype="FLOAT")

    result = subprocess.run([PY, str(EVAL), "--truth", str(truth), "--stems", str(estimate)],
                            capture_output=True, text=True)
    if result.returncode:
        print(result.stderr[-2000:])
        return 1
    data = json.loads(result.stdout)
    print("mean SI-SDR %.2f dB   mean SDR %.2f dB" % (data["mean_si_sdr_db"], data["mean_sdr_db"]))
    for name, values in data["stems"].items():
        print("  %-13s SI-SDR %6.2f  SDR %6.2f" % (name, values["si_sdr_db"], values["sdr_db"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
