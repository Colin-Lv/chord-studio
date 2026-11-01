"""Build the three input variants for the fixed-weights input-swap experiment.

A  original mix           (whole song, as the analysis hears it)
B  accompaniment + bass   (other + guitar + piano + bass, equal gain)
C  accompaniment only     (other + guitar + piano)

No per-track normalisation and no peak repair beyond a single shared scale factor, so the
only difference between conditions is which stems are present.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_AUDIO = Path(r"D:\ChordStudio\data\projects\aa5a41797bb343e590abcf4d8b27ae54\runs\11fa80454ad94da3a81efa19c1cab64c\source\audio.m4a")
STEMS = Path(r"D:\ChordStudio\data\stems\d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f\bs_roformer_sw\wav")
WORK = Path(r"D:\ChordStudio\research\separation\stem-input")
FFMPEG = r"C:\Users\33573\AppData\Roaming\Python\Python313\Scripts\ffmpeg.EXE"

CONDITIONS = {
    "A_original": None,
    "B_accomp_plus_bass": ["other", "guitar", "piano", "bass"],
    "C_accomp_only": ["other", "guitar", "piano"],
}


def decode_original(target: Path) -> Path:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(PROJECT_AUDIO),
                    "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", str(target)], check=True)
    return target


def sum_stems(names, target: Path) -> Path:
    total = None
    rate = None
    for name in names:
        audio, rate = sf.read(str(STEMS / f"{name}.wav"), always_2d=True, dtype="float32")
        total = audio if total is None else total + audio
    peak = float(np.abs(total).max())
    if peak > 0.999:
        total *= 0.999 / peak
    sf.write(str(target), total, rate, subtype="FLOAT")
    return target


def main():
    out = WORK / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, stems in CONDITIONS.items():
        target = out / f"{name}.wav"
        if target.is_file():
            print(f"{name}: exists")
            continue
        if stems is None:
            decode_original(target)
        else:
            sum_stems(stems, target)
        info = sf.info(str(target))
        print(f"{name}: {info.duration:.1f}s {info.samplerate} Hz {info.channels} ch")
    (WORK / "conditions.json").write_text(json.dumps(CONDITIONS, ensure_ascii=False, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
