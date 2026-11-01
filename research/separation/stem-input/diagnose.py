"""Diagnostics: (D) mix without vocals, and how much high-band content the bass stem carries."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

WORK = Path(r"D:\ChordStudio\research\separation\stem-input")
STEMS = Path(r"D:\ChordStudio\data\stems\d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f\bs_roformer_sw\wav")


def build_no_vocals():
    target = WORK / "inputs" / "D_no_vocals.wav"
    if target.is_file():
        print("D_no_vocals: exists")
        return
    total = None
    rate = None
    for name in ("drums", "bass", "other", "guitar", "piano"):
        audio, rate = sf.read(str(STEMS / f"{name}.wav"), always_2d=True, dtype="float32")
        total = audio if total is None else total + audio
    peak = float(np.abs(total).max())
    if peak > 0.999:
        total *= 0.999 / peak
    sf.write(str(target), total, rate, subtype="FLOAT")
    print(f"D_no_vocals: peak {peak:.2f}")


def band_share(path: Path, lo: float, hi: float) -> float:
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / rate)
    band = (freqs >= lo) & (freqs < hi)
    return float(spectrum[band].sum() / max(spectrum.sum(), 1e-12))


def main():
    build_no_vocals()
    print("\nband energy shares (of total power):")
    for name in ("bass", "other", "guitar", "piano"):
        path = STEMS / f"{name}.wav"
        low = band_share(path, 20, 300)
        mid = band_share(path, 300, 2000)
        high = band_share(path, 2000, 12000)
        print(f"  {name:7s} <300Hz {low * 100:5.1f}%  300-2k {mid * 100:5.1f}%  >2k {high * 100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
