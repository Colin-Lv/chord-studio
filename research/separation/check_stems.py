"""Sanity-check separated stems: level, activity and a rough spectral profile."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(r"D:\ChordStudio\research\separation\out_cpu\clip60")


def main():
    for path in sorted(OUT.glob("*.wav")):
        data, sr = sf.read(str(path), always_2d=True)
        mono = data.mean(axis=1)
        rms = float(np.sqrt((mono ** 2).mean()))
        peak = float(np.abs(mono).max())
        active = float((np.abs(mono) > 0.01).mean())
        print(f"{path.stem:8s} sr={sr} ch={data.shape[1]} rms={20 * np.log10(rms + 1e-9):6.1f} dBFS "
              f"peak={20 * np.log10(peak + 1e-9):6.1f} dBFS active={active:.2f}")


if __name__ == "__main__":
    main()
