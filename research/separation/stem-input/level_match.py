"""Level-matched variants: scale B and C to the RMS of A so the comparison is not a gain test."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path

WORK = Path(r"D:\ChordStudio\research\separation\stem-input\inputs")


def rms(path: Path) -> float:
    data, _ = sf.read(str(path), always_2d=True, dtype="float32")
    return float(np.sqrt((data.astype(np.float64) ** 2).mean()))


def main():
    reference = rms(WORK / "A_original.wav")
    for name in ("B_accomp_plus_bass", "C_accomp_only"):
        source = WORK / f"{name}.wav"
        target = WORK / f"{name}_rms.wav"
        if target.is_file():
            print(f"{target.name}: exists")
            continue
        data, rate = sf.read(str(source), always_2d=True, dtype="float32")
        current = float(np.sqrt((data.astype(np.float64) ** 2).mean()))
        gain = reference / max(current, 1e-9)
        data *= gain
        peak = float(np.abs(data).max())
        if peak > 0.999:
            data *= 0.999 / peak
        sf.write(str(target), data, rate, subtype="FLOAT")
        print(f"{target.name}: gain {20 * np.log10(gain):+.2f} dB, peak {peak:.2f}")


if __name__ == "__main__":
    main()
