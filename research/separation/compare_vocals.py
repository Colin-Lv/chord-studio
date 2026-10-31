"""Compare the vocal stem from demucs and BS-RoFormer on the same 20 s excerpt."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(r"D:\ChordStudio\research\separation")
ROFORMER = BASE / "bsroformer"
DEMUCS = BASE / "out_gpu" / "htdemucs" / "clip20"


def load(path: Path) -> np.ndarray:
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data.mean(axis=1), rate


def rms(x):
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()))


def envelope(x, rate, hop=0.05):
    step = int(rate * hop)
    n = len(x) // step
    return np.sqrt((x[:n * step].reshape(n, step).astype(np.float64) ** 2).mean(axis=1))


def main():
    candidates = list(ROFORMER.glob("*Vocals*.wav"))
    if not candidates:
        print("no roformer vocals yet")
        return
    rof, rate = load(candidates[0])
    dem, rate2 = load(DEMUCS / "vocals.wav")
    n = min(len(rof), len(dem))
    rof, dem = rof[:n], dem[:n]
    print(f"file: {candidates[0].name}")
    print(f"length {n / rate:.1f}s  demucs vocals {20 * np.log10(rms(dem) + 1e-12):.1f} dBFS  "
          f"roformer vocals {20 * np.log10(rms(rof) + 1e-12):.1f} dBFS")
    # best simple gain fit, then correlation
    gain = float((rof @ dem) / (dem @ dem + 1e-12))
    residual = rof - gain * dem
    print(f"gain roformer/demucs {20 * np.log10(abs(gain) + 1e-12):.1f} dB")
    print(f"waveform corr {np.corrcoef(rof, dem)[0, 1]:.3f}")
    print(f"residual after gain fit {10 * np.log10((rms(residual) ** 2) / (rms(rof) ** 2 + 1e-12)):.1f} dB rel roformer")
    e1, e2 = envelope(rof, rate), envelope(dem, rate2)
    m = min(len(e1), len(e2))
    print(f"envelope corr {np.corrcoef(e1[:m], e2[:m])[0, 1]:.3f}")


if __name__ == "__main__":
    main()
