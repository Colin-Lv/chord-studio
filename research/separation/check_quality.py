"""Objective self-checks on the separated stems: reconstruction and cross-stem envelope correlation."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

STEM_DIR = Path(r"D:\ChordStudio\data\stems\d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f\wav")
MIX = Path(r"D:\ChordStudio\data\projects\aa5a41797bb343e590abcf4d8b27ae54\runs\11fa80454ad94da3a81efa19c1cab64c\source\audio.m4a")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        data = np.frombuffer(handle.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0
        data = data.reshape(-1, handle.getnchannels())
    return data, rate


def read_mix(path: Path) -> tuple[np.ndarray, int]:
    import subprocess
    import tempfile
    out = Path(tempfile.mkdtemp()) / "mix.wav"
    ffmpeg = r"C:\Users\33573\AppData\Roaming\Python\Python313\Scripts\ffmpeg.EXE"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                    "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(out)], check=True)
    return read_wav(out)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()))


def main():
    mix, rate = read_mix(MIX)
    stems = {name: read_wav(STEM_DIR / f"{name}.wav")[0] for name in ("vocals", "drums", "bass", "other")}
    length = min([len(mix)] + [len(v) for v in stems.values()])
    mix = mix[:length]
    stems = {k: v[:length] for k, v in stems.items()}
    total = sum(stems.values())
    residual = mix - total
    print(f"mix rms        {20 * np.log10(rms(mix) + 1e-12):6.1f} dBFS")
    print(f"sum of stems   {20 * np.log10(rms(total) + 1e-12):6.1f} dBFS")
    print(f"residual       {20 * np.log10(rms(residual) + 1e-12):6.1f} dBFS "
          f"(residual/mix {20 * np.log10((rms(residual) + 1e-12) / (rms(mix) + 1e-12)):6.1f} dB)")

    # cross-stem envelope correlation (50 ms RMS envelope)
    hop = int(rate * 0.05)
    env = {k: np.sqrt((v[:len(v) // hop * hop].reshape(-1, hop, v.shape[1]).astype(np.float64) ** 2).mean(axis=(1, 2)))
           for k, v in stems.items()}
    names = list(env)
    print("\nenvelope correlation")
    print("        " + "".join(f"{n:>9s}" for n in names))
    for a in names:
        row = [np.corrcoef(env[a], env[b])[0, 1] for b in names]
        print(f"{a:>8s}" + "".join(f"{v:9.2f}" for v in row))


if __name__ == "__main__":
    main()
