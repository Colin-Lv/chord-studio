"""Build the four listening conditions for the chord-downstream experiment.

All four are written as 48 kHz mono float32 WAVs with the SAME length as mix.wav,
so the only thing that varies is the content.  Conditions 2-4 are RMS-matched to
the mix (one scalar gain each); the gain is reported in dB.

  1 mix              synth6/mix.wav                       (byte-identical copy)
  2 truth_harm       truth/piano.wav + truth/bass.wav
  3 sep_harm         model6 piano + bass stems
  4 sep_harm4        model6 piano + bass + guitar + other stems
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from cd_common import SYNTH, WORK

SR = 48000
MIX = SYNTH / "mix.wav"
TRUTH = SYNTH / "truth"
MODEL6 = SYNTH / "model6"


def load_48k_mono(path: Path, n: int) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != SR:
        g = np.gcd(int(sr), SR)
        mono = resample_poly(mono, SR // g, sr // g).astype(np.float32)
    if mono.size < n:
        mono = np.pad(mono, (0, n - mono.size))
    return np.ascontiguousarray(mono[:n], dtype=np.float32)


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
                             1e-12))


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    mix_audio, sr = sf.read(str(MIX), always_2d=True, dtype="float32")
    assert sr == SR and mix_audio.shape[1] == 1, (sr, mix_audio.shape)
    n = mix_audio.shape[0]
    target_rms = float(np.sqrt(np.mean(mix_audio[:, 0].astype(np.float64) ** 2)))
    print(f"mix: {n} frames = {n / SR:.3f}s, RMS {20 * np.log10(target_rms):.2f} dBFS")

    cond_dir = WORK / "cond"
    cond_dir.mkdir(exist_ok=True)

    # condition 1: the literal mix, copied so all four files live in one place
    shutil.copyfile(MIX, cond_dir / "01_mix.wav")

    defs = [
        ("02_truth_harm", [TRUTH / "piano.wav", TRUTH / "bass.wav"], True),
        ("03_sep_harm", [MODEL6 / "mix_(piano)_BS-Roformer-SW.wav",
                         MODEL6 / "mix_(bass)_BS-Roformer-SW.wav"], True),
        ("04_sep_harm4", [MODEL6 / "mix_(piano)_BS-Roformer-SW.wav",
                          MODEL6 / "mix_(bass)_BS-Roformer-SW.wav",
                          MODEL6 / "mix_(guitar)_BS-Roformer-SW.wav",
                          MODEL6 / "mix_(other)_BS-Roformer-SW.wav"], True),
    ]

    stats = {
        "sample_rate": SR,
        "frames": int(n),
        "duration_sec": round(n / SR, 4),
        "mix_rms_dbfs": round(20 * np.log10(target_rms), 3),
        "mix_peak_dbfs": round(20 * np.log10(float(np.max(np.abs(mix_audio[:, 0])))), 3),
        "conditions": {},
    }

    for name, paths, do_normalise in defs:
        total = np.zeros(n, dtype=np.float32)
        per_part = {}
        for path in paths:
            part = load_48k_mono(path, n)
            per_part[path.stem] = round(rms_db(part), 3)
            total += part
        pre_rms = float(np.sqrt(np.mean(total.astype(np.float64) ** 2)))
        gain = 1.0
        if do_normalise and pre_rms > 0:
            gain = target_rms / pre_rms
        out = (total.astype(np.float64) * gain).astype(np.float32)
        peak = float(np.max(np.abs(out)))
        path = cond_dir / f"{name}.wav"
        sf.write(str(path), out, SR, subtype="FLOAT")
        stats["conditions"][name] = {
            "file": str(path),
            "sources": [str(p) for p in paths],
            "part_rms_dbfs": per_part,
            "sum_rms_dbfs_before": round(20 * np.log10(max(pre_rms, 1e-12)), 3),
            "gain_db": round(20 * np.log10(gain), 3),
            "rms_dbfs_after": round(rms_db(out), 3),
            "peak_dbfs_after": round(20 * np.log10(max(peak, 1e-12)), 3),
            "clip": bool(peak > 0.999),
        }
        print(f"{name}: pre {stats['conditions'][name]['sum_rms_dbfs_before']:6.2f} dBFS "
              f"gain {stats['conditions'][name]['gain_db']:+6.2f} dB "
              f"-> rms {stats['conditions'][name]['rms_dbfs_after']:6.2f} "
              f"peak {stats['conditions'][name]['peak_dbfs_after']:6.2f} dBFS")
        del total, out

    stats["conditions"]["01_mix"] = {
        "file": str(cond_dir / "01_mix.wav"),
        "sources": [str(MIX)],
        "gain_db": 0.0,
        "rms_dbfs_after": stats["mix_rms_dbfs"],
        "peak_dbfs_after": stats["mix_peak_dbfs"],
        "clip": False,
        "note": "byte-identical copy of synth6/mix.wav, no gain applied",
    }
    (WORK / "levels.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("wrote", WORK / "levels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
