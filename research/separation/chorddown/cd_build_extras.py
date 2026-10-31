"""Two control conditions that isolate the confounds of the main four.

06_sep_harm_natural : separated piano+bass at NATURAL level (gain 0 dB, i.e. its own
                      RMS, 2 dB below the mix) -- tests whether condition 3's +2.0 dB
                      RMS matching is what changed the answer.
07_truth_harm4      : truth piano+bass+guitar, RMS-matched to the mix -- tests whether
                      adding the guitar costs a bar by itself, without separation
                      artefacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from cd_build_conditions import MODEL6, TRUTH, load_48k_mono, rms_db
from cd_common import SYNTH, WORK

SR = 48000
MIX_PATH = SYNTH / "mix.wav"


def main() -> int:
    mix_audio, sr = sf.read(str(MIX_PATH), always_2d=True, dtype="float32")
    n = mix_audio.shape[0]
    target_rms = float(np.sqrt(np.mean(mix_audio[:, 0].astype(np.float64) ** 2)))
    cond = WORK / "cond"
    stats = json.loads((WORK / "levels.json").read_text(encoding="utf-8"))

    defs = [
        ("06_sep_harm_natural",
         [MODEL6 / "mix_(piano)_BS-Roformer-SW.wav",
          MODEL6 / "mix_(bass)_BS-Roformer-SW.wav"], False),
        ("07_truth_harm4",
         [TRUTH / "piano.wav", TRUTH / "bass.wav", TRUTH / "guitar.wav"], True),
    ]

    for name, paths, do_norm in defs:
        total = np.zeros(n, dtype=np.float32)
        for path in paths:
            total += load_48k_mono(path, n)
        pre = float(np.sqrt(np.mean(total.astype(np.float64) ** 2)))
        gain = (target_rms / pre) if (do_norm and pre > 0) else 1.0
        out = (total.astype(np.float64) * gain).astype(np.float32)
        path = cond / f"{name}.wav"
        sf.write(str(path), out, SR, subtype="FLOAT")
        peak = float(np.max(np.abs(out)))
        stats["conditions"][name] = {
            "file": str(path),
            "sources": [str(p) for p in paths],
            "sum_rms_dbfs_before": round(20 * np.log10(max(pre, 1e-12)), 3),
            "gain_db": round(20 * np.log10(gain), 3),
            "rms_dbfs_after": round(rms_db(out), 3),
            "peak_dbfs_after": round(20 * np.log10(max(peak, 1e-12)), 3),
            "clip": bool(peak > 0.999),
        }
        print(f"{name}: gain {stats['conditions'][name]['gain_db']:+.2f} dB "
              f"-> rms {stats['conditions'][name]['rms_dbfs_after']:.2f} dBFS "
              f"peak {stats['conditions'][name]['peak_dbfs_after']:.2f} dBFS")

    (WORK / "levels.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
