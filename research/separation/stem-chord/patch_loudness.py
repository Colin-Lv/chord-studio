"""Match the harmonic mix loudness to the mixture before fusion (research note §8.5)."""
from __future__ import annotations

from pathlib import Path

ENGINE = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\fusion_engine.py")
text = ENGINE.read_text(encoding="utf-8")

OLD = '''    peak = float(np.abs(total).max())
    if peak > 0.999:
        total *= 0.999 / peak
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), total, rate, subtype="PCM_16")
    return target'''
NEW = '''    # Match the loudness of the full mixture (sum of every stem) so the two posterior runs
    # see comparable levels; without this the fusion partly measures a gain difference.
    import math

    mixture_rms = math.sqrt(sum(10 ** (track["rms_db"] / 10)
                                for track in manifest["tracks"] if "rms_db" in track))
    harmonic_rms = float(np.sqrt((total.astype(np.float64) ** 2).mean()))
    if harmonic_rms > 0 and mixture_rms > 0:
        gain = mixture_rms / harmonic_rms
        gain = max(0.25, min(4.0, gain))
        total *= gain
    peak = float(np.abs(total).max())
    if peak > 0.999:
        total *= 0.999 / peak
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), total, rate, subtype="PCM_16")
    return target'''
assert text.count(OLD) == 1
ENGINE.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("fusion_engine: loudness match")
