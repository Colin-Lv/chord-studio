"""Build a controlled synthetic drum mix with known ground-truth parts.

Why: the only available song's htdemucs drums stem is ~95% below 150 Hz (kick only),
so it cannot exercise the snare/toms/hh/ride/crash paths of the model at all.
This synthetic track has known parts, so the model's routing can be measured.

Writes: synth.wav (mixture) + synth_sources/<part>.wav (ground truth).
"""
import os
import json

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

H = r"D:\ChordStudio\research\separation\drums"
SR = 44100
DUR = 20.0
N = int(SR * DUR)
rng = np.random.default_rng(1234)


def bp(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x)


def hp(x, f, order=4):
    sos = butter(order, f, btype="high", fs=SR, output="sos")
    return sosfilt(sos, x)


def add(buf, start_s, sig):
    i = int(start_s * SR)
    j = min(N, i + len(sig))
    buf[i:j] += sig[: j - i]


def kick_hit():
    L = int(0.30 * SR)
    tt = np.arange(L) / SR
    f = 110 * np.exp(-tt * 18) + 48
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 9)


def snare_hit():
    L = int(0.20 * SR)
    tt = np.arange(L) / SR
    noise = bp(rng.standard_normal(L), 1200, 9000) * np.exp(-tt * 30)
    body = np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 22)
    return 0.55 * noise + 0.45 * body


def hat_hit():
    L = int(0.06 * SR)
    tt = np.arange(L) / SR
    return hp(rng.standard_normal(L), 7000) * np.exp(-tt * 90)


def crash_hit():
    L = int(1.6 * SR)
    tt = np.arange(L) / SR
    return hp(rng.standard_normal(L), 3500) * np.exp(-tt * 2.2)


def tom_hit(f0=150):
    L = int(0.35 * SR)
    tt = np.arange(L) / SR
    f = f0 * np.exp(-tt * 6) + f0 * 0.55
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 8)


parts = {k: np.zeros(N) for k in ["kick", "snare", "hh", "crash", "toms"]}

for t in np.arange(0, DUR, 1.0):                       # 60 BPM-ish kick on 1 & 3
    add(parts["kick"], t, 0.95 * kick_hit())
for t in np.arange(0.5, DUR, 1.0):                     # snare on 2 & 4
    add(parts["snare"], t, 0.7 * snare_hit())
for t in np.arange(0.0, DUR, 0.25):                    # 8th-note hats
    add(parts["hh"], t, 0.18 * hat_hit())
for t in [0.0, 10.0]:                                  # two crashes
    add(parts["crash"], t, 0.35 * crash_hit())
for t in [14.0, 14.5, 15.0, 18.0, 18.5, 19.0]:          # tom fills
    add(parts["toms"], t, 0.45 * tom_hit(140 if t % 1 else 170))

mix = sum(parts.values())
gain = 0.7 / np.max(np.abs(mix))
mix = mix * gain

os.makedirs(os.path.join(H, "synth_sources"), exist_ok=True)
st = lambda x: np.stack([x, x], axis=1)
sf.write(os.path.join(H, "synth.wav"), st(mix), SR, subtype="PCM_16")
for k, v in parts.items():
    sf.write(os.path.join(H, "synth_sources", f"{k}.wav"), st(v), SR, subtype="PCM_16")
with open(os.path.join(H, "synth_gain.json"), "w", encoding="utf-8") as f:
    json.dump({"gain": gain, "note": "synth.wav = gain * sum(parts); multiply parts by gain to match"},
              f)
print("wrote synth.wav", os.path.getsize(os.path.join(H, "synth.wav")), "bytes; parts:", list(parts),
      "gain:", gain)
