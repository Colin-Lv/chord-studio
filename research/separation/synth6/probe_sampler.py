"""Probe: does the project sampler expose a GM percussion bank (128, program 0)?

Also checks what render_pm's hardcoded bank=0 would do with an is_drum track.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, r"D:\ChordStudio\research\boundary-dataset")
import render as R  # noqa: E402

sfx = R.make_sampler_class(R.load_sf2_module())(str(R.find_soundfont(None)))
print("soundfont:", sfx.path)

for bank, prog in ((0, 0), (0, 24), (0, 25), (0, 33), (128, 0)):
    zones = sfx.preset_zones(bank, prog)
    names = sorted({z.sample_id for z in zones if z.sample_id is not None})[:6]
    print(f"bank={bank:3d} program={prog:3d} zones={len(zones):3d} "
          f"keyrange={min((z.key_low for z in zones), default=-1)}..{max((z.key_high for z in zones), default=-1)}")

# What a kick note (36) renders as under bank 0 vs bank 128
for bank in (0, 128):
    t0 = time.time()
    audio, _pan = sfx.render(0, 36, 100, 0.5, sr=48000, bank=bank, gain=0.65)
    print(f"bank={bank:3d} note36 len={audio.size / 48000:.2f}s "
          f"peak={float(np.max(np.abs(audio))):.4f} rms={float(np.sqrt(np.mean(audio ** 2))):.4f} "
          f"({time.time() - t0:.2f}s)")

# Spectral centroid comparison: piano note vs kick-ish drum hit
def centroid(x, sr=48000):
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, 1 / sr)
    denom = spec.sum()
    return float((spec * freqs).sum() / denom) if denom else 0.0

piano36, _ = sfx.render(0, 36, 100, 0.5, sr=48000, bank=0, gain=0.65)
kick128, _ = sfx.render(0, 36, 100, 0.5, sr=48000, bank=128, gain=0.65)
print(f"centroid bank0 note36 = {centroid(piano36):7.0f} Hz")
print(f"centroid bank128 note36 = {centroid(kick128):7.0f} Hz")
