"""Band energy + 5 s level profile: input drum stem vs the 6 separated stems.

Answers: is the drum-stem split (kick 86% of energy) a model failure or does the
input itself carry almost no high-band energy? Prints PROFILE {...}.
"""
import json
import os

import numpy as np
import soundfile as sf

H = r"D:\ChordStudio\research\separation\drums"
STEMS = (r"D:\ChordStudio\data\stems\d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f"
         r"\htdemucs_ft\wav")
NAMES = ["kick", "snare", "toms", "hh", "ride", "crash"]
SR = 44100


def load(path):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    return x.T, sr


def db(x):
    v = float(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(v) if v > 1e-12 else -np.inf


def bands(x):
    m = x.mean(axis=0)
    f = np.fft.rfft(m * np.hanning(len(m)))
    fr = np.fft.rfftfreq(len(m), 1 / SR)
    p = np.abs(f) ** 2
    t = p.sum() + 1e-30
    edges = [(0, 60), (60, 150), (150, 400), (400, 2000), (2000, 6000), (6000, 12000), (12000, 20000)]
    return {f"{a}-{b}Hz": round(100 * float(p[(fr >= a) & (fr < b)].sum() / t), 2) for a, b in edges}


def profile(x, win_s=5.0):
    w = int(win_s * SR)
    k = x.shape[1] // w
    return [round(db(x[:, i * w:(i + 1) * w]), 1) for i in range(k)]


def main():
    out = {}
    dr, _ = load(os.path.join(STEMS, "drums.wav"))
    out["input_drums"] = {
        "rms_dbfs": round(db(dr), 2),
        "bands_full": bands(dr),
        "bands_96_116": bands(dr[:, int(96 * SR):int(116 * SR)]),
        "rms_5s_profile": profile(dr),
    }
    out["stems"] = {}
    for n in NAMES:
        cand = [p for p in os.listdir(os.path.join(H, "out_full")) if f"_({n})_" in p]
        x, _ = load(os.path.join(H, "out_full", cand[0]))
        out["stems"][n] = {
            "rms_dbfs": round(db(x), 2),
            "bands_full": bands(x),
            "rms_5s_profile": profile(x),
        }
    print("PROFILE " + json.dumps(out))


if __name__ == "__main__":
    main()
