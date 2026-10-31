"""Routing matrix on the synthetic drum mix: which output stem captured which true part.

For every (output stem i, true part j): alpha = <stem_i, part_j> / <part_j, part_j>
(alpha ~ 1 means stem i carries part j at full level; negative means inverted/anti-phase).
Also reports energy capture pct = alpha * <stem_i,part_j> / E_j (share of part j's energy
accounted for by stem i), and the sum-reconstruction residual.

Prints ROUTING {...}.
"""
import json
import os

import numpy as np
import soundfile as sf

H = r"D:\ChordStudio\research\separation\drums"
OUT = ["kick", "snare", "toms", "hh", "ride", "crash"]
SRC = ["kick", "snare", "hh", "crash", "toms"]
SR = 44100


def load(path):
    x, _ = sf.read(path, always_2d=True, dtype="float64")
    return x.T


def db(x):
    v = float(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(v) if v > 1e-12 else -np.inf


def main():
    mix = load(os.path.join(H, "synth.wav"))
    srcs = {k: load(os.path.join(H, "synth_sources", f"{k}.wav")) for k in SRC}
    stems = {}
    for n in OUT:
        cand = [p for p in os.listdir(os.path.join(H, "out_synth")) if f"_({n})_" in p]
        stems[n] = load(os.path.join(H, "out_synth", cand[0]))

    n = min([mix.shape[1]] + [v.shape[1] for v in srcs.values()] + [v.shape[1] for v in stems.values()])
    mix = mix[:, :n]
    with open(os.path.join(H, "synth_gain.json"), encoding="utf-8") as f:
        g = float(json.load(f)["gain"])
    srcs = {k: v[:, :n] * g for k, v in srcs.items()}  # match the mixture's peak-normalisation gain
    stems = {k: v[:, :n] for k, v in stems.items()}

    e = {k: float(np.sum(v ** 2)) for k, v in srcs.items()}
    out = {
        "synth_gain": round(g, 4),
        "note": "alpha ~ 1.0 means this stem carries the whole true part at the mixture's level",
        "levels_dbfs": {"mix": round(db(mix), 2), **{k: round(db(v), 2) for k, v in srcs.items()},
                        **{f"stem:{k}": round(db(v), 2) for k, v in stems.items()}},
        "routing_alpha": {}, "capture_pct": {},
    }
    for i in OUT:
        out["routing_alpha"][i] = {}
        out["capture_pct"][i] = {}
        for j in SRC:
            a = float(np.dot(stems[i].ravel(), srcs[j].ravel()) / max(e[j], 1e-30))
            out["routing_alpha"][i][j] = round(a, 3)
            out["capture_pct"][i][j] = round(100 * a * a, 1)

    # which stem is the best home for each source (largest capture)
    out["best_stem_per_source"] = {
        j: max(OUT, key=lambda i: out["capture_pct"][i][j]) for j in SRC
    }
    s = sum(stems.values())
    out["reconstruction"] = {
        "sum_rms_dbfs": round(db(s), 2),
        "residual_rel_db": round(db(mix - s) - db(mix), 2),
        "corr_sum_vs_mix": round(float(np.corrcoef(s.ravel(), mix.ravel())[0, 1]), 4),
    }
    print("ROUTING " + json.dumps(out))


if __name__ == "__main__":
    main()
