"""Cross-check MDX23C-DrumSep: full-mix input vs drums-stem input, leakage, band energy.

Both runs cover the same music (drums20.wav == 96..116 s of virtual.wav == 96..116 s of
the htdemucs_ft drums stem), so the two input modes can be compared sample-region-aligned.

Prints COMPARE {...}.
"""
import json
import os

import numpy as np
import soundfile as sf

H = r"D:\ChordStudio\research\separation\drums"
STEMS = (r"D:\ChordStudio\data\stems\d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f"
         r"\htdemucs_ft\wav")
MIX = r"D:\ChordStudio\research\separation\virtual.wav"
NAMES = ["kick", "snare", "toms", "hh", "ride", "crash"]
SR = 44100
T0, DUR = 96.0, 20.0


def load(path, start_s=None, dur_s=None):
    if start_s is None:
        x, sr = sf.read(path, always_2d=True, dtype="float64")
        return x.T, sr
    sr = sf.info(path).samplerate
    x, _ = sf.read(path, start=int(round(start_s * sr)), frames=int(round(dur_s * sr)),
                   always_2d=True, dtype="float64")
    return x.T, sr


def db(x):
    r = float(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(r) if r > 1e-12 else -np.inf


def r(x, y):
    return round(float(np.corrcoef(x.ravel(), y.ravel())[0, 1]), 4)


def load_stems(d, start=None, dur=None):
    out = {}
    for n in NAMES:
        cand = [p for p in os.listdir(d) if f"_({n})_" in p]
        assert len(cand) == 1, (d, n, cand)
        out[n], _ = load(os.path.join(d, cand[0]), start, dur)
    return out


def band_shares(x, sr=SR):
    """Energy share in low (<150 Hz) / mid (150-2000) / high (>6000) bands, mono."""
    m = x.mean(axis=0)
    f = np.fft.rfft(m * np.hanning(len(m)))
    freqs = np.fft.rfftfreq(len(m), 1 / sr)
    p = np.abs(f) ** 2
    tot = p.sum() + 1e-30
    return {
        "low_pct": round(100 * float(p[freqs < 150].sum() / tot), 2),
        "mid_pct": round(100 * float(p[(freqs >= 150) & (freqs < 2000)].sum() / tot), 2),
        "high_pct": round(100 * float(p[freqs > 6000].sum() / tot), 2),
    }


def main():
    mix, sr = load(MIX, T0, DUR)
    drums_ref, _ = load(os.path.join(STEMS, "drums.wav"), T0, DUR)
    other = {}
    for n in ["vocals", "bass", "other"]:
        other[n], _ = load(os.path.join(STEMS, f"{n}.wav"), T0, DUR)
    non_drum = sum(other.values())
    n = min([mix.shape[1], drums_ref.shape[1]] + [v.shape[1] for v in other.values()])
    mix, drums_ref, non_drum = mix[:, :n], drums_ref[:, :n], non_drum[:, :n]

    from_mix = load_stems(os.path.join(H, "out_20s"))
    from_drums = load_stems(os.path.join(H, "out_full"), T0, DUR)
    s_mix = sum(from_mix[k][:, :n] for k in NAMES)
    s_drm = sum(from_drums[k][:, :n] for k in NAMES)

    # how well does the htdemucs 4-stem partition reconstruct the mix? (context for `non_drum`)
    recon = drums_ref + non_drum

    out = {
        "region": f"{T0}-{T0 + DUR}s",
        "levels": {
            "mix": round(db(mix), 2),
            "htdemucs_drums": round(db(drums_ref), 2),
            "htdemucs_non_drum": round(db(non_drum), 2),
            "htdemucs_4stem_sum_vs_mix_rel_db": round(db(mix - recon) - db(mix), 2),
            "drumsep_sum_from_mix": round(db(s_mix), 2),
            "drumsep_sum_from_drums": round(db(s_drm), 2),
        },
        "corr": {
            "sum_from_mix vs mix": r(s_mix, mix),
            "sum_from_mix vs htdemucs_drums": r(s_mix, drums_ref),
            "sum_from_mix vs non_drum": r(s_mix, non_drum),
            "sum_from_drums vs htdemucs_drums": r(s_drm, drums_ref),
            "sum_from_drums vs non_drum": r(s_drm, non_drum),
        },
        "residual": {
            "from_mix: (mix-sum) rel dB": round(db(mix - s_mix) - db(mix), 2),
            "from_mix: (drums-sum) rel dB": round(db(drums_ref - s_mix) - db(drums_ref), 2),
            "from_drums: (drums-sum) rel dB": round(db(drums_ref - s_drm) - db(drums_ref), 2),
        },
        "leakage_from_mix": {
            "gain_on_non_drum": round(float(np.dot(s_mix.ravel(), non_drum.ravel()) /
                                            max(np.dot(non_drum.ravel(), non_drum.ravel()), 1e-30)), 4),
            "energy_explained_by_non_drum_pct": round(100 * float(np.dot(s_mix.ravel(), non_drum.ravel()) ** 2 /
                                                                   (np.dot(s_mix.ravel(), s_mix.ravel()) *
                                                                    np.dot(non_drum.ravel(), non_drum.ravel()) + 1e-30)), 2),
            "excess_vs_drum_ref_db": round(db(s_mix) - db(drums_ref), 2),
        },
        "per_stem_from_mix": {k: {"rms_dbfs": round(db(from_mix[k]), 2), **band_shares(from_mix[k]),
                                  "sum_vs_mix_corr": r(from_mix[k], mix)} for k in NAMES},
        "per_stem_from_drums": {k: {"rms_dbfs": round(db(from_drums[k][:, :n]), 2),
                                    **band_shares(from_drums[k][:, :n]),
                                    "sum_vs_drums_corr": r(from_drums[k][:, :n], drums_ref)} for k in NAMES},
        "same_stem_corr_mix_vs_drums_input": {k: r(from_mix[k][:, :n], from_drums[k][:, :n]) for k in NAMES},
    }

    # cross-stem correlation matrix (from_drums): are the stems actually distinct?
    m = np.array([from_drums[k][:, :n].ravel() for k in NAMES])
    c = np.corrcoef(m)
    out["cross_stem_corr_from_drums"] = {NAMES[i]: {NAMES[j]: round(float(c[i, j]), 3) for j in range(len(NAMES))}
                                         for i in range(len(NAMES))}
    print("COMPARE " + json.dumps(out))


if __name__ == "__main__":
    main()
