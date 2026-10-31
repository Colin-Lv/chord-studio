"""Per-stem sanity metrics for the MDX23C-DrumSep outputs.

Reads the input, every stem in an output dir, and optionally a reference
drum stem, then prints METRICS {...} with:
  rms_dbfs, peak_dbfs, active_ratio (50 ms frames above -40 dBFS),
  energy share, and sum/residual checks.

Usage:
  python analyze_drums.py --input <wav> --stems <dir> [--ref <ref drums.wav>]
                          [--trim-start 96] [--trim-dur 20]
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf


def load(path):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    return x.T, sr  # channels x samples


def rms_dbfs(x):
    r = float(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(r) if r > 1e-12 else -np.inf


def peak_dbfs(x):
    p = float(np.max(np.abs(x)))
    return 20 * np.log10(p) if p > 1e-12 else -np.inf


def active_ratios(x, sr, frame_s=0.05, floor_db=-40.0):
    """50 ms frame RMS. Returns (absolute ratio above floor_db, ratio above floor relative to peak)."""
    n = max(1, int(round(frame_s * sr)))
    mono = x.mean(axis=0)
    frames = len(mono) // n
    if frames == 0:
        return 0.0, 0.0
    f = mono[: frames * n].reshape(frames, n)
    fr = np.sqrt(np.mean(f ** 2, axis=1))
    db = 20 * np.log10(np.maximum(fr, 1e-12))
    ref = db.max() if db.max() > -200 else -200.0
    return float(np.mean(db > floor_db)), float(np.mean(db > ref + floor_db))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--trim-start", type=float, default=None)
    ap.add_argument("--trim-dur", type=float, default=None)
    args = ap.parse_args()

    mix, sr = load(args.input)
    files = sorted(glob.glob(os.path.join(args.stems, "*.wav")))
    if not files:
        raise SystemExit(f"no wav files in {args.stems}")

    stems = {}
    names = []
    for p in files:
        x, sr2 = load(p)
        assert sr2 == sr, (sr, sr2, p)
        base = os.path.basename(p)
        # drums20_(kick)_MDX23C-....wav -> kick
        name = base.split("_(")[1].split(")")[0] if "_(" in base else os.path.splitext(base)[0]
        stems[name] = x
        names.append(name)

    n = min([mix.shape[1]] + [v.shape[1] for v in stems.values()])
    mix = mix[:, :n]
    stems = {k: v[:, :n] for k, v in stems.items()}

    total_rms = rms_dbfs(mix)
    per = {}
    for k in names:
        v = stems[k]
        ar_abs, ar_rel = active_ratios(v, sr)
        per[k] = {
            "rms_dbfs": round(rms_dbfs(v), 2),
            "peak_dbfs": round(peak_dbfs(v), 2),
            "active_ratio_abs_minus40": round(ar_abs, 3),
            "active_ratio_rel_minus40": round(ar_rel, 3),
        }

    s = np.sum([stems[k] for k in names], axis=0)
    resid = mix - s
    sum_rms = rms_dbfs(s)

    out = {
        "input": os.path.basename(args.input),
        "sample_rate": sr,
        "duration_s": round(n / sr, 3),
        "input_rms_dbfs": round(total_rms, 2),
        "input_peak_dbfs": round(peak_dbfs(mix), 2),
        "stem_count": len(names),
        "stem_order": names,
        "per_stem": per,
        "sum_rms_dbfs": round(sum_rms, 2),
        "residual_rms_dbfs": round(rms_dbfs(resid), 2),
        "residual_rel_db": round(rms_dbfs(resid) - total_rms, 2),
        "energy_share_pct": {k: round(100 * float(np.sum(stems[k] ** 2)) / max(float(np.sum(s ** 2)), 1e-20), 2)
                             for k in names},
    }

    # correlation of sum with the mix (1.0 == perfect reconstruction shape)
    a = s.ravel()
    b = mix.ravel()
    out["sum_vs_input_corr"] = round(float(np.corrcoef(a, b)[0, 1]), 4)

    if args.ref:
        ref, rsr = load(args.ref)
        assert rsr == sr, (sr, rsr)
        if args.trim_start is not None:
            st = int(round(args.trim_start * sr))
            du = int(round((args.trim_dur or (n / sr)) * sr))
            ref = ref[:, st : st + du]
        m = min(n, ref.shape[1])
        ref = ref[:, :m]
        sm = s[:, :m]
        mm = mix[:, :m]
        # gain-fit the stem sum to the reference (removes pure level differences)
        g = float(np.dot(sm.ravel(), ref.ravel()) / max(np.dot(sm.ravel(), sm.ravel()), 1e-20))
        out["ref"] = {
            "file": os.path.basename(args.ref),
            "samples": m,
            "ref_rms_dbfs": round(rms_dbfs(ref), 2),
            "sum_rms_dbfs": round(rms_dbfs(sm), 2),
            "corr_sum_vs_ref": round(float(np.corrcoef(sm.ravel(), ref.ravel())[0, 1]), 4),
            "residual_rel_db": round(rms_dbfs(ref - sm) - rms_dbfs(ref), 2),
            "gain_to_ref": round(g, 4),
            "residual_rel_db_gainfit": round(rms_dbfs(ref - g * sm) - rms_dbfs(ref), 2),
            "mix_vs_ref_corr": round(float(np.corrcoef(mm.ravel(), ref.ravel())[0, 1]), 4),
            "mix_vs_ref_residual_rel_db": round(rms_dbfs(mm - ref) - rms_dbfs(ref), 2),
        }
        # frame-envelope correlation, 50 ms
        fr = max(1, int(0.05 * sr))
        def env(z):
            k = z.shape[1] // fr
            return np.sqrt(np.mean(z[:, : k * fr].reshape(2, k, fr) ** 2, axis=(0, 2)))
        out["ref"]["env_corr_50ms"] = round(float(np.corrcoef(env(sm), env(ref))[0, 1]), 4)

    print("METRICS " + json.dumps(out))


if __name__ == "__main__":
    main()
