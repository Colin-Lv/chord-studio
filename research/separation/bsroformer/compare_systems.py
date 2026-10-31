"""Cross-system agreement between BS-RoFormer (2 stems) and htdemucs (4 stems).

Since there is no ground truth, the only honest "which is better" proxy is
(a) how closely the two independent systems agree on the same stem, and
(b) internal consistency (residual, solo/leakage behaviour).

Compares vocals vs vocals, and roformer-instrumental vs demucs(drums+bass+other),
using 50 ms RMS envelope correlation and a null-test correlation against a
time-shifted copy (shifting by 2 s destroys alignment, so a high null value
would indicate the correlation is driven by slow loudness trends, not content).

Usage: python compare_systems.py --dirA <roformer out dir> --dirB <demucs stem dir> --label x
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

FRAME = 0.050


def load(path):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    return x, sr


def mono(x):
    return x.mean(axis=1)


def envelope(x, sr, frame=FRAME):
    hop = max(1, int(round(frame * sr)))
    m = mono(x)
    n = len(m) // hop
    return np.sqrt(np.mean(m[: n * hop].reshape(n, hop) ** 2, axis=1))


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def shift_null(a, b, sr, seconds=2.0):
    """Correlation after circularly shifting b: destroys time alignment."""
    k = int(round(seconds / FRAME))
    return corr(a, np.roll(b, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roformer-dir", required=True)
    ap.add_argument("--demucs-dir", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    def find(d, token):
        hits = [f for f in os.listdir(d) if token.lower() in f.lower() and f.lower().endswith(".wav")]
        if not hits:
            raise SystemExit(f"no {token} in {d}")
        return os.path.join(d, hits[0])

    rv = find(args.roformer_dir, "vocals")
    ri = find(args.roformer_dir, "instrumental")
    dv = os.path.join(args.demucs_dir, "vocals.wav")
    idd = args.demucs_dir

    rv_x, sr = load(rv)
    ri_x, _ = load(ri)
    dv_x, _ = load(dv)
    parts = []
    for n in ("drums", "bass", "other"):
        x, _ = load(os.path.join(idd, f"{n}.wav"))
        parts.append(x)
    di_x = np.sum(parts, axis=0)

    n = min(len(rv_x), len(dv_x), len(di_x))
    rv_x, ri_x, dv_x, di_x = rv_x[:n], ri_x[:n], dv_x[:n], di_x[:n]

    ev_r, ev_d = envelope(rv_x, sr), envelope(dv_x, sr)
    ei_r, ei_d = envelope(ri_x, sr), envelope(di_x, sr)

    # energy ratio of each system's vocals, to compare how much signal each calls "vocals"
    def rms_db(x):
        r = np.sqrt(np.mean(x**2))
        return round(float(20 * np.log10(r)), 2) if r > 0 else None

    out = {
        "label": args.label,
        "vocals_env_corr": round(corr(ev_r, ev_d), 3),
        "vocals_env_corr_null_2s": round(shift_null(ev_r, ev_d, sr), 3),
        "instrumental_env_corr": round(corr(ei_r, ei_d), 3),
        "instrumental_env_corr_null_2s": round(shift_null(ei_r, ei_d, sr), 3),
        "rms_dbfs": {
            "roformer_vocals": rms_db(rv_x),
            "demucs_vocals": rms_db(dv_x),
            "roformer_instrumental": rms_db(ri_x),
            "demucs_instrumental_sum": rms_db(di_x),
        },
        "energy_split_vocals_fraction": {
            "roformer": round(float(np.sum(rv_x**2) / (np.sum(rv_x**2) + np.sum(ri_x**2))), 3),
            "demucs": round(float(np.sum(dv_x**2) / (np.sum(dv_x**2) + np.sum(di_x**2))), 3),
        },
    }
    print("COMPARE " + json.dumps(out))


if __name__ == "__main__":
    main()
