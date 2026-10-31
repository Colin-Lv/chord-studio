"""Reproducible, ground-truth-free comparison metrics for separated stems.

No SDR is computed: there is no multitrack ground truth. Only self-consistent
numbers are reported:
  * reconstruction residual: rms(original - sum(stems)) / rms(original), in dB
  * per-stem rms dBFS and "active ratio" (fraction of 50 ms frames above -40 dBFS)
  * cross-stem envelope correlation (50 ms RMS envelopes)

Usage:
  python analyze_stems.py --original <wav> --stems a.wav b.wav --label name
Outputs one JSON line prefixed with METRICS.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

FRAME = 0.050  # 50 ms envelope frames
ACTIVE_DBFS = -40.0


def load(path):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    return x, sr


def align(x, n):
    """Pad/trim channel count and length to match the reference."""
    if x.shape[1] < n.shape[1]:
        x = np.repeat(x, n.shape[1], axis=1)[:, : n.shape[1]]
    elif x.shape[1] > n.shape[1]:
        x = x[:, : n.shape[1]]
    if x.shape[0] < n.shape[0]:
        x = np.pad(x, ((0, n.shape[0] - x.shape[0]), (0, 0)))
    elif x.shape[0] > n.shape[0]:
        x = x[: n.shape[0]]
    return x


def dbfs(x):
    r = np.sqrt(np.mean(x**2))
    return float(20 * np.log10(r)) if r > 0 else -np.inf


def envelope(x, sr, frame=FRAME):
    hop = max(1, int(round(frame * sr)))
    mono = x.mean(axis=1)
    n = len(mono) // hop
    if n == 0:
        return np.zeros(1)
    return np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1))


def active_ratio(x, sr):
    env = envelope(x, sr)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.maximum(env, 1e-12))
    return float(np.mean(db > ACTIVE_DBFS))


def corr(a, b):
    if len(a) < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    orig, sr = load(args.original)
    stems = []
    for p in args.stems:
        s, sr2 = load(p)
        if sr2 != sr:
            raise SystemExit(f"sample rate mismatch: {p} {sr2} vs {sr}")
        stems.append(align(s, orig))

    total = np.sum(stems, axis=0)
    num = np.sqrt(np.mean((orig - total) ** 2))
    den = np.sqrt(np.mean(orig**2))
    residual_db = float(20 * np.log10(num / den)) if den > 0 and num > 0 else float("-inf")

    envs = {os.path.basename(p): envelope(s, sr) for p, s in zip(args.stems, stems)}
    names = list(envs)
    corr_mat = {a: {b: corr(envs[a], envs[b]) for b in names} for a in names}

    out = {
        "label": args.label,
        "original": os.path.basename(args.original),
        "sample_rate": sr,
        "duration_s": round(len(orig) / sr, 3),
        "n_stems": len(stems),
        "reconstruction_residual_db": round(residual_db, 2),
        "stems": [
            {
                "file": os.path.basename(p),
                "rms_dbfs": round(dbfs(s), 2),
                "peak_dbfs": round(dbfs(np.abs(s).max() * np.ones(1)), 2),
                "active_ratio": round(active_ratio(s, sr), 3),
            }
            for p, s in zip(args.stems, stems)
        ],
        "envelope_corr_50ms": corr_mat,
    }
    print("METRICS " + json.dumps(out))


if __name__ == "__main__":
    main()
