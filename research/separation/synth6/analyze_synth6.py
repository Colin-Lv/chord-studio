"""Align model outputs with the synthetic truth and score them.

Pipeline
--------
* truth stems are 48 kHz mono float32 (from build_synth6.py)
* model outputs are whatever audio-separator writes (44.1 kHz stereo)
* everything is brought to the *model's* sample rate (44.1 kHz), mono, and a
  common length, so only the estimate<->truth band is compared.  The 48->44.1
  downsample is the near-lossless direction here: the FluidR3 samples are
  44.1 kHz, so the 48 kHz renders carry essentially nothing above 22.05 kHz.

Outputs
-------
* stems6/            six-stem model outputs, renamed, 44.1 kHz mono
* stems6_eval/       same, but only piano/guitar/bass/drums (harness input)
* stems2/            two-stem model outputs (vocals, instrumental)
* truth_eval/        truth stems at the eval rate/length
* truth2_eval/       vocals = digital silence, instrumental = mix
* score_six.json, score_two.json, confusion.json, levels.json

SI-SDR is computed with the project harness itself
(D:\\ChordStudio\\research\\separation\\eval\\stem_eval.py), both through its
CLI (diagonal-only reference mode) and through its imported `si_sdr_db`
function for the full truth x model cross matrix.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
EVAL_DIR = Path(r"D:\ChordStudio\research\separation\eval")
sys.path.insert(0, str(EVAL_DIR))
import stem_eval  # noqa: E402

TRUTH_NAMES = ("piano", "guitar", "bass", "drums")
SIX_STEMS = ("piano", "guitar", "bass", "drums", "vocals", "other")
TWO_STEMS = ("vocals", "instrumental")


def load_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1).astype(np.float64)
    if sr is not None and rate != sr:
        from math import gcd
        g = gcd(sr, rate)
        mono = resample_poly(mono, sr // g, rate // g)
        rate = sr
    return mono, rate


def discover_model_outputs(model_dir: Path) -> dict[str, Path]:
    """Map `mix_(piano)_MODEL.wav` -> 'piano'."""
    found: dict[str, Path] = {}
    if not model_dir.is_dir():
        return found
    for path in sorted(model_dir.glob("*.wav")):
        stem = path.stem
        if "(" not in stem or ")" not in stem:
            continue
        name = stem[stem.index("(") + 1:stem.index(")")].lower()
        found[name] = path
    return found


def rms_dbfs(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(x * x)))
    return round(20 * np.log10(max(r, 1e-12)), 3)


def peak_dbfs(x: np.ndarray) -> float:
    return round(20 * np.log10(max(float(np.max(np.abs(x))), 1e-12)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--six-dir", default=str(HERE / "model6"))
    ap.add_argument("--two-dir", default=str(HERE / "model2"))
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    six_raw = discover_model_outputs(Path(args.six_dir))
    two_raw = discover_model_outputs(Path(args.two_dir))
    print(f"[prep] six-stem outputs: {sorted(six_raw)}", flush=True)
    print(f"[prep] two-stem outputs: {sorted(two_raw)}", flush=True)
    if not six_raw:
        raise SystemExit(f"no six-stem outputs in {args.six_dir}")

    probe, probe_sr = load_mono(next(iter(six_raw.values())))
    eval_sr = probe_sr
    print(f"[prep] eval sample rate = {eval_sr} Hz", flush=True)

    truth = {name: load_mono(HERE / "truth" / f"{name}.wav", eval_sr)[0]
             for name in TRUTH_NAMES}
    mix, _ = load_mono(HERE / "mix.wav", eval_sr)

    six = {name: load_mono(path, eval_sr)[0] for name, path in six_raw.items()}
    two = {name: load_mono(path, eval_sr)[0] for name, path in two_raw.items()}

    lengths = [a.size for a in (*truth.values(), mix, *six.values(), *two.values())]
    n = min(lengths)
    print(f"[prep] length before trim: {sorted(set(lengths))} -> common {n} samples "
          f"({n / eval_sr:.3f}s)", flush=True)
    truth = {k: v[:n] for k, v in truth.items()}
    mix = mix[:n]
    six = {k: v[:n] for k, v in six.items()}
    two = {k: v[:n] for k, v in two.items()}

    truth_dir = HERE / "truth_eval"
    truth_dir.mkdir(exist_ok=True)
    for name, sig in truth.items():
        sf.write(str(truth_dir / f"{name}.wav"), sig.astype(np.float32), eval_sr,
                 subtype="FLOAT")

    stems6_dir = HERE / "stems6"
    stems6_dir.mkdir(exist_ok=True)
    for name, sig in six.items():
        sf.write(str(stems6_dir / f"{name}.wav"), sig.astype(np.float32), eval_sr,
                 subtype="FLOAT")

    stems6_eval = HERE / "stems6_eval"
    stems6_eval.mkdir(exist_ok=True)
    for name in TRUTH_NAMES:
        if name in six:
            sf.write(str(stems6_eval / f"{name}.wav"), six[name].astype(np.float32),
                     eval_sr, subtype="FLOAT")

    stems2_dir = HERE / "stems2"
    stems2_dir.mkdir(exist_ok=True)
    for name, sig in two.items():
        sf.write(str(stems2_dir / f"{name}.wav"), sig.astype(np.float32), eval_sr,
                 subtype="FLOAT")

    truth2_dir = HERE / "truth2_eval"
    truth2_dir.mkdir(exist_ok=True)
    if two:
        sf.write(str(truth2_dir / "vocals.wav"), np.zeros(n, dtype=np.float32),
                 eval_sr, subtype="FLOAT")
        sf.write(str(truth2_dir / "instrumental.wav"), mix.astype(np.float32),
                 eval_sr, subtype="FLOAT")

    mix_eval_path = HERE / "mix_eval.wav"
    sf.write(str(mix_eval_path), mix.astype(np.float32), eval_sr, subtype="FLOAT")

    # ---------------- harness runs (CLI, exactly as the project calls it) ----
    def run_harness(truth_path: Path, stems_path: Path, out_json: Path):
        proc = subprocess.run(
            [sys.executable, str(EVAL_DIR / "stem_eval.py"),
             "--truth", str(truth_path), "--stems", str(stems_path)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[harness] {stems_path.name} failed: {proc.stderr.strip()}", flush=True)
            return None
        result = json.loads(proc.stdout)
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[harness] {stems_path.name}: mean SI-SDR "
              f"{result.get('mean_si_sdr_db')} dB", flush=True)
        return result

    score_six = run_harness(truth_dir, stems6_eval, HERE / "score_six.json")
    score_two = (run_harness(truth2_dir, stems2_dir, HERE / "score_two.json")
                 if two else None)

    # ---------------- full cross matrix via the harness's own SI-SDR ---------
    def si(ref: np.ndarray, est: np.ndarray):
        return stem_eval._round_or_none(
            stem_eval.si_sdr_db(ref[:, None], est[:, None]), 3)

    candidates: dict[str, np.ndarray] = {}
    for name in TRUTH_NAMES:
        if name in six:
            candidates[f"6stem:{name}"] = six[name]
    for name in ("vocals", "other"):
        if name in six:
            candidates[f"6stem:{name}"] = six[name]
    for name in ("vocals", "instrumental"):
        if name in two:
            candidates[f"2stem:{name}"] = two[name]
    candidates["baseline:mix"] = mix

    matrix: dict[str, dict[str, float | None]] = {}
    for tname in TRUTH_NAMES:
        matrix[tname] = {cname: si(truth[tname], sig) for cname, sig in candidates.items()}

    levels = {
        "eval_sample_rate": eval_sr,
        "duration_sec": round(n / eval_sr, 3),
        "truth": {k: {"rms_dbfs": rms_dbfs(v), "peak_dbfs": peak_dbfs(v)}
                  for k, v in truth.items()},
        "mix": {"rms_dbfs": rms_dbfs(mix), "peak_dbfs": peak_dbfs(mix)},
        "six_stem": {k: {"rms_dbfs": rms_dbfs(v), "peak_dbfs": peak_dbfs(v)}
                     for k, v in six.items()},
        "two_stem": {k: {"rms_dbfs": rms_dbfs(v), "peak_dbfs": peak_dbfs(v)}
                     for k, v in two.items()},
    }

    # best model stem for each truth row (excluding the mix baseline)
    best: dict[str, dict] = {}
    for tname, row in matrix.items():
        scored = {k: v for k, v in row.items()
                  if v is not None and not k.startswith("baseline:")}
        if scored:
            k = max(scored, key=lambda key: scored[key])
            best[tname] = {"best_stem": k, "si_sdr_db": scored[k],
                           "own_stem_si_sdr_db": row.get(f"6stem:{tname}"),
                           "mix_baseline_si_sdr_db": row.get("baseline:mix")}

    # ---------------- how much of each model stem is explained by each truth -
    # Least-squares projection of every model stem onto the four truth stems.
    # share = energy explained by truth stem t / energy of the model stem.
    # This is the direct "does the piano stem actually contain guitar?" test.
    basis = np.stack([truth[k] for k in TRUTH_NAMES], axis=1)      # (n, 4)
    gram = basis.T @ basis
    cond = float(np.linalg.cond(gram))
    shares: dict[str, dict[str, float]] = {}
    fit: dict[str, dict[str, float]] = {}
    for cname, sig in candidates.items():
        coeffs, *_ = np.linalg.lstsq(basis, sig, rcond=None)
        energy = float(np.dot(sig, sig))
        if energy <= 0:
            continue
        share = {t: round(float(c * (gram @ coeffs)[i]) / energy, 4)
                 for i, (t, c) in enumerate(zip(TRUTH_NAMES, coeffs))}
        reconstructed = basis @ coeffs
        r2 = 1.0 - float(np.dot(sig - reconstructed, sig - reconstructed)) / energy
        shares[cname] = share
        fit[cname] = {"sum_share_of_four_truths": round(sum(share.values()), 4),
                      "r2_vs_truth_basis": round(r2, 4)}
    truth_corr = np.corrcoef(basis.T)
    truth_corr_table = {
        f"{TRUTH_NAMES[i]}~{TRUTH_NAMES[j]}": round(float(truth_corr[i, j]), 4)
        for i in range(len(TRUTH_NAMES)) for j in range(i + 1, len(TRUTH_NAMES))
    }

    # Where does each truth part end up?  Regress every truth stem on all six
    # model stems at once: a high R^2 means the content still exists somewhere
    # in the model output (possibly in the wrong file); a low R^2 means the
    # model simply destroyed / attenuated it.
    six_basis_names = [k for k in SIX_STEMS if k in six]
    recovery: dict[str, dict] = {}
    if six_basis_names:
        six_basis = np.stack([six[k] for k in six_basis_names], axis=1)
        for tname in TRUTH_NAMES:
            sig = truth[tname]
            coeffs, *_ = np.linalg.lstsq(six_basis, sig, rcond=None)
            energy = float(np.dot(sig, sig))
            recon = six_basis @ coeffs
            r2 = 1.0 - float(np.dot(sig - recon, sig - recon)) / energy
            contrib = {k: round(float(coeffs[i] * (six_basis[:, i] @ recon)) / energy, 4)
                       for i, k in enumerate(six_basis_names)}
            recovery[tname] = {
                "r2_explained_by_all_six_stems": round(r2, 4),
                "energy_share_by_model_stem": contrib,
                "own_stem_share": contrib.get(tname),
                "off_own_stem_share": round(
                    sum(v for k, v in contrib.items() if k != tname), 4),
                "own_stem_corr": round(float(np.corrcoef(sig, six[tname])[0, 1]), 4)
                if tname in six else None,
            }

    # ---------------- time alignment (a lag would silently steal SI-SDR) ----
    def best_lag_ms(ref: np.ndarray, est: np.ndarray, max_ms: float = 50.0) -> dict:
        m = int(max_ms * eval_sr / 1000)
        a = ref - ref.mean()
        b = est - est.mean()
        nfft = 1 << int(np.ceil(np.log2(a.size + b.size)))
        corr = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
        corr = np.concatenate([corr[-m:], corr[:m + 1]])
        lags = np.arange(-m, m + 1)
        i = int(np.argmax(corr))
        return {"lag_ms": round(float(lags[i]) / eval_sr * 1000, 3),
                "lag_samples": int(lags[i]),
                "peak_corr": round(float(corr[i] / np.sqrt(np.dot(a, a) * np.dot(b, b))), 4)}

    alignment = {}
    for tname in TRUTH_NAMES:
        if f"6stem:{tname}" in candidates:
            alignment[f"{tname}_vs_own"] = best_lag_ms(truth[tname], candidates[f"6stem:{tname}"])
    for tname in TRUTH_NAMES:
        for other in TRUTH_NAMES:
            if other != tname and f"6stem:{other}" in candidates:
                alignment[f"{tname}_vs_6stem:{other}"] = best_lag_ms(
                    truth[tname], candidates[f"6stem:{other}"])

    empty_detail = {
        k: {"max_abs": float(np.max(np.abs(v))), "rms_dbfs": rms_dbfs(v),
            "nonzero_samples": int(np.count_nonzero(v)), "samples": int(v.size)}
        for k, v in six.items() if rms_dbfs(v) < -60
    }

    # ---------------- spectral centroid (register sanity) -------------------
    def centroid(sig: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(sig.size, 1 / eval_sr)
        denom = float(spec.sum())
        return round(float((spec * freqs).sum() / denom), 1) if denom else 0.0

    centroids = {f"truth:{k}": centroid(v) for k, v in truth.items()}
    centroids.update({f"6stem:{k}": centroid(v) for k, v in six.items()})
    centroids["baseline:mix"] = centroid(mix)

    # ---------------- reference-free residual over the six stems ------------
    residual = None
    if six:
        proc = subprocess.run(
            [sys.executable, str(EVAL_DIR / "stem_eval.py"),
             "--mix", str(mix_eval_path), "--stems", str(stems6_dir)],
            capture_output=True, text=True)
        if proc.returncode == 0:
            residual = json.loads(proc.stdout)
            (HERE / "score_six_reference_free.json").write_text(
                json.dumps(residual, indent=2), encoding="utf-8")
            print("[harness] reference-free six stems: residual "
                  f"{residual['reconstruction']['residual_db_rel_mix']} dB rel mix",
                  flush=True)
        else:
            print(f"[harness] reference-free failed: {proc.stderr.strip()}", flush=True)

    confusion = {
        "note": ("rows = ground-truth part, columns = a single model output stem; "
                 "cell = SI-SDR(truth, estimate) in dB, computed with "
                 "stem_eval.si_sdr_db.  'baseline:mix' is the unseparated mix, i.e. "
                 "what you get by claiming every stem is the full mix."),
        "model_outputs": {k: str(v) for k, v in {**six_raw, **two_raw}.items()},
        "matrix": matrix,
        "best_match": best,
        "leakage_shares": shares,
        "leakage_fit": fit,
        "truth_recovery": recovery,
        "truth_stem_correlation": truth_corr_table,
        "truth_gram_condition_number": round(cond, 2),
        "alignment": alignment,
        "empty_stems": empty_detail,
        "spectral_centroid_hz": centroids,
        "levels": levels,
        "harness_six": score_six,
        "harness_two": score_two,
        "harness_six_reference_free": residual,
        "manifest": {"mix": manifest["mix"], "duration_sec": manifest["duration_sec"]},
    }
    (HERE / "confusion.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")

    # ---------------- stdout summary ---------------------------------------
    order = list(candidates)
    w = max(len(k) for k in order) + 2
    print("\n[matrix] SI-SDR dB, rows = truth, cols = model stem")
    print(" " * 9 + "".join(f"{k:>{w}}" for k in order))
    for tname in TRUTH_NAMES:
        print(f"{tname:9s}" + "".join(
            f"{(matrix[tname][k] if matrix[tname][k] is not None else float('nan')):>{w}.2f}"
            for k in order))
    print("\n[levels] RMS dBFS")
    for group in ("truth", "six_stem", "two_stem"):
        print(f"  {group:9s} " + "  ".join(
            f"{k}={levels[group][k]['rms_dbfs']:.2f}" for k in levels[group]))
    print(f"  mix       rms={levels['mix']['rms_dbfs']:.2f} "
          f"peak={levels['mix']['peak_dbfs']:.2f}")

    print("\n[leakage] share of each model stem's energy explained by each truth part")
    print(" " * 18 + "".join(f"{t:>9}" for t in TRUTH_NAMES) + "      r2")
    for cname, share in shares.items():
        print(f"{cname:18s}" + "".join(f"{share[t]:>9.3f}" for t in TRUTH_NAMES)
              + f"  {fit[cname]['r2_vs_truth_basis']:>7.3f}")

    print("\n[centroid] spectral centroid Hz")
    print("  " + "  ".join(f"{k}={v:.0f}" for k, v in centroids.items()))

    print("\n[alignment] lag of model stem vs truth (0 ms == no offset)")
    for k, v in alignment.items():
        print(f"  {k:26s} lag {v['lag_ms']:>7.3f} ms  corr {v['peak_corr']:>6.3f}")

    print(f"\n[empty] {json.dumps(empty_detail)}")
    print(f"[truth corr] {json.dumps(truth_corr_table)} "
          f"cond={cond:.1f}")

    print("\n[recovery] share of each TRUTH part's energy found in the six stems")
    print(" " * 9 + "".join(f"{k:>9}" for k in six_basis_names)
          + "      R2   own  off-own corr")
    for tname in TRUTH_NAMES:
        rec = recovery.get(tname)
        if not rec:
            continue
        print(f"{tname:9s}" + "".join(
            f"{rec['energy_share_by_model_stem'][k]:>9.3f}" for k in six_basis_names)
            + f"  {rec['r2_explained_by_all_six_stems']:>6.3f} "
              f"{rec['own_stem_share']:>6.3f} {rec['off_own_stem_share']:>6.3f} "
              f"{rec['own_stem_corr']:>6.3f}")

    print("\n[best] " + json.dumps(best, indent=2))
    print(f"\n[done] wrote {HERE / 'confusion.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
