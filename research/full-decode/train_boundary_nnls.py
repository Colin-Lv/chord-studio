"""Train the boundary-candidate classifier on the synthetic corpus, test cross-domain on 虚拟.

The features come from official NNLS note evidence only, so synthetic renders and the real
recording are described by the same transform. The classifier is used by full_decode.py to
pick which beat/half-bar/bar candidates are real chord changes.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"c:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import boundary_lib as B  # noqa: E402
import decode_lib as D  # noqa: E402
import harmonic_note_features as H  # noqa: E402

H.sha256_file = functools.lru_cache(maxsize=None)(H.sha256_file)

DATA = Path(r"D:\ChordStudio\research\boundary-dataset")
CACHE = Path(r"D:\ChordStudio\research\train-v2\data\nnls_cache")
SONIC = r"D:\ChordStudio\Dialogue\tools\sonic-annotator-1.7-win64\sonic-annotator-win64\sonic-annotator.exe"
PLUGIN = r"C:\Program Files\Vamp Plugins\nnls-chroma.dll"
TOL = 0.06
OUT = Path(__file__).resolve().parent


def sample_rows(meta):
    labels = json.loads((DATA / meta["labels"]).read_text(encoding="utf-8"))
    bpm = float(labels["bpm"])
    beat = 60.0 / bpm
    n_beats = int(labels["beats"])
    beats = [i * beat for i in range(n_beats)]
    bounds = np.array(labels["boundaries_sec"], dtype=float)
    ev = H.extract_note_evidence(DATA / meta["audio"], cache_dir=CACHE,
                                 sonic_annotator=SONIC, plugin_path=PLUGIN)
    cand = B.beat_candidates(beats)
    X = B.candidate_features(ev.activations, ev.times, cand, beat)
    y = np.array([int(np.min(np.abs(bounds - t)) <= TOL) if len(bounds) else 0 for t, _ in cand])
    return X, y, meta["source_song"]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    samples = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))["samples"]
    print(f"{len(samples)} samples", flush=True)
    Xs, ys, groups = [], [], []
    for i, meta in enumerate(samples):
        X, y, song = sample_rows(meta)
        Xs.append(X)
        ys.append(y)
        groups.extend([song] * len(y))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)} samples, {len(groups)} candidates", flush=True)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    groups = np.array(groups)
    print(f"candidates {len(y)}  positives {int(y.sum())} ({y.mean():.3f})")

    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_depth=3,
                                           min_samples_leaf=10, l2_regularization=1.0)
    model.fit(X, y)
    import joblib
    joblib.dump(model, OUT / "boundary_nnls.joblib")
    (OUT / "boundary_features.json").write_text(json.dumps(B.FEATURES, indent=1), encoding="utf-8")

    # held-out-by-song check
    songs = sorted(set(groups.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    folds = np.array_split(np.array(songs), 5)
    aucs = []
    for held in folds:
        test = np.isin(groups, held)
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_depth=3,
                                           min_samples_leaf=10, l2_regularization=1.0)
        m.fit(X[~test], y[~test])
        p = m.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], p))
    print(f"synthetic 5-fold (by song) AUC: {np.mean(aucs):.4f}  folds {[round(a, 3) for a in aucs]}")

    # cross-domain on 虚拟
    run = D.latest_xuni_run()
    art = D.load_run(run)
    bars = D.phase_corrected_bars(art["beats"], art["beat_pos"], art["chroma"])
    beat = float(np.median(np.diff(art["beats"])))
    cand = B.bar_candidates(bars)
    Xv = B.candidate_features(art["act"], art["note_times"], cand, beat)
    ref = D.load_lab(D.GOLDEN)
    truth = [a for a, _, _ in ref[1:]]
    yv = np.array([int(min(np.abs(np.array(truth) - t)) <= TOL) for t, _ in cand])
    pv = model.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(yv, pv)
    precision, recall, thresholds = precision_recall_curve(yv, pv)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(np.argmax(f1))
    print(f"虚拟 cross-domain: candidates {len(yv)} positives {int(yv.sum())} AUC {auc:.3f} "
          f"best F1 {f1[best]:.3f} (P {precision[best]:.2f} R {recall[best]:.2f} "
          f"@ thr {thresholds[min(best, len(thresholds) - 1)]:.2f})")


if __name__ == "__main__":
    main()
