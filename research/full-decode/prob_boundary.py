"""How much extra boundary evidence does the frozen posterior carry at candidate points?

The NNLS classifier is trained on synthetic audio only, so it cannot see the frozen model's
posterior. At inference that posterior is free evidence: a real chord change should move it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import boundary_lib as B  # noqa: E402
import decode_lib as D  # noqa: E402
import full_decode as F  # noqa: E402


def prob_features(art, times, beat):
    w = 0.5 * beat
    starts, ends = art["starts"], art["ends"]
    out = []
    for t in times:
        lc = [i for i, (x, y) in enumerate(zip(starts, ends)) if x < t and y > t - w]
        rc = [i for i, (x, y) in enumerate(zip(starts, ends)) if x < t + w and y > t]
        if not lc or not rc:
            out.append((0.0, 0.0, 0.0, 0.0))
            continue
        l = np.maximum(art["prob"][lc], 1e-9).mean(axis=0)
        r = np.maximum(art["prob"][rc], 1e-9).mean(axis=0)
        l /= l.sum()
        r /= r.sum()
        cos = float(1 - (l @ r) / ((np.linalg.norm(l) * np.linalg.norm(r)) + 1e-12))
        change = float(l.argmax() != r.argmax())
        ent = float(-(l * np.log(l)).sum() - (r * np.log(r)).sum()) / 2
        out.append((cos, change, ent, float(1 - (l.max() + r.max()) / 2)))
    return np.asarray(out)


def main():
    art = F._prepare(D.load_run(D.latest_xuni_run()))
    model = F.load_model()
    ref = D.load_lab(D.GOLDEN)
    truth = [a for a, _, _ in ref[1:]]
    sc = F.score_candidates(art, model)
    times = sc["times"]
    Xp = prob_features(art, times, sc["beat"])
    y = np.array([int(min(np.abs(np.array(truth) - t)) <= D.TOL) for t in times])
    from sklearn.metrics import roc_auc_score

    names = ["prob_cos", "prob_label_change", "prob_entropy", "prob_flatness"]
    print(f"candidates {len(y)} positives {int(y.sum())}")
    for i, name in enumerate(names):
        print(f"  {name:20s} AUC {roc_auc_score(y, Xp[:, i]):.3f}")
    print(f"  {'nnls logit':20s} AUC {roc_auc_score(y, sc['logit']):.3f}")
    for g in (0.5, 1.0, 2.0, 3.0, 5.0):
        z = Xp[:, 0] / (Xp[:, 0].std() + 1e-9)
        comb = sc["logit"] + g * z
        print(f"  logit + {g:.1f}*z(prob_cos)  AUC {roc_auc_score(y, comb):.3f}")
    np.savez(HERE / "cand_features.npz", times=times, y=y, prob_feats=Xp, logit=sc["logit"],
             prob=sc["prob"])


if __name__ == "__main__":
    main()
