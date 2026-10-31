"""Root / quality re-scoring on the pooled golden segments: how far does head structure go?

Sweeps the bass weight and an optional sustained-tone term, reports exact/family per
configuration together with the label confusion it produces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decode_lib as D  # noqa: E402

INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
             "min9": (0, 3, 7, 10, 14), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}
FLATS = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}


def canon(label):
    root, _, q = label.partition(":")
    if root in FLATS:
        root = D.NAMES[FLATS[root]]
    return f"{root}:{q}"


def label_pcs(label):
    root, _, q = label.partition(":")
    if root in FLATS:
        r = FLATS[root]
    elif root in D.NAMES:
        r = D.NAMES.index(root)
    else:
        return None, None
    if q in INTERVALS:
        return r, {((r + iv) % 12) for iv in INTERVALS[q]}
    if q.startswith("add9"):
        return r, {r % 12, (r + 4) % 12, (r + 7) % 12, (r + 2) % 12}
    return None, None


def tone_score(q, sup, root):
    pcs = {(root + iv) % 12 for iv in INTERVALS[q]}
    inside = float(np.mean([sup[i] for i in pcs]))
    outside = float(np.mean([sup[i] for i in range(12) if i not in pcs]))
    return inside - outside


def tone_log_score(q, sup, root, tau=0.2, eps=0.02):
    """Sum log(support/tau) over chord tones: every tone must be present, tau sets the bar."""
    pcs = {(root + iv) % 12 for iv in INTERVALS[q]}
    return float(sum(np.log((sup[i] + eps) / tau) for i in pcs))


def evaluate(p, bass, sup, truth, roots_true, n, lam, mu, tau, logform):
    exact = fam = root_hit = 0
    preds = []
    for i in range(n):
        pr = p[i]
        roots = pr.sum(axis=1)
        score = np.log(roots + 1e-9) + lam * np.log(bass[i] + 1e-3)
        r = int(score.argmax())
        qs = np.log(pr[r] + 1e-9)
        if mu:
            if logform:
                qs = qs + mu * np.array([tone_log_score(q, sup[i], r, tau) for q in D.QUALITIES])
            else:
                qs = qs + mu * np.array([tone_score(q, sup[i], r) for q in D.QUALITIES])
        q = int(qs.argmax())
        pred = f"{D.NAMES[r]}:{D.QUALITIES[q]}"
        preds.append(pred)
        exact += pred == truth[i]
        root_hit += r == roots_true[i]
        fam += (pred.split(":")[0] == truth[i].split(":")[0]
                and pred.split(":")[1][:3] == truth[i].split(":")[1][:3])
    return exact, fam, root_hit, preds


def main():
    z = np.load(HERE / "golden_pooled.npz", allow_pickle=True)
    labels = [str(x) for x in z["label"]]
    p = z["p"]
    bass = z["bass"]
    sup = z["sup"]
    truth = [canon(x) for x in labels]
    n = len(labels)
    roots_true = np.array([D.NAMES.index(t.split(":")[0]) for t in truth])

    print(f"{n} segments; max achievable exact (labels the 10-class set can express): "
          f"{sum(1 for t in truth if t.split(':')[1] in INTERVALS)}/{n}")

    best = None
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0):
        for mu in (0.0, 0.3, 0.5, 1.0):
            exact, fam, root_hit, preds = evaluate(p, bass, sup, truth, roots_true, n, lam, mu,
                                                   0.2, False)
            tag = f"mean  lam {lam:.1f} mu {mu:.1f}"
            print(f"{tag:22s} exact {exact / n:.3f} family {fam / n:.3f} root {root_hit / n:.3f}")
            if best is None or exact > best[0]:
                best = (exact, tag, preds)
    for lam in (1.0, 1.5, 2.0):
        for mu in (0.2, 0.4, 0.6, 0.8):
            for tau in (0.15, 0.25, 0.35):
                exact, fam, root_hit, preds = evaluate(p, bass, sup, truth, roots_true, n, lam,
                                                       mu, tau, True)
                tag = f"sum  lam {lam:.1f} mu {mu:.1f} tau {tau:.2f}"
                print(f"{tag:30s} exact {exact / n:.3f} family {fam / n:.3f} root {root_hit / n:.3f}")
                if best is None or exact > best[0]:
                    best = (exact, tag, preds)
    print("\nbest:", best[1], f"{best[0]}/{n}")
    for t, pred in zip(truth, best[2]):
        if t != pred:
            print(f"   {t:>9} -> {pred}")


if __name__ == "__main__":
    main()
