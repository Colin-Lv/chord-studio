"""Tone support v2: per-pc sustained presence and longest continuous run inside a segment.

A chord tone is present for nearly the whole segment; a melody note or a passing tone is
intermittent. Mean support cannot tell them apart as well as the longest run can.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decode_lib as D  # noqa: E402
import head_eval as HE  # noqa: E402

MIDI = np.arange(21, 105)
INTERVALS = HE.INTERVALS


def presence(act, lo=48, hi=105, rel=0.5):
    mask = (MIDI >= lo) & (MIDI < hi)
    pc = MIDI[mask] % 12
    reg = act[mask]
    prof = np.zeros((12, act.shape[1]), dtype=np.float32)
    np.add.at(prof, pc, reg)
    mx = prof.max(axis=0, keepdims=True)
    return (prof >= rel * np.maximum(mx, 1e-9)) & (mx > 1e-6)


def longest_run_frac(pres):
    out = np.zeros(12, dtype=np.float32)
    for p in range(12):
        best = cur = 0
        for v in pres[p]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        out[p] = best / max(1, pres.shape[1])
    return out


def segment_tone(act, times, a, b, lo=48, hi=105, rel=0.5):
    sel = (times >= a) & (times < b)
    if not sel.any():
        sel = np.zeros(len(times), dtype=bool)
        sel[int(np.clip(np.searchsorted(times, (a + b) / 2), 0, len(times) - 1))] = True
    pres = presence(act[:, sel], lo, hi, rel)
    sup = pres.mean(axis=1)
    run = longest_run_frac(pres)
    return sup, run


def main():
    run = D.latest_xuni_run()
    art = D.load_run(run)
    ref = D.load_lab(D.GOLDEN)
    rows = []
    for a, b, label in ref:
        r, pcs = HE.label_pcs(label)
        if pcs is None:
            continue
        cells = [i for i, (x, y) in enumerate(zip(art["starts"], art["ends"])) if x < b and y > a]
        p = np.maximum(art["prob"][cells], 1e-9).mean(axis=0).reshape(12, 10)
        bass = D.span_bass(art["act"], art["note_times"], a, b)
        sup, run_frac = segment_tone(art["act"], art["note_times"], a, b)
        rows.append((label, r, p, bass, sup, run_frac, a))
    truth = [HE.canon(r[0]) for r in rows]
    roots_true = np.array([D.NAMES.index(t.split(":")[0]) for t in truth])
    p = np.stack([r[2] for r in rows])
    bass = np.stack([r[3] for r in rows])
    sup = np.stack([r[4] for r in rows])
    runf = np.stack([r[5] for r in rows])
    n = len(rows)
    print(f"{n} segments")

    best = None
    for lam in (1.0, 1.5, 2.0):
        for mu in (0.2, 0.35, 0.5, 0.7):
            for tau in (0.2, 0.3, 0.4):
                exact = fam = root_hit = 0
                preds = []
                for i in range(n):
                    pr = p[i]
                    score = np.log(pr.sum(axis=1) + 1e-9) + lam * np.log(bass[i] + 1e-3)
                    r = int(score.argmax())
                    qs = np.log(pr[r] + 1e-9)
                    qs = qs + mu * np.array([
                        sum(np.log((max(sup[i][(r + iv) % 12], runf[i][(r + iv) % 12]) + 0.02) / tau)
                            for iv in INTERVALS[q]) for q in D.QUALITIES])
                    q = int(qs.argmax())
                    pred = f"{D.NAMES[r]}:{D.QUALITIES[q]}"
                    preds.append(pred)
                    exact += pred == truth[i]
                    root_hit += r == roots_true[i]
                    fam += (pred.split(":")[0] == truth[i].split(":")[0]
                            and pred.split(":")[1][:3] == truth[i].split(":")[1][:3])
                tag = f"lam {lam} mu {mu} tau {tau}"
                if best is None or exact > best[0]:
                    best = (exact, tag, preds, (lam, mu, tau))
                    print(f"{tag}: exact {exact / n:.3f} family {fam / n:.3f} "
                          f"root {root_hit / n:.3f}  <- best")
    print("best", best[1], best[0], "/", n)
    print("params", best[3])


if __name__ == "__main__":
    main()
