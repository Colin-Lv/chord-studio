"""Full-segment decoder: phase-corrected bar grid + NNLS boundary classifier + head pooling.

This is the candidate second-tier output (`retained10_full`). It only reads artifacts the
frozen pipeline already writes, so it can run inside the backend without new dependencies
beyond numpy/joblib/sklearn (both already in the web environment).

Pipeline
 1. bar grid: madmom downbeats with span_pool's per-bar 2-beat phase DP (chroma novelty);
 2. candidates: every phase-corrected bar line, its half, its interior beats and eighths;
 3. boundary classifier: GBDT on NNLS note-evidence features -> probability per candidate;
 4. two ways to turn scores into spans
      barsplit : pool each bar; split at the half-bar when the classifier fires
      dp       : choose a boundary subset maximising sum(logit - lambda), min span
 5. per span: pool the 120-class posterior, the NNLS bass profile and the sustained-tone
    support, then decode with head structure (root = log p_root + lam_bass*log bass,
    quality = log p(q|root) + mu * tone support);
 6. merge equal neighbours.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app")

import boundary_lib as B  # noqa: E402
import decode_lib as D  # noqa: E402


def load_model(path=None):
    import joblib
    path = Path(path or HERE / "boundary_nnls.joblib")
    return joblib.load(path)


def boundary_dp(times, logit, lam, min_span):
    """Optimal boundary subset: maximise sum(logit - lam) over chosen interior candidates."""
    n = len(times)
    if n == 0:
        return []
    times = np.asarray(times, dtype=float)
    gain = logit - lam
    dp = np.full(n, -np.inf)
    prev = np.full(n, -1, dtype=int)
    dp[0] = 0.0
    best_val = dp[0]
    best_idx = 0
    best_gain_val = dp[0] + gain[0]
    best_gain_idx = 0
    k = 0
    for j in range(1, n):
        while k < j and times[j] - times[k] >= min_span - 1e-9:
            if dp[k] > best_val:
                best_val, best_idx = dp[k], k
            if dp[k] + gain[k] > best_gain_val:
                best_gain_val, best_gain_idx = dp[k] + gain[k], k
            k += 1
        if j == n - 1:
            if best_val > -np.inf:
                dp[j] = best_val
                prev[j] = best_idx
        else:
            if best_gain_val > -np.inf:
                dp[j] = best_gain_val + gain[j]
                prev[j] = best_gain_idx
    if not np.isfinite(dp[-1]):
        return []
    path = []
    j = n - 1
    while j > 0:
        j = int(prev[j])
        if j < 0:
            return []
        if j > 0:
            path.append(float(times[j]))
    return path[::-1]


def _prepare(art):
    if "bass_cells" not in art:
        art["bass_cells"] = D.cell_bass(art["act"], art["note_times"], art["starts"], art["ends"])
    if "tone_support" not in art:
        art["tone_support"] = D.tone_support_matrix(art["act"])
    return art


def _cells(art, a, b):
    return [i for i, (x, y) in enumerate(zip(art["starts"], art["ends"])) if x < b and y > a]


def _span_label(art, a, b, lam_bass=1.5, mu=0.7, tau=0.3):
    cells = _cells(art, a, b)
    if not cells:
        return None, []
    support = D.span_tone(art["tone_support"], art["note_times"], a, b)
    return D.pooled_label_plus(art["prob"], art["bass_cells"], support, cells,
                               lam_bass, mu, tau), cells


def label_loglik(art, label, cells, mu=0.7, tau=0.3, a=None, b=None):
    """Sum of the decoder's own log-score for one label over a cell set (gain test)."""
    root, _, quality = label.partition(":")
    if root not in D.NAMES or quality not in D.QUALITIES:
        return -1e18
    index = D.NAMES.index(root) * 10 + D.QUALITIES.index(quality)
    ll = float(np.log(np.maximum(art["prob"][cells, index], 1e-9)).sum())
    if mu and a is not None:
        support = D.span_tone(art["tone_support"], art["note_times"], a, b)
        ll += mu * len(cells) * D.tone_log_score(quality, support, D.NAMES.index(root), tau)
    return ll


def prune(art, rows, delta=0.0, lam_bass=1.5, mu=0.7, tau=0.3):
    """Drop a boundary unless splitting beats merging by more than `delta` nats."""
    out = []
    for row in rows:
        out.append(row)
        while len(out) >= 2:
            left, right = out[-2], out[-1]
            if left[2] == right[2]:
                out[-2:] = [(left[0], right[1], left[2])]
                continue
            lc, rc = _cells(art, left[0], left[1]), _cells(art, right[0], right[1])
            bc = _cells(art, left[0], right[1])
            if not lc or not rc or not bc:
                out[-2:] = [(left[0], right[1], left[2])]
                continue
            split = (label_loglik(art, left[2], lc, mu, tau, left[0], left[1])
                     + label_loglik(art, right[2], rc, mu, tau, right[0], right[1]))
            merged_label, _ = _span_label(art, left[0], right[1], lam_bass, mu, tau)
            merged = label_loglik(art, merged_label, bc, mu, tau, left[0], right[1])
            if split - merged >= delta:
                break
            out[-2:] = [(left[0], right[1], merged_label)]
    return out


def _assign(art, spans, lam_bass=1.5, mu=0.7, tau=0.3, delta=None):
    rows = []
    for a, b in spans:
        label, cells = _span_label(art, a, b, lam_bass, mu, tau)
        if label is None:
            continue
        rows.append((a, b, label))
    rows = D.merge_rows(rows)
    if delta is not None:
        rows = prune(art, rows, delta, lam_bass, mu, tau)
    return rows



def score_candidates(art, model, bars=None, phase=True):
    beats = art["beats"]
    beat = float(np.median(np.diff(beats)))
    if bars is None:
        bars = (D.phase_corrected_bars(beats, art["beat_pos"], art["chroma"]) if phase
                else np.array([beats[i] for i in range(0, len(beats), 4)], dtype=float))
    bars = bars[bars <= art["ends"][-1] + 1e-6]
    cand = B.bar_candidates(bars)
    times = np.array([t for t, _ in cand], dtype=float)
    X = B.candidate_features(art["act"], art["note_times"], cand, beat)
    p = model.predict_proba(X)[:, 1]
    return dict(bars=bars, times=times, prob=p,
                logit=np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))),
                beat=beat)


def emission_matrix(art, spans, lam_bass=1.5, mu=0.7, tau=0.3):
    """(spans, 120) log score of every label for every span.

    log p(label) summed over the span's cells, plus the bass-supported root term and the
    sustained-tone term, both scaled by the cell count so longer spans are not favoured.
    """
    prob = np.maximum(art["prob"], 1e-9)
    logp = np.log(prob)
    out = np.zeros((len(spans), 120), dtype=np.float64)
    for i, (a, b) in enumerate(spans):
        cells = _cells(art, a, b)
        if not cells:
            out[i] = -1e9
            continue
        out[i] = logp[cells].sum(axis=0)
        bass = art["bass_cells"][cells].mean(axis=0)
        root_term = np.log(bass + 1e-3)
        out[i] += lam_bass * len(cells) * np.repeat(root_term, 10)
        support = D.span_tone(art["tone_support"], art["note_times"], a, b)
        tone = np.zeros((12, 10), dtype=np.float64)
        for r in range(12):
            tone[r] = [D.tone_log_score(q, support, r, tau) for q in D.QUALITIES]
        out[i] += mu * len(cells) * tone.reshape(-1)
    return out


def viterbi_spans(art, spans, emission, edge_scores, kappa=8.0, kappa_root=14.0, beta=0.0):
    """Pick one label per span; switching costs nats, the boundary classifier discounts it."""
    n = len(spans)
    dp = emission[0].copy()
    back = np.zeros((n, 120), dtype=np.int16)
    same = np.arange(120)
    roots = same // 10
    for i in range(1, n):
        switch = kappa - beta * edge_scores[i - 1]
        root_switch = kappa_root - beta * edge_scores[i - 1]
        cost = -np.where(roots[:, None] == roots[None, :], switch, root_switch)
        cost[np.arange(120), np.arange(120)] = 0.0
        total = dp[:, None] + cost
        back[i] = total.argmax(axis=0)
        dp = emission[i] + total.max(axis=0)
    path = [int(dp.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    return path


def decode_viterbi(art, model, kappa=8.0, kappa_root=14.0, beta=0.0, lam_bass=1.5, mu=0.7,
                   tau=0.3, phase=True, relabel=False, gamma=0.0, prune_delta=None):
    art = _prepare(art)
    sc = score_candidates(art, model, phase=phase)
    bars = sc["bars"]
    spans = []
    for a, b in zip(bars, bars[1:]):
        mid = (a + b) / 2.0
        spans.append((float(a), float(mid)))
        spans.append((float(mid), float(b)))
    emission = emission_matrix(art, spans, lam_bass, mu, tau)
    score_at = {round(float(t), 3): float(l) for t, l in zip(sc["times"], sc["logit"])}
    edge = np.zeros(max(0, len(spans) - 1))
    for i in range(1, len(spans)):
        edge[i - 1] = score_at.get(round(float(spans[i][0]), 3), 0.0)
    if gamma:
        beat = sc["beat"]
        w = 0.5 * beat
        cos = np.zeros(len(edge))
        starts, ends = art["starts"], art["ends"]
        for i in range(1, len(spans)):
            t = spans[i][0]
            lc = [k for k, (x, y) in enumerate(zip(starts, ends)) if x < t and y > t - w]
            rc = [k for k, (x, y) in enumerate(zip(starts, ends)) if x < t + w and y > t]
            if lc and rc:
                left = np.maximum(art["prob"][lc], 1e-9).mean(axis=0)
                right = np.maximum(art["prob"][rc], 1e-9).mean(axis=0)
                cos[i - 1] = 1 - (left @ right) / ((np.linalg.norm(left) * np.linalg.norm(right))
                                                   + 1e-12)
        if cos.std() > 1e-9:
            edge = edge + gamma * (cos - cos.mean()) / cos.std()
    path = viterbi_spans(art, spans, emission, edge, kappa, kappa_root, beta)
    rows = []
    for (a, b), state in zip(spans, path):
        rows.append((a, b, f"{D.NAMES[state // 10]}:{D.QUALITIES[state % 10]}"))
    rows = D.merge_rows(rows)
    if relabel:
        out = []
        for a, b, _old in rows:
            label, cells = _span_label(art, a, b, lam_bass, mu, tau)
            if label is not None:
                out.append((a, b, label))
        rows = D.merge_rows(out)
    if prune_delta is not None:
        rows = prune(art, rows, prune_delta, lam_bass, mu, tau)
    return rows


def decode_semimarkov(art, model, lam=18.0, beta=8.0, mu=0.6, tau=0.28, lam_bass=1.5,
                      dur_center=8.0, dur_sigma=0.30, phase=True, relabel=True,
                      gamma=0.0):
    """Semi-Markov decode: a log-normal duration prior per segment plus a switch cost.

    The span set is every half bar. Score of one segment [i, j) =
        sum of the best label's log score over its spans
        + log P(duration in beats)            (log-normal, centre `dur_center`)
        + beta * boundary logit at its start  - lam
    """
    art = _prepare(art)
    sc = score_candidates(art, model, phase=phase)
    bars = sc["bars"]
    spans = []
    for a, b in zip(bars, bars[1:]):
        mid = (a + b) / 2.0
        spans.append((float(a), float(mid)))
        spans.append((float(mid), float(b)))
    beat = sc["beat"]
    emission = emission_matrix(art, spans, lam_bass, mu, tau)
    n = len(spans)
    cum = np.vstack([np.zeros(120), np.cumsum(emission, axis=0)])
    times = np.array([s[0] for s in spans] + [spans[-1][1]])
    score_at = {round(float(t), 3): float(l) for t, l in zip(sc["times"], sc["logit"])}
    edge = np.zeros(n)
    for i in range(n):
        edge[i] = score_at.get(round(float(times[i]), 3), 0.0)
    if gamma:
        w = 0.5 * beat
        cos = np.zeros(n)
        starts, ends = art["starts"], art["ends"]
        for i in range(1, n):
            t = times[i]
            lc = [k for k, (x, y) in enumerate(zip(starts, ends)) if x < t and y > t - w]
            rc = [k for k, (x, y) in enumerate(zip(starts, ends)) if x < t + w and y > t]
            if lc and rc:
                left = np.maximum(art["prob"][lc], 1e-9).mean(axis=0)
                right = np.maximum(art["prob"][rc], 1e-9).mean(axis=0)
                cos[i] = 1 - (left @ right) / ((np.linalg.norm(left) * np.linalg.norm(right))
                                               + 1e-12)
        if cos.std() > 1e-9:
            edge = edge + gamma * (cos - cos.mean()) / cos.std()

    dp = np.full(n + 1, -np.inf)
    dp[0] = 0.0
    prev = np.full(n + 1, -1, dtype=int)
    prev_label = np.zeros(n + 1, dtype=int)
    for j in range(1, n + 1):
        best_i, best_val, best_lab = -1, -np.inf, 0
        for i in range(j):
            seg = cum[j] - cum[i]
            lab = int(seg.argmax())
            dur = (times[j] - times[i]) / beat
            logp = -0.5 * (np.log(max(dur, 1e-3) / dur_center) / dur_sigma) ** 2
            start_bonus = (beta * edge[i] - lam) if i > 0 else 0.0
            val = dp[i] + seg[lab] + logp + start_bonus
            if val > best_val:
                best_i, best_val, best_lab = i, val, lab
        if best_i >= 0:
            dp[j] = best_val
            prev[j] = best_i
            prev_label[j] = best_lab
    if not np.isfinite(dp[n]):
        return []
    bounds, labels = [], []
    j = n
    while j > 0:
        i = int(prev[j])
        labels.append(int(prev_label[j]))
        bounds.append((times[i], times[j]))
        j = i
    bounds.reverse()
    labels.reverse()
    rows = [(a, b, f"{D.NAMES[lab // 10]}:{D.QUALITIES[lab % 10]}")
            for (a, b), lab in zip(bounds, labels)]
    rows = D.merge_rows(rows)
    if relabel:
        out = []
        for a, b, _old in rows:
            label, _cells = _span_label(art, a, b, lam_bass, mu, tau)
            if label is not None:
                out.append((a, b, label))
        rows = D.merge_rows(out)
    return rows


def decode(art, model, mode="barsplit", thr_half=0.5, lam_boundary=-0.5, lam_bass=1.5,
           mu=0.7, tau=0.3, min_span=None, phase=True, delta=None, kappa=8.0,
           kappa_root=14.0, beta=0.0):
    art = _prepare(art)
    sc = score_candidates(art, model, phase=phase)
    bars = sc["bars"]
    if mode == "barsplit":
        score_at = {round(float(t), 3): float(s) for t, s in zip(sc["times"], sc["prob"])}
        spans = []
        for a, b in zip(bars, bars[1:]):
            mid = (a + b) / 2.0
            if score_at.get(round(float(mid), 3), 0.0) >= thr_half and mid - a > 0.2:
                spans.append((float(a), float(mid)))
                spans.append((float(mid), float(b)))
            else:
                spans.append((float(a), float(b)))
        return _assign(art, spans, lam_bass, mu, tau, delta)
    if mode == "dp":
        beat = sc["beat"]
        min_span = (2.0 * beat if min_span is None else min_span) - 1e-6
        chosen = boundary_dp(sc["times"], sc["logit"], lam_boundary, min_span)
        bounds = [float(sc["times"][0])] + chosen + [float(min(sc["times"][-1], art["ends"][-1]))]
        bounds = sorted(set(round(b, 4) for b in bounds))
        spans = list(zip(bounds, bounds[1:]))
        return _assign(art, spans, lam_bass, mu, tau, delta)
    if mode == "viterbi":
        return decode_viterbi(art, model, kappa=kappa, kappa_root=kappa_root, beta=beta,
                              lam_bass=lam_bass, mu=mu, tau=tau, phase=phase)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--mode", default="barsplit", choices=("barsplit", "dp"))
    ap.add_argument("--thr-half", type=float, default=0.5)
    ap.add_argument("--lam-boundary", type=float, default=-0.5)
    ap.add_argument("--lam-bass", type=float, default=1.5)
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--min-span", type=float, default=None)
    ap.add_argument("--no-phase", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    run = Path(args.run) if args.run else D.latest_xuni_run()
    art = D.load_run(run)
    model = load_model(args.model)
    rows = decode(art, model, args.mode, args.thr_half, args.lam_boundary, args.lam_bass,
                  args.mu, args.tau, args.min_span, phase=not args.no_phase)
    ref = D.load_lab(D.GOLDEN)
    m = D.score(rows, ref)
    print(f"run {run.name}  mode {args.mode} thr {args.thr_half} lam {args.lam_boundary} "
          f"lam_bass {args.lam_bass} mu {args.mu} tau {args.tau} phase {not args.no_phase}")
    print(f"  segments {m['segments']:3d} exact {m['exact']:.3f} family {m['family']:.3f} "
          f"P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f}")
    if args.out:
        D.write_lab(args.out, rows)
        print("  wrote", args.out)


if __name__ == "__main__":
    main()


