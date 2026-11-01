"""Full-segment chord decoder (second-tier, switchable output).

This is a *derived* timeline: it never touches the frozen model, the weights, the acoustic
frontend or the per-cell posteriors. It re-decodes the sequence the model already produced.

Inputs (all already in the run/cache):
    posterior.npz   chord (cells, 120) posterior
    cells.json      cell start/end
    deep.npz        DeepChroma, used by the bar-phase DP
    notes.npz       official NNLS-Chroma note evidence, 84 x T
    beats.json      madmom beats with a bar position

Decoder
    1. bar grid: madmom downbeats + the per-bar 2-beat phase DP of span_pool;
    2. boundary candidates: every bar line and its half (both interior == half bars);
    3. boundary evidence: GBDT (trained on the synthetic corpus, NNLS note features only)
       -> logit per candidate;
    4. semi-Markov decode: half-bar spans, a log-normal duration prior per segment
       (centre 8 beats, sigma 0.3), a switch cost and the classifier logit as a per-boundary
       discount/bonus;
    5. per segment: pool the posterior, the NNLS bass profile and the sustained-tone
       support, then decode with head structure
           root    = argmax log p_root + lambda_bass * log bass
           quality = argmax log p(q|root) + mu * sum_tone log(support / tau)
    6. merge equal neighbours.

Measured on 虚拟 against the accepted annotation: exact 0.682, family 0.758,
boundary P 0.903 / R 0.918 / F1 0.911, 63 segments (truth 62).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
             "min9": (0, 3, 7, 10, 14), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}
MIDI = np.arange(21, 105)
REGISTERS = ((21, 48), (48, 72), (72, 105))
FEATURES = ["nnls_bass_move", "nnls_bass_root_changed", "nnls_bass_flux",
            "nnls_mid_move", "nnls_high_move", "nnls_mid_flux",
            "nnls_tone_jaccard", "nnls_low_ratio", "nnls_total_ratio",
            "level_bar", "level_half", "level_beat", "level_eighth",
            "nnls_bass_move_wide", "nnls_mid_move_wide",
            "nnls_bass_sustain_left", "nnls_bass_sustain_right", "nnls_tone_hamming"]
TONE_THRESHOLD = 0.02
PARAMS = dict(lambda_switch=18.0, beta_boundary=8.0, lambda_bass=1.5, mu=0.7, tau=0.28,
              dur_center_beats=8.0, dur_sigma=0.30, bass_mode='root')


def read_lab(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right, label = line.split(maxsplit=2)
        rows.append((float(left), float(right), label.strip()))
    return rows


def write_lab(path, rows):
    Path(path).write_text("\n".join(f"{a:.3f} {b:.3f} {label}" for a, b, label in rows),
                          encoding="utf-8")


# --------------------------------------------------------------------------- note evidence
def register_profiles(act):
    norm = act / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    prof = np.zeros((len(REGISTERS), 12, act.shape[1]), dtype=np.float32)
    for r, (lo, hi) in enumerate(REGISTERS):
        mask = (MIDI >= lo) & (MIDI < hi)
        np.add.at(prof[r], MIDI[mask] % 12, norm[mask])
    return prof, norm


def _window(prof, times, a, b):
    sel = (times >= a) & (times < b)
    if not sel.any():
        idx = int(np.clip(np.searchsorted(times, (a + b) / 2), 0, len(times) - 1))
        return prof[:, idx]
    return prof[:, sel].mean(axis=1)


def _cos(x, y):
    return float(1 - (x @ y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-9))


def _flux(prof, times, t):
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    return float(np.clip(prof[:, min(len(times) - 1, i)] - prof[:, max(0, i - 1)], 0, None).sum())


def tone_support(act):
    """(12, T) sustained-tone support in 0..2 (mid + high register presence fractions)."""
    norm = act / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    out = np.zeros((12, act.shape[1]), dtype=np.float32)
    for lo, hi in ((48, 72), (72, 105)):
        mask = (MIDI >= lo) & (MIDI < hi)
        prof = np.zeros((12, act.shape[1]), dtype=np.float32)
        np.add.at(prof, MIDI[mask] % 12, norm[mask])
        out += (prof >= TONE_THRESHOLD).astype(np.float32)
    return out


def span_tone(support, times, a, b):
    sel = (times >= a) & (times < b)
    if not sel.any():
        idx = int(np.clip(np.searchsorted(times, (a + b) / 2), 0, len(times) - 1))
        return support[:, idx]
    return support[:, sel].mean(axis=1)


def span_bass(act, times, a, b, lo=21, hi=48):
    mask = (MIDI >= lo) & (MIDI < hi)
    norm = act[mask] / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    prof = np.zeros((12, len(times)), dtype=np.float32)
    np.add.at(prof, MIDI[mask] % 12, norm)
    sel = (times >= a) & (times < b)
    if not sel.any():
        return np.zeros(12, dtype=np.float32)
    out = prof[:, sel].mean(axis=1)
    return out / (out.sum() + 1e-9)


def candidate_features(act, times, candidates, beat):
    prof, norm = register_profiles(act)
    low, mid, high = prof[0], prof[1], prof[2]
    w, ww = 0.5 * beat, 1.5 * beat
    rows = []
    for t, level in candidates:
        lb = _window(low, times, t - w, t)
        rb = _window(low, times, t, t + w)
        lm = _window(mid, times, t - w, t)
        rm = _window(mid, times, t, t + w)
        lh = _window(high, times, t - w, t)
        rh = _window(high, times, t, t + w)
        sel_l = (times >= t - w) & (times < t)
        sel_r = (times >= t) & (times < t + w)
        left_note = norm[:, sel_l].mean(axis=1) if sel_l.any() else np.zeros(84, dtype=np.float32)
        right_note = norm[:, sel_r].mean(axis=1) if sel_r.any() else np.zeros(84, dtype=np.float32)
        top_l = set(np.argsort(left_note)[::-1][:6].tolist())
        top_r = set(np.argsort(right_note)[::-1][:6].tolist())
        seg_l = low[:, sel_l] if sel_l.any() else low[:, :1]
        seg_r = low[:, sel_r] if sel_r.any() else low[:, :1]
        bl, br = lb.argmax(), rb.argmax()
        rows.append((
            _cos(lb, rb), int(bl != br), _flux(low, times, t),
            _cos(lm, rm), _cos(lh, rh), _flux(mid, times, t),
            float(1 - len(top_l & top_r) / max(1, len(top_l | top_r))),
            float(np.clip((rb.sum() + 1e-9) / (lb.sum() + 1e-9), 0.1, 10.0)),
            float(np.clip((rm.sum() + rh.sum() + 1e-9) / (lm.sum() + lh.sum() + 1e-9), 0.1, 10.0)),
            float(level == "bar"), float(level == "half"),
            float(level == "beat"), float(level == "eighth"),
            _cos(_window(low, times, t - ww, t), _window(low, times, t, t + ww)),
            _cos(_window(mid, times, t - ww, t), _window(mid, times, t, t + ww)),
            float((seg_l.argmax(axis=0) == bl).mean()), float((seg_r.argmax(axis=0) == br).mean()),
            float(len(top_l ^ top_r) / 6.0),
        ))
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------- labels
def tone_log_score(quality, support, root, tau):
    return float(sum(np.log((support[(root + iv) % 12] + 0.02) / tau)
                     for iv in INTERVALS[quality]))


def bass_log_scores(bass):
    """Bass supports chord members, including inversions; it does not define root.

    A floor bounds influence from unreliable mixed-audio low registers. No key,
    song identity or expected progression enters this likelihood.
    """
    bass = np.asarray(bass, dtype=float)
    out = np.zeros((12, 10))
    for root in range(12):
        for q, quality in enumerate(QUALITIES):
            pcs = [(root + iv) % 12 for iv in INTERVALS[quality]]
            support = .5 * bass[root] + .5 * bass[pcs].sum()
            out[root, q] = np.log(.15 + .85 * support)
    return out


def pooled_label(prob, bass_cells, support, cells, lam=1.5, mu=0.6, tau=0.28,
                 bass_mode='root'):
    p = np.maximum(prob[cells], 1e-9).mean(axis=0)
    b = bass_cells[cells].mean(axis=0)
    if bass_mode == 'compatible':
        adjusted = p.reshape(12, 10) * np.exp(lam * bass_log_scores(b))
        root = int(adjusted.sum(axis=1).argmax())
    else:
        roots = p.reshape(12, 10).sum(axis=1)
        root = int(np.argmax(np.log(roots) + lam * np.log(b + 1e-3)))
    scores = np.log(p.reshape(12, 10)[root] + 1e-9)
    if mu:
        scores = scores + mu * np.array([tone_log_score(q, support, root, tau)
                                         for q in QUALITIES])
    return f"{NAMES[root]}:{QUALITIES[int(scores.argmax())]}"


def bar_candidates(bars):
    out = []
    for a, b in zip(bars, bars[1:]):
        if b - a <= 1e-6:
            continue
        step = (b - a) / 4.0
        out.extend([(float(a), "bar"), (float(a + step), "beat"),
                    (float(a + 2 * step), "half"), (float(a + 3 * step), "beat")])
        out.extend((float(a + (k + 0.5) * step), "eighth") for k in range(4))
    out.append((float(bars[-1]), "bar"))
    seen, unique = set(), []
    for t, level in out:
        key = round(t, 3)
        if key not in seen:
            seen.add(key)
            unique.append((t, level))
    return unique


def _cells_between(starts, ends, a, b):
    return [i for i, (x, y) in enumerate(zip(starts, ends)) if x < b and y > a]


def decode(posterior, starts, ends, chroma, act, note_times, beats, beat_pos,
           params=None):
    """Return [(start, end, label), ...] for the whole recording."""
    try:
        from .span_pool import estimate_phase
    except ImportError:
        from span_pool import estimate_phase

    p = dict(PARAMS)
    p.update(params or {})
    beat = float(np.median(np.diff(beats))) if len(beats) > 2 else 0.5
    bar_index = [i for i, n in enumerate(beat_pos) if n == 1]
    if len(bar_index) < 2:
        bar_index = list(range(0, len(beats), 4))
    if chroma is not None and len(bar_index) >= 4:
        path = estimate_phase(chroma, beats, bar_index)
        lines = [float(beats[i] if s == 0 else beats[min(i + 2, len(beats) - 1)])
                 for i, s in zip(bar_index, path)]
    else:
        lines = [float(beats[i]) for i in bar_index]
    bars = [lines[0]]
    for t in lines[1:]:
        if t > bars[-1] + 1e-6:
            bars.append(t)
    bars = [t for t in bars if t <= ends[-1] + 1e-6]
    if len(bars) < 3:
        raise ValueError("需要至少三个小节才能做完整段解码")

    cand = bar_candidates(bars)
    times = np.array([t for t, _ in cand], dtype=float)
    model = _load_model()
    X = candidate_features(act, note_times, cand, beat)
    prob = model.predict_proba(X)[:, 1]
    logit = np.log(np.clip(prob, 1e-4, 1 - 1e-4) / (1 - np.clip(prob, 1e-4, 1 - 1e-4)))
    score_at = {round(float(t), 3): float(l) for t, l in zip(times, logit)}

    bass_cells = np.zeros((len(starts), 12), dtype=np.float32)
    for i, (a, b) in enumerate(zip(starts, ends)):
        bass_cells[i] = span_bass(act, note_times, a, b)
    support = tone_support(act)

    spans = []
    for a, b in zip(bars, bars[1:]):
        mid = (a + b) / 2.0
        spans.append((float(a), float(mid)))
        spans.append((float(mid), float(b)))
    n = len(spans)
    span_times = np.array([s[0] for s in spans] + [spans[-1][1]])
    emission = np.zeros((n, 120), dtype=np.float64)
    logp = np.log(np.maximum(posterior, 1e-9))
    for i, (a, b) in enumerate(spans):
        cells = _cells_between(starts, ends, a, b)
        if not cells:
            emission[i] = -1e9
            continue
        emission[i] = logp[cells].sum(axis=0)
        bass = bass_cells[cells].mean(axis=0)
        bass_score = (bass_log_scores(bass).reshape(-1) if p['bass_mode'] == 'compatible'
                      else np.repeat(np.log(bass + 1e-3), 10))
        emission[i] += p["lambda_bass"] * len(cells) * bass_score
        tone = np.zeros((12, 10), dtype=np.float64)
        support_i = span_tone(support, note_times, a, b)
        for r in range(12):
            tone[r] = [tone_log_score(q, support_i, r, p["tau"]) for q in QUALITIES]
        emission[i] += p["mu"] * len(cells) * tone.reshape(-1)

    cum = np.vstack([np.zeros(120), np.cumsum(emission, axis=0)])
    dp = np.full(n + 1, -np.inf)
    dp[0] = 0.0
    prev = np.full(n + 1, -1, dtype=int)
    prev_label = np.zeros(n + 1, dtype=int)
    for j in range(1, n + 1):
        best_i, best_val, best_lab = -1, -np.inf, 0
        for i in range(j):
            seg = cum[j] - cum[i]
            lab = int(seg.argmax())
            dur = (span_times[j] - span_times[i]) / beat
            logp_dur = -0.5 * (np.log(max(dur, 1e-3) / p["dur_center_beats"])
                               / p["dur_sigma"]) ** 2
            bonus = (p["beta_boundary"] * score_at.get(round(float(span_times[i]), 3), 0.0)
                     - p["lambda_switch"]) if i > 0 else 0.0
            val = dp[i] + seg[lab] + logp_dur + bonus
            if val > best_val:
                best_i, best_val, best_lab = i, val, lab
        dp[j] = best_val
        prev[j] = best_i
        prev_label[j] = best_lab
    if not np.isfinite(dp[n]):
        raise ValueError("完整段解码未得到可行路径")

    bounds, labels = [], []
    j = n
    while j > 0:
        i = int(prev[j])
        if i < 0:
            raise ValueError("完整段解码路径不完整")
        bounds.append((span_times[i], span_times[j]))
        labels.append(int(prev_label[j]))
        j = i
    bounds.reverse()
    labels.reverse()

    rows = []
    for (a, b), lab in zip(bounds, labels):
        cells = _cells_between(starts, ends, a, b)
        if not cells:
            continue
        support_seg = span_tone(support, note_times, a, b)
        label = pooled_label(posterior, bass_cells, support_seg, cells,
                             p["lambda_bass"], p["mu"], p["tau"], p['bass_mode'])
        if rows and rows[-1][2] == label and abs(rows[-1][1] - a) < 1e-6:
            rows[-1] = (rows[-1][0], float(b), label)
        else:
            rows.append((float(a), float(b), label))
    return rows


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        import joblib
        if not (MODEL_DIR / "boundary_nnls.joblib").is_file():
            raise FileNotFoundError("完整段解码模型缺失：models/boundary_nnls.joblib")
        _MODEL = joblib.load(MODEL_DIR / "boundary_nnls.joblib")
    return _MODEL


