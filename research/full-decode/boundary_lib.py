"""Boundary-candidate features from NNLS note evidence (no chroma, no probabilities).

Every feature is computed identically on synthetic renders and on real recordings because
both come from the same Vamp NNLS-Chroma transform; only the note evidence differs.

Features (order matters: trained models store this list):
  nnls_bass_move          cosine distance of the low-register pc profile around t
  nnls_bass_root_changed  argmax of the low-register profile changes at t
  nnls_bass_flux          positive flux of the low-register pc profile at t
  nnls_mid_move           cosine distance of the mid-register profile around t
  nnls_high_move          cosine distance of the high-register profile around t
  nnls_mid_flux           positive flux of the mid-register pc profile at t
  nnls_tone_jaccard       1 - Jaccard of the strongest 6 sustained notes before/after
  nnls_low_ratio          low-register energy after / before
  nnls_total_ratio        total note energy after / before
  level_bar, level_half, level_beat, level_eighth   one-hot grid position
"""
from __future__ import annotations

import numpy as np

MIDI = np.arange(21, 105)
REGISTERS = ((21, 48), (48, 72), (72, 105))
FEATURES = ["nnls_bass_move", "nnls_bass_root_changed", "nnls_bass_flux",
            "nnls_mid_move", "nnls_high_move", "nnls_mid_flux",
            "nnls_tone_jaccard", "nnls_low_ratio", "nnls_total_ratio",
            "level_bar", "level_half", "level_beat", "level_eighth",
            "nnls_bass_move_wide", "nnls_mid_move_wide",
            "nnls_bass_sustain_left", "nnls_bass_sustain_right", "nnls_tone_hamming"]



def register_profiles(act):
    """(3, 12, T) L1-normalised per frame over all 84 notes, then per register."""
    norm = act / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    out = np.zeros((len(REGISTERS), 12, act.shape[1]), dtype=np.float32)
    for r, (lo, hi) in enumerate(REGISTERS):
        mask = (MIDI >= lo) & (MIDI < hi)
        np.add.at(out[r], MIDI[mask] % 12, norm[mask])
    return out, norm


def _window(prof, times, a, b):
    sel = (times >= a) & (times < b)
    if not sel.any():
        idx = int(np.clip(np.searchsorted(times, (a + b) / 2), 0, len(times) - 1))
        return prof[:, idx]
    return prof[:, sel].mean(axis=1)


def _cos(x, y):
    return float(1 - (x @ y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-9))


def _flux(prof, times, t, half):
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    prev = prof[:, max(0, i - 1)]
    nxt = prof[:, min(len(times) - 1, i)]
    return float(np.clip(nxt - prev, 0, None).sum())


def candidate_features(act, times, candidates, beat):
    """candidates: sequence of (time, level). Returns X matching FEATURES."""
    prof, norm = register_profiles(act)
    low, mid, high = prof[0], prof[1], prof[2]
    w = 0.5 * beat
    ww = 1.5 * beat
    rows = []
    for t, level in candidates:
        lb = _window(low, times, t - w, t)
        rb = _window(low, times, t, t + w)
        lm = _window(mid, times, t - w, t)
        rm = _window(mid, times, t, t + w)
        lh = _window(high, times, t - w, t)
        rh = _window(high, times, t, t + w)
        wlb = _window(low, times, t - ww, t)
        wrb = _window(low, times, t, t + ww)
        wlm = _window(mid, times, t - ww, t)
        wrm = _window(mid, times, t, t + ww)
        sel_l = (times >= t - w) & (times < t)
        sel_r = (times >= t) & (times < t + w)
        left_note = norm[:, sel_l].mean(axis=1) if sel_l.any() else np.zeros(84, dtype=np.float32)
        right_note = norm[:, sel_r].mean(axis=1) if sel_r.any() else np.zeros(84, dtype=np.float32)
        order_l = np.argsort(left_note)[::-1][:6]
        order_r = np.argsort(right_note)[::-1][:6]
        top_l, top_r = set(order_l.tolist()), set(order_r.tolist())
        jac = 1 - len(top_l & top_r) / max(1, len(top_l | top_r))
        hamming = len(top_l ^ top_r) / 6.0
        # how consistently the low register points at one pitch class inside each half
        bl = _window(low, times, t - w, t).argmax()
        br = _window(low, times, t, t + w).argmax()
        seg_l = low[:, sel_l] if sel_l.any() else low[:, :1]
        seg_r = low[:, sel_r] if sel_r.any() else low[:, :1]
        sustain_l = float((seg_l.argmax(axis=0) == bl).mean())
        sustain_r = float((seg_r.argmax(axis=0) == br).mean())
        low_ratio = float((rb.sum() + 1e-9) / (lb.sum() + 1e-9))
        total_ratio = float((rm.sum() + rh.sum() + 1e-9) / (lm.sum() + lh.sum() + 1e-9))
        rows.append((
            _cos(lb, rb),
            int(np.argmax(lb) != np.argmax(rb)),
            _flux(low, times, t, w),
            _cos(lm, rm),
            _cos(lh, rh),
            _flux(mid, times, t, w),
            float(jac),
            float(np.clip(low_ratio, 0.1, 10.0)),
            float(np.clip(total_ratio, 0.1, 10.0)),
            float(level == "bar"), float(level == "half"),
            float(level == "beat"), float(level == "eighth"),
            _cos(wlb, wrb),
            _cos(wlm, wrm),
            sustain_l, sustain_r, float(hamming),
        ))
    return np.asarray(rows, dtype=np.float64)



def beat_candidates(beats):
    """Bar / half / beat / eighth candidates on a 4-beat grid."""
    out = []
    for i, t in enumerate(beats):
        pos = i % 4
        level = "bar" if pos == 0 else ("half" if pos == 2 else "beat")
        out.append((float(t), level))
        if i + 1 < len(beats):
            out.append((float((beats[i] + beats[i + 1]) / 2), "eighth"))
    return out


def bar_candidates(bar_lines):
    """Bar / half / beat / eighth candidates from an explicit (phase-corrected) bar list.

    bar_lines must be strictly increasing times of downbeats; the interior is subdivided
    into beats by assuming 4 equal beats per bar.
    """
    out = []
    for a, b in zip(bar_lines, bar_lines[1:]):
        if b - a <= 1e-6:
            continue
        step = (b - a) / 4.0
        out.append((float(a), "bar"))
        out.append((float(a + step), "beat"))
        out.append((float(a + 2 * step), "half"))
        out.append((float(a + 3 * step), "beat"))
        for k in range(4):
            out.append((float(a + (k + 0.5) * step), "eighth"))
    out.append((float(bar_lines[-1]), "bar"))
    seen = set()
    unique = []
    for t, level in out:
        key = round(t, 3)
        if key in seen:
            continue
        seen.add(key)
        unique.append((t, level))
    return unique
