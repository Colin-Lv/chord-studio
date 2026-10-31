"""Shared loader / decoder primitives for the full-segment decoder.

Everything here runs on artifacts the frozen production pipeline already writes:
  run/retained10/posterior.npz      chord (cells, 120) posterior + N
  cache/cells.json                  cell start/end
  cache/deep.npz                    DeepChroma (10 fps) for the bar-phase DP
  cache/notes.npz                   official NNLS note evidence (84 x T)
  run/beats.json                    madmom beats with a bar position
No new dependency, no new model, no audio re-decode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\ChordStudio")
BACKEND_WEB = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web")
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min", "7", "maj7", "min7", "9", "maj9", "min9", "sus2", "sus4"]
TOL = 0.06
GOLDEN = ROOT / "research/key-v3/chord_seg/reference_golden.lab"

if str(BACKEND_WEB) not in sys.path:
    sys.path.insert(0, str(BACKEND_WEB))


def load_lab(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b, label = line.split(maxsplit=2)
            rows.append((float(a), float(b), label.strip()))
    return rows


def write_lab(path, rows):
    Path(path).write_text("\n".join(f"{a:.3f} {b:.3f} {label}" for a, b, label in rows),
                          encoding="utf-8")


def latest_xuni_run():
    runs = sorted((ROOT / "data/projects/aa5a41797bb343e590abcf4d8b27ae54/runs").iterdir(),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1]


def load_run(run):
    run = Path(run)
    pred = json.loads((run / "retained10/prediction.json").read_text(encoding="utf-8"))
    cache = Path(pred["cache_directory"])
    post = np.load(run / "retained10/posterior.npz")
    cells = json.loads((cache / "cells.json").read_text(encoding="utf-8"))
    deep = np.load(cache / "deep.npz")
    notes = np.load(cache / "notes.npz")
    beats = json.loads((run / "beats.json").read_text(encoding="utf-8"))
    return dict(
        run=run, cache=cache,
        prob=post["chord"], n=post["N"],
        starts=np.array([c["start"] for c in cells], dtype=float),
        ends=np.array([c["end"] for c in cells], dtype=float),
        chroma=deep["chroma"], chroma_fps=float(deep["fps"]),
        act=notes["activations"], note_times=notes["times"],
        beats=np.array([float(b["time"]) for b in beats], dtype=float),
        beat_pos=np.array([int(b.get("beat", 0)) for b in beats], dtype=int),
        duration=float(pred["audio_seconds"]),
    )


def phase_corrected_bars(beats, beat_pos, chroma, penalty=0.05):
    """Bar lines from madmom downbeats, with the per-bar 2-beat phase DP of span_pool."""
    from span_pool import estimate_phase

    bar_index = [i for i, n in enumerate(beat_pos) if n == 1]
    if len(bar_index) < 2:
        bar_index = list(range(0, len(beats), 4))
    if chroma is None or len(bar_index) < 4:
        return np.array([beats[i] for i in bar_index], dtype=float)
    path = estimate_phase(chroma, beats, bar_index, penalty=penalty)
    lines = [float(beats[i] if s == 0 else beats[min(i + 2, len(beats) - 1)])
             for i, s in zip(bar_index, path)]
    keep = [lines[0]]
    for t in lines[1:]:
        if t > keep[-1] + 1e-6:
            keep.append(t)
    return np.array(keep, dtype=float)


def cell_bass(act, note_times, starts, ends, lo=21, hi=48):
    """(cells, 12) low-register pc profile per cell, L1 normalised per cell."""
    midi = np.arange(21, 105)
    mask = (midi >= lo) & (midi < hi)
    norm = act[mask] / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    pc = midi[mask] % 12
    prof = np.zeros((12, len(note_times)), dtype=np.float32)
    np.add.at(prof, pc, norm)
    out = np.zeros((len(starts), 12), dtype=np.float32)
    for i, (a, b) in enumerate(zip(starts, ends)):
        sel = (note_times >= a) & (note_times < b)
        if sel.any():
            out[i] = prof[:, sel].mean(axis=1)
    out /= (out.sum(axis=1, keepdims=True) + 1e-9)
    return out


def span_bass(act, note_times, a, b, lo=21, hi=48):
    midi = np.arange(21, 105)
    mask = (midi >= lo) & (midi < hi)
    norm = act[mask] / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    pc = midi[mask] % 12
    prof = np.zeros((12, len(note_times)), dtype=np.float32)
    np.add.at(prof, pc, norm)
    sel = (note_times >= a) & (note_times < b)
    if not sel.any():
        return np.zeros(12, dtype=np.float32)
    out = prof[:, sel].mean(axis=1)
    return out / (out.sum() + 1e-9)


INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
             "min9": (0, 3, 7, 10, 14), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}
TONE_THRESHOLD = 0.02


def tone_support_matrix(act, lo_mid=48, hi_mid=72, lo_high=72, hi_high=105,
                        threshold=TONE_THRESHOLD):
    """(12, T) sustained-tone support in 0..2 (mid register + high register fractions).

    Matches tone_support.py exactly: global L1 normalisation per frame, per-register pc
    profile, then the fraction of frames above the absolute threshold, summed over the two
    upper registers.
    """
    midi = np.arange(21, 105)
    norm = act / np.maximum(act.sum(axis=0, keepdims=True), 1e-9)
    out = np.zeros((12, act.shape[1]), dtype=np.float32)
    for lo, hi in ((lo_mid, hi_mid), (lo_high, hi_high)):
        mask = (midi >= lo) & (midi < hi)
        pc = midi[mask] % 12
        prof = np.zeros((12, act.shape[1]), dtype=np.float32)
        np.add.at(prof, pc, norm[mask])
        out += (prof >= threshold).astype(np.float32)
    return out


def span_tone(support, note_times, a, b):
    sel = (note_times >= a) & (note_times < b)
    if not sel.any():
        idx = int(np.clip(np.searchsorted(note_times, (a + b) / 2), 0, len(note_times) - 1))
        return support[:, idx]
    return support[:, sel].mean(axis=1)


def tone_log_score(quality, support, root, tau=0.3, eps=0.02):
    return float(sum(np.log((support[(root + iv) % 12] + eps) / tau)
                     for iv in INTERVALS[quality]))


def pooled_label_plus(prob, bass, support, cells, lam=1.5, mu=0.7, tau=0.3):
    """Root from log p_root + lam*log bass; quality from log p(q|root) + mu * tone log-score."""
    p = np.maximum(prob[cells], 1e-9).mean(axis=0)
    b = bass[cells].mean(axis=0)
    roots = p.reshape(12, 10).sum(axis=1)
    root = int(np.argmax(np.log(roots) + lam * np.log(b + 1e-3)))
    scores = np.log(p.reshape(12, 10)[root] + 1e-9)
    if mu:
        scores = scores + mu * np.array([tone_log_score(q, support, root, tau) for q in QUALITIES])
    quality = int(scores.argmax())
    return f"{NAMES[root]}:{QUALITIES[quality]}"



def head_label(prob_row, bass_row, lam=1.5, quality_only_prob=None):
    """Head structure: root from log p_root + lam*log bass, quality from p(q|root)."""
    p = np.maximum(prob_row, 1e-9)
    roots = p.reshape(12, 10).sum(axis=1)
    score = np.log(roots) + lam * np.log(bass_row + 1e-3)
    root = int(score.argmax())
    quality = int(p.reshape(12, 10)[root].argmax())
    return f"{NAMES[root]}:{QUALITIES[quality]}", root, quality


def pooled_label(prob, bass, cells, lam=1.5):
    p = np.maximum(prob[cells], 1e-9).mean(axis=0)
    b = bass[cells].mean(axis=0)
    return head_label(p, b, lam)[0]


def merge_rows(rows):
    out = []
    for a, b, label in rows:
        if out and out[-1][2] == label and abs(out[-1][1] - a) < 1e-6:
            out[-1] = (out[-1][0], float(b), label)
        else:
            out.append((float(a), float(b), label))
    return out


def frame_acc(pred, ref, step=0.02):
    end = min(pred[-1][1], ref[-1][1])
    grid = np.arange(ref[0][0], end, step)

    def at(table, t):
        for row in table:
            if row[0] <= t < row[1]:
                return row[2]
        return "N"

    exact = family = 0
    for t in grid:
        p, q = at(pred, t), at(ref, t)
        exact += p == q
        pr, _, pq = p.partition(":")
        qr, _, qq = q.partition(":")
        family += (pr == qr) and (pq[:3] == qq[:3])
    return exact / len(grid), family / len(grid)


def boundary_f1(pred, truth, tol=TOL):
    pb = [a for a, _, _ in pred[1:]]
    tb = [a for a, _, _ in truth[1:]]
    hp = sum(1 for p in pb if any(abs(p - t) <= tol for t in tb))
    ht = sum(1 for t in tb if any(abs(p - t) <= tol for p in pb))
    prec, rec = hp / max(1, len(pb)), ht / max(1, len(tb))
    return prec, rec, 2 * prec * rec / max(1e-9, prec + rec)


def score(pred, ref):
    ex, fam = frame_acc(pred, ref)
    p, r, f1 = boundary_f1(pred, ref)
    return dict(segments=len(pred), exact=ex, family=fam, precision=p, recall=r, f1=f1)
