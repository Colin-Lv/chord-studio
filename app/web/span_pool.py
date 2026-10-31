"""Bar-level pooling of a chord timeline (derived source, read-only).

Two fixes on top of the shipped timeline, neither touching the model:
  1. pool each bar to its duration-weighted majority label and merge neighbours
     (the model decodes 0.25 s cells independently, so it jitters inside a bar);
  2. choose the bar *phase* per bar with a DP over chroma novelty, because madmom's
     downbeat can drift by half a bar for long stretches (虚拟: 83-147 s).

Measured on 虚拟 against the user-confirmed annotation:
  single phase : exact 0.401 -> 0.443, family 0.637 -> 0.648, F1 0.30 -> 0.59, 177 -> 82
  piecewise    : exact 0.444, family 0.690, F1 0.81, 81 segments
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PHASE_PENALTY = 0.05


def read_lab(path: Path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        rows.append((float(parts[0]), float(parts[1]), parts[2].strip()))
    return rows


def pool_segments(segments, bars, half: bool = False, min_span: float = 0.6):
    """Duration-weighted majority label per bar; returns (rows, purity)."""
    out = []
    winners = []
    for i in range(len(bars) - 1):
        a, b = bars[i], bars[i + 1]
        if b - a <= 1e-6:
            continue
        spans = [(a, b)]
        if half:
            mid = (a + b) / 2
            if mid - a >= min_span and b - mid >= min_span:
                spans = [(a, mid), (mid, b)]
        for x, y in spans:
            votes = {}
            for sa, sb, label in segments:
                overlap = min(sb, y) - max(sa, x)
                if overlap > 0:
                    votes[label] = votes.get(label, 0.0) + overlap
            if not votes:
                continue
            label = max(votes, key=votes.get)
            winners.append((votes[label], y - x))
            if out and out[-1][2] == label:
                out[-1] = (out[-1][0], y, label)
            else:
                out.append((x, y, label))
    purity = (sum(w for w, _ in winners) / sum(d for _, d in winners)) if winners else 0.0
    return out, purity


def _novelty(chroma, times, t, half=0.45):
    i = int(np.searchsorted(times, t))
    a = int(np.searchsorted(times, t - half))
    b = int(np.searchsorted(times, t + half))
    if a >= i or i >= b:
        return 0.0
    left = chroma[a:i].mean(axis=0)
    right = chroma[i:b].mean(axis=0)
    denom = (np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9
    return float(1 - (left @ right) / denom)


def estimate_phase(chroma, beat_times, bar_index, penalty=PHASE_PENALTY):
    """DP over bars: phase 0 = given downbeat, phase 1 = shifted by two beats."""
    times = np.arange(chroma.shape[0]) / 10.0
    n0 = np.array([_novelty(chroma, times, beat_times[i]) for i in bar_index])
    n1 = np.array([_novelty(chroma, times, beat_times[min(i + 2, len(beat_times) - 1)]) for i in bar_index])
    dp = np.zeros((len(n0), 2))
    back = np.zeros((len(n0), 2), dtype=int)
    dp[0] = [n0[0], n1[0]]
    for i in range(1, len(n0)):
        for state, values in ((0, n0), (1, n1)):
            stay = dp[i - 1, state]
            switch = dp[i - 1, 1 - state] - penalty
            dp[i, state] = max(stay, switch) + values[i]
            back[i, state] = state if stay >= switch else 1 - state
    state = int(np.argmax(dp[-1]))
    path = [state]
    for i in range(len(n0) - 1, 0, -1):
        state = int(back[i, state])
        path.append(state)
    return path[::-1]


def _bars_from_beats(beats, chroma=None):
    beat_times = np.array([float(b["time"]) for b in beats])
    numbers = [int(b.get("beat", 0)) for b in beats]
    bar_index = [i for i, n in enumerate(numbers) if n == 1]
    if chroma is None or len(bar_index) < 4:
        return [float(beat_times[i]) for i in bar_index]
    path = estimate_phase(chroma, beat_times, bar_index)
    return [float(beat_times[i] if s == 0 else beat_times[min(i + 2, len(beat_times) - 1)])
            for i, s in zip(bar_index, path)]


def pool_lab(input_path: Path, output_path: Path, beats, chroma=None) -> list:
    segments = read_lab(Path(input_path))
    bars = _bars_from_beats(beats, chroma)
    if len(bars) < 2:
        raise ValueError("需要至少两个小节才能做小节池化")
    rows, _purity = pool_segments(segments, bars, half=False)
    Path(output_path).write_text(
        "\n".join(f"{a:.3f} {b:.3f} {label}" for a, b, label in rows), encoding="utf-8")
    return rows
