"""Allowed-first root policy over frozen v1 candidate utility scores.

This changes selection order only. It is not a new feature or semantic model.
Neither this module nor its selection API accepts references or audio.
"""
from __future__ import annotations

import math

import mir_eval
import numpy as np

from evaluate_chords import TimedChord, harte_chord

POLICY_SCHEMA = "raw_root_allowed_first_candidate_policy_v2"
FIXED_MARGIN = 0.02
POLICY_RULE = (
    "Preserve raw N/X exactly. Otherwise keep candidates with a valid chord and "
    "the same root as raw, rank only this allowed set, then replace raw only if "
    "the best clipped v1 utility exceeds raw utility + 0.02 + 1e-12. "
    "Raw wins a tied score; other ties keep the existing candidate order. "
    "Original raw intervals and gaps are retained. No features or weights change."
)


def _validated_inputs(raw, groups, scores, margin):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and nonnegative")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Candidate utilities must be a finite one-dimensional array")
    if len(raw) != len(groups):
        raise ValueError("One candidate group is required per raw interval")
    previous_end, cursor = -math.inf, 0
    for event, group in zip(raw, groups):
        if (not np.isfinite([event.start, event.end]).all() or event.start < 0
                or event.end <= event.start or event.start < previous_end):
            raise ValueError("Raw intervals must be finite, ordered and nonoverlapping")
        previous_end = event.end
        start, end = group["start"], group["end"]
        candidates = group["candidates"]
        if (not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer))
                or start != cursor or end < start or end > len(values) or end - start != len(candidates)):
            raise ValueError("Candidate groups must partition the utility vector in order")
        if candidates and candidates[0]["label"] != event.chord:
            raise ValueError("The first candidate must be the original raw label")
        cursor = end
    if cursor != len(values):
        raise ValueError("Unassigned candidate utility values")
    return np.clip(values, 0., 1.)


def preserve_root_allowed_first(raw, groups, scores, margin=FIXED_MARGIN):
    """Return (TimedChord rows, decision details) using no reference data."""
    values = _validated_inputs(raw, groups, scores, margin)
    output, decisions = [], []
    for event, group in zip(raw, groups):
        candidates = group["candidates"]
        baseline = harte_chord(event.chord)
        allowed = [0] if candidates else []
        if baseline not in {"N", "X"}:
            for index, candidate in enumerate(candidates[1:], 1):
                label = harte_chord(candidate["label"])
                if label not in {"N", "X"} and mir_eval.chord.root([baseline], [label])[0] == 1:
                    allowed.append(index)
        local = values[group["start"]:group["end"]]
        best = max(allowed, key=lambda index: local[index]) if allowed else None
        chosen = best if best is not None and local[best] > local[0] + margin + 1e-12 else 0
        label = candidates[chosen]["label"] if candidates else event.chord
        output.append(TimedChord(event.start, event.end, label))
        decisions.append(dict(allowed_indices=allowed, selected_index=chosen if candidates else None,
            best_allowed_index=best, candidate_count=len(candidates),
            raw_utility=float(local[0]) if len(local) else None,
            chosen_utility=float(local[chosen]) if len(local) else None,
            raw_status=baseline if baseline in {"N", "X"} else "chord"))
    return output, dict(schema=POLICY_SCHEMA, rule=POLICY_RULE, margin=margin,
        interval_count=len(raw), original_interval_support_preserved=True,
        empty_candidate_groups=sum(not group["candidates"] for group in groups), decisions=decisions)


def frozen_v1_full_head_from_scores(raw, groups, scores, margin=FIXED_MARGIN):
    """Exact v1 rank-then-margin rule using the same already-computed scores."""
    values = _validated_inputs(raw, groups, scores, margin)
    output = []
    for event, group in zip(raw, groups):
        candidates = group["candidates"]
        local = values[group["start"]:group["end"]]
        choice = int(np.argmax(local)) if len(local) else 0
        if len(local) and local[choice] <= local[0] + margin + 1e-12:
            choice = 0
        output.append(TimedChord(event.start, event.end,
                                 candidates[choice]["label"] if candidates else event.chord))
    return output
