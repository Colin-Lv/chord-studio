"""Reference-free v2 features. Never compatible with the frozen 136-D head."""
from __future__ import annotations

import numpy as np

import harmonic_note_features as notes
from chord_semantics_v2 import parse_chord
from evaluate_chords import TimedChord
from multisource_candidates_v2 import SOURCE_NAMES, candidate_pool, source_support_for
from nnls_seventh_model import acoustic_windows

SCHEMA = "nnls_named_sources_full_chord_explicit_bass_139_v2"
DIMENSION = 139
STATES = ("chord", "N", "X", "missing")
FEATURE_NAMES = (tuple("nnls_" + name for name in notes.FEATURE_NAMES)
                 + tuple(f"{source}_{level}_support" for source in SOURCE_NAMES for level in ("root", "triad", "full"))
                 + tuple(source + "_coverage" for source in SOURCE_NAMES)
                 + ("log_duration",) + tuple("state_" + state for state in STATES)
                 + tuple(f"chord_relative_pc_{i}" for i in range(12))
                 + tuple(f"explicit_bass_relative_pc_{i}" for i in range(12)) + ("bass_unannotated",))
assert len(FEATURE_NAMES) == DIMENSION


def candidate_components(label):
    semantic = parse_chord(label)
    mask = np.asarray(semantic.chord_bitmap, dtype=float)
    bass = np.zeros(13)
    bass[semantic.bass_relative if semantic.explicit_bass else 12] = 1
    surrogate = "N"
    if semantic.state == "chord":
        surrogate = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[semantic.root_pc]
        if mask[3] and not mask[4]:
            surrogate += "m"
    return mask, bass, surrogate, semantic.state


def automatic_candidate_features(anchors, sources, evidence):
    """Anchors carry time and the explicit fallback label; sources keep names.

    No reference/target argument exists. Changing anchors must not swap raw and
    large-vocabulary support channels. Source gaps and X remain distinct.
    For a new time grid the caller must explicitly supply raw-derived fallback
    labels; simply passing LV rows changes the fallback and is a separate
    intervention. Current training uses original raw anchors exclusively.
    """
    groups, windows, auxiliary = [], [], []
    for event in anchors:
        pool = candidate_pool(event, sources)
        start = len(windows)
        for candidate in pool.candidates:
            support = np.column_stack([source_support_for(event, candidate, sources, level=level)
                                       for level in ("root", "triad", "full")]).reshape(-1)
            mask, bass, surrogate, state = candidate_components(candidate.label)
            windows.append(TimedChord(event.start, event.end, surrogate))
            auxiliary.append(np.concatenate((support, pool.source_coverage,
                [np.log1p(event.end - event.start)], [float(state == value) for value in STATES], mask, bass)))
        groups.append(dict(start=start, end=len(windows), pool=pool, raw_index=pool.raw_index,
                           candidates=[dict(label=c.label, source_support=np.asarray(c.source_support), semantic=c.semantics)
                                       for c in pool.candidates]))
    if not windows:
        return np.empty((0, DIMENSION), dtype=np.float32), groups
    acoustic = notes.features_for_intervals(evidence, acoustic_windows(windows, evidence.metadata["duration_seconds"]))
    features = np.concatenate((acoustic, np.asarray(auxiliary).reshape(-1, DIMENSION-97)), axis=1).astype(np.float32)
    if features.shape != (len(windows), DIMENSION) or not np.isfinite(features).all():
        raise ValueError("Invalid v2 feature shape or values")
    return features, groups
