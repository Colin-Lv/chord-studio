"""Transposition-invariant learned major/minor emissions, separate from priors.

Models are experimental local artifacts. Only the acoustic candidate ranker is
learned here: timing, key, extensions, and section roles remain separate layers.
"""

from __future__ import annotations

import numpy as np

FEATURE_SCHEMA = "relative_chroma_bass_context_centered_v1"
QUALITY_NAMES = ("", "m")


def candidate_features(chroma_vectors: np.ndarray, bass_vectors: np.ndarray) -> np.ndarray:
    """One row per (interval, root, major/minor), with no absolute-key input.

    Relative pitch coordinates share statistical strength across all 12 keys.
    Neighboring windows describe acoustic context, never reference chords.
    """
    chroma = np.asarray(chroma_vectors, dtype=np.float32)
    bass = np.asarray(bass_vectors, dtype=np.float32)
    if chroma.ndim != 2 or chroma.shape[1] != 12 or bass.shape != chroma.shape:
        raise ValueError("Chroma and bass must both have shape (intervals, 12).")
    if not np.all(np.isfinite(chroma)) or not np.all(np.isfinite(bass)):
        raise ValueError("Features must be finite.")
    if not len(chroma):
        return np.empty((0, 87), dtype=np.float32)
    chroma = chroma / np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), 1e-8)
    bass = bass / np.maximum(np.sum(bass, axis=1, keepdims=True), 1e-8)
    padded_chroma = np.pad(chroma, ((1, 1), (0, 0)), mode="edge")
    padded_bass = np.pad(bass, ((1, 1), (0, 0)), mode="edge")
    components = (chroma, bass, padded_chroma[:-2], padded_chroma[2:],
                  padded_bass[:-2], padded_bass[2:],
                  (padded_chroma[:-2] + chroma + padded_chroma[2:]) / 3)
    roots = np.repeat(np.arange(12), 2)
    relative_pcs = (roots[:, None] + np.arange(12)) % 12
    rotated = [component[:, relative_pcs].reshape(len(chroma), 24, 12) for component in components]
    quality = np.broadcast_to(np.tile([0., 1.], 12)[None, :, None], (len(chroma), 24, 1))
    change_before = np.broadcast_to((1 - np.sum(chroma * padded_chroma[:-2], axis=1))[:, None, None],
                                    (len(chroma), 24, 1))
    change_after = np.broadcast_to((1 - np.sum(chroma * padded_chroma[2:], axis=1))[:, None, None],
                                   (len(chroma), 24, 1))
    return np.concatenate([*rotated, quality, change_before, change_after], axis=2).reshape(-1, 87).astype(np.float32)


def emission_scores(artifact: dict, vectors: np.ndarray, bass_vectors: np.ndarray) -> np.ndarray:
    if artifact.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Acoustic model feature schema does not match this extractor.")
    features = candidate_features(vectors, bass_vectors)
    if not len(features):
        return np.empty((0, 24), dtype=np.float32)
    probabilities = artifact["model"].predict_proba(features)[:, 1].reshape(-1, 24)
    # Binary candidate scores are normalized within each interval. These are
    # ranking scores, not calibrated probabilities of transcription correctness.
    probabilities = np.clip(probabilities, 1e-5, 1 - 1e-5)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    logits -= np.max(logits, axis=1, keepdims=True)
    return logits.astype(np.float32)


def decode_emissions(scores: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Continuity-only sequence decoder; no imposed common progression."""
    import chord_studio as studio
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Transition strength must be finite and nonnegative.")
    if scores.ndim != 2 or scores.shape[1] != 24:
        raise ValueError("Expected a score for each of 24 major/minor chords.")
    transitions = np.full((24, 24), -strength, dtype=np.float32)
    np.fill_diagonal(transitions, 0)
    _, path = studio.viterbi_indices(scores, transitions)
    return path


def recognize_acoustic(y, chroma, bass_chroma, intervals, *, artifact, key, notation="sharps",
                       harmonic_unit="half-bar", beats_per_bar=4, transition_strength=0.5,
                       no_chord_threshold=0.025):
    import chord_studio as studio
    cells = studio.harmonic_intervals_from_beats(intervals, harmonic_unit, beats_per_bar)
    vectors = studio.chroma_for_intervals(chroma, y, studio.SAMPLE_RATE, 2048, cells)
    rms = studio.rms_for_intervals(y, studio.SAMPLE_RATE, cells)
    bv, br, bc = studio.bass_for_intervals(bass_chroma, studio.SAMPLE_RATE, 2048, cells, notation)
    scores = emission_scores(artifact, vectors, bv)
    path = decode_emissions(scores, transition_strength)
    cutoff = max(float(np.quantile(rms, 0.95)) * no_chord_threshold, 1e-5) if len(rms) else 1e-5
    distribution = np.exp(scores)
    distribution /= np.maximum(distribution.sum(axis=1, keepdims=True), 1e-8)
    spans, candidates = [], []
    for i, cell in enumerate(cells):
        index = int(path[i])
        chord = studio.note_name(index // 2, notation) + QUALITY_NAMES[index % 2]
        quiet = rms[i] < cutoff
        confidence = float(distribution[i, index])
        spans.append(studio.ChordSpan("beat", cell.bar, cell.beat_start, cell.beat_end,
                    cell.start, cell.end, "N" if quiet else chord,
                    0.0 if quiet else confidence, "N" if quiet else br[i], 0.0 if quiet else float(bc[i])))
        candidates.append({"bar": cell.bar, "start": cell.start, "end": cell.end,
            "score_kind": "uncalibrated_relative_acoustic_score",
            "candidates": [{"chord": studio.note_name(int(j) // 2, notation) + QUALITY_NAMES[int(j) % 2],
                            "score": float(distribution[i, j])} for j in np.argsort(scores[i])[-5:][::-1]]})
    return studio.collapse_adjacent_spans(spans), candidates
