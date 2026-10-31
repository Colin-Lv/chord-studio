"""Optional acoustic seventh head that cannot change the upstream root/triad.

The upstream recognizer owns chord boundaries and major/minor quality. This
head learns only whether a minor/major seventh is supported by the audio.
"""
from __future__ import annotations

import numpy as np
import mir_eval

from acoustic_chord_model import candidate_features
from evaluate_chords import TimedChord, harte_chord

FEATURE_SCHEMA = "fixed_root_triad_seventh_centered_fast_v1"
FEATURE_SCHEMA_V2 = "fixed_root_triad_seventh_centered_fast_v2"
FEATURE_DIMENSIONS = {FEATURE_SCHEMA: 87, FEATURE_SCHEMA_V2: 148}
TEMPORAL_SUPPORT_THRESHOLD = 0.10


def normalize_feature_schema(feature_schema):
    schema = {"v1": FEATURE_SCHEMA, "v2": FEATURE_SCHEMA_V2}.get(feature_schema, feature_schema)
    if schema not in FEATURE_DIMENSIONS:
        raise ValueError(f"Unknown extension feature schema: {feature_schema!r}")
    return schema


def temporal_features(chroma, rows, attributes, sample_rate, hop_length):
    """61 acoustic features: relative q10/q50/q90/std/support, log duration.

    Frame t is centered at t*hop/sample_rate, without an FFT-window offset.
    Pitch support is the fraction of interval frames with L1-normalized chroma
    >= .10. Silence stays zero. A sub-hop interval uses its nearest center.
    The upstream root supplies rotation only; no reference label/key is read.
    """
    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.ndim != 2 or chroma.shape[0] != 12 or not np.all(np.isfinite(chroma)) or np.any(chroma < 0):
        raise ValueError("Temporal chroma must be a finite nonnegative (12, frames) array")
    if len(rows) != len(attributes) or sample_rate <= 0 or hop_length <= 0:
        raise ValueError("Temporal features require one attribute row per interval and a valid frame clock")
    output = np.zeros((len(rows), 61), dtype=np.float32)
    times = np.arange(chroma.shape[1], dtype=np.float64) * (hop_length / sample_rate)
    normalized = chroma / np.maximum(chroma.sum(axis=0, keepdims=True), 1e-9)
    for i, (row, attributes_row) in enumerate(zip(rows, attributes)):
        start, end = float(row.start), float(row.end)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Extension intervals must have finite, positive duration")
        output[i, -1] = np.log1p(end-start)
        if not len(times):
            continue
        left, right = np.searchsorted(times, [start, end], side="left")
        if left == right:
            center = (start+end)/2
            nearest = int(np.argmin(np.abs(times-center)))
            values = normalized[:, nearest:nearest+1]
        else:
            values = normalized[:, left:right]
        root = attributes_row[0] if attributes_row else 0
        relative = values[(root + np.arange(12)) % 12]
        output[i, :36] = np.quantile(relative, [.10, .50, .90], axis=1).reshape(-1)
        output[i, 36:48] = np.std(relative, axis=1)
        output[i, 48:60] = np.mean(relative >= TEMPORAL_SUPPORT_THRESHOLD, axis=1)
    return output


def chord_attributes(label):
    normalized = harte_chord(label)
    if normalized in {"N", "X"}:
        return None
    root, tones, _ = mir_eval.chord.encode(normalized)
    # This head deliberately has no authority over suspended/diminished chords.
    if not tones[0] or not tones[7] or bool(tones[3]) == bool(tones[4]):
        return None
    if tones[10] and tones[11]:
        return None
    return int(root), int(bool(tones[3])), 1 if tones[10] else 2 if tones[11] else 0


def features_for_rows(y, rows, feature_schema=FEATURE_SCHEMA):
    """Extract v1 (87) or opt-in v2 (148); v2 preserves the first 87 columns."""
    feature_schema = normalize_feature_schema(feature_schema)
    if not rows:
        return np.empty((0, FEATURE_DIMENSIONS[feature_schema]), dtype=np.float32), []
    import chord_studio as studio
    intervals = [studio.BeatInterval(i + 1, 1, row.start, row.end) for i, row in enumerate(rows)]
    chroma = studio.extract_chroma(y, studio.SAMPLE_RATE, 2048, "fast")
    bass = studio.extract_bass_chroma(y, studio.SAMPLE_RATE, 2048, "fast")
    vectors = studio.chroma_for_intervals(chroma, y, studio.SAMPLE_RATE, 2048, intervals)
    bv, _, _ = studio.bass_for_intervals(bass, studio.SAMPLE_RATE, 2048, intervals, "sharps")
    all_features = candidate_features(vectors, bv).reshape(len(rows), 24, 87)
    attributes = [chord_attributes(row.chord) for row in rows]
    indices = np.asarray([item[0] * 2 + item[1] if item else 0 for item in attributes])
    selected = all_features[np.arange(len(rows)), indices]
    if feature_schema == FEATURE_SCHEMA_V2:
        temporal = temporal_features(chroma, rows, attributes, studio.SAMPLE_RATE, 2048)
        selected = np.concatenate([selected, temporal], axis=1)
    return selected, attributes


def extend_rows(rows, features, artifact, threshold=None, feature_schema=None):
    """Keep all intervals, roots, triads, silence and existing rich labels."""
    import chord_studio as studio
    schema = normalize_feature_schema(artifact.get("feature_schema"))
    if feature_schema is not None and normalize_feature_schema(feature_schema) != schema:
        raise ValueError("Extension model feature schema mismatch")
    features = np.asarray(features)
    if not rows and features.size == 0:
        features = np.empty((0, FEATURE_DIMENSIONS[schema]), dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSIONS[schema]:
        raise ValueError(f"Extension schema {schema} requires {FEATURE_DIMENSIONS[schema]} feature columns")
    if not np.all(np.isfinite(features)):
        raise ValueError("Extension features must be finite")
    model_dimension = getattr(artifact.get("model"), "n_features_in_", None)
    if model_dimension is not None and int(model_dimension) != FEATURE_DIMENSIONS[schema]:
        raise ValueError("Extension classifier feature dimension does not match its declared schema")
    threshold = artifact["threshold"] if threshold is None else float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Extension threshold must be in [0, 1]")
    if len(features) != len(rows):
        raise ValueError("Every chord interval must have exactly one extension feature row")
    if not rows:
        return [], []
    if threshold >= 1:
        return list(rows), [{"start": row.start, "end": row.end, "original": row.chord, "chord": row.chord,
                            "seventh_class": None, "score": None, "score_kind": "disabled_identity"} for row in rows]
    probabilities = artifact["model"].predict_proba(features)
    classes = artifact["model"].classes_
    output, evidence = [], []
    for i, row in enumerate(rows):
        attributes = chord_attributes(row.chord)
        chord = row.chord
        best = int(np.argmax(probabilities[i]))
        extension = int(classes[best])
        score = float(probabilities[i, best])
        if attributes and attributes[2] == 0 and extension and score >= threshold:
            root, minor, _ = attributes
            # Only bare major/minor labels are mutable; retain add9/6 etc.
            body = harte_chord(chord).split("/", 1)[0]
            if body.endswith((":maj", ":min")):
                suffix = {(0, 1): "7", (0, 2): "maj7", (1, 1): "m7", (1, 2): "mMaj7"}[minor, extension]
                chord = studio.note_name(root, "sharps") + suffix
                # Preserve explicit bass spelling independently of the seventh.
                if "/" in row.chord:
                    from evaluate_chords import convert_annotation_chord
                    chord += "/" + convert_annotation_chord(row.chord).split("/", 1)[1]
        output.append(TimedChord(row.start, row.end, chord))
        evidence.append({"start": row.start, "end": row.end, "original": row.chord, "chord": chord,
                         "seventh_class": extension, "score": score,
                         "score_kind": "uncalibrated_extension_evidence"})
    return output, evidence
