"""Experimental note-evidence quality head on reference-independent subwindows."""
from __future__ import annotations

import math
import re
import numpy as np

from chord_extension_model import chord_attributes
from evaluate_chords import TimedChord, harte_chord, convert_annotation_chord

MODEL_SCHEMA = "nnls_fixed_root_seventh_v1"


def acoustic_windows(rows, duration):
    """Clip feature support only, retaining raw output's 100ms rounding clock."""
    if duration <= 0 or any(row.end > duration + .100001 for row in rows):
        raise ValueError('Upstream LAB exceeds audio support beyond a 100ms source frame')
    return [TimedChord(min(row.start, max(0., duration-1/44100)), min(row.end, duration), row.chord)
            for row in rows]


def split_upstream(rows, maximum_window=0.):
    """Retain every source edge and gap; subdivide by time without annotations."""
    if not math.isfinite(maximum_window) or maximum_window < 0:
        raise ValueError("Window size must be finite and nonnegative")
    windows, parents = [], []
    for i, row in enumerate(rows):
        if row.end <= row.start or not math.isfinite(row.start + row.end):
            raise ValueError("Invalid upstream interval")
        count = max(1, int(math.ceil((row.end-row.start)/maximum_window - 1e-10))) if maximum_window else 1
        for part in range(count):
            start = row.start + part*maximum_window if maximum_window else row.start
            end = min(row.end, row.start+(part+1)*maximum_window) if maximum_window else row.end
            windows.append(TimedChord(start, end, row.chord))
            parents.append(i)
    return windows, np.asarray(parents, dtype=np.int64)


def mutable_label(label):
    attributes = chord_attributes(label)
    return attributes is not None and attributes[2] == 0 and harte_chord(label).split('/')[0].endswith((':maj', ':min'))


def label_with_seventh(label, seventh):
    if seventh == 0 or not mutable_label(label):
        return label
    attributes = chord_attributes(label)
    suffix = {(0, 1): '7', (0, 2): 'maj7', (1, 1): 'm7', (1, 2): 'mMaj7'}[(attributes[1], int(seventh))]
    root = re.match(r'[A-G][b#]*', harte_chord(label)).group(0)
    bass = '/' + convert_annotation_chord(label).split('/', 1)[1] if '/' in label else ''
    return root + suffix + bass


def supervision(rows, reference, minimum_support=.8):
    """Labels supervise fixed windows, and never choose their edges or roots."""
    targets = np.full(len(rows), -1, dtype=np.int8)
    for i, row in enumerate(rows):
        base = chord_attributes(row.chord)
        if not mutable_label(row.chord):
            continue
        support = np.zeros(3)
        for ref in reference:
            overlap = min(row.end, ref.end)-max(row.start, ref.start)
            attributes = chord_attributes(ref.chord) if overlap > 0 else None
            if attributes and attributes[:2] == base[:2]:
                support[attributes[2]] += overlap
        best = int(support.argmax())
        if support[best] >= minimum_support * (row.end-row.start):
            targets[i] = best
    return targets


def predict_probabilities(model, features):
    values = np.asarray(model.predict_proba(features))
    classes = np.asarray(model.classes_)
    if (values.shape != (len(features), len(classes)) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1)
            or not np.isin(classes, [0, 1, 2]).all() or len(set(classes)) != len(classes)):
        raise ValueError("Invalid seventh evidence")
    output = np.zeros((len(features), 3), dtype=np.float64)
    output[:, classes.astype(int)] = values
    return output


def decode_windows(upstream, windows, parents, probabilities, threshold):
    """Add seventh quality only; unchanged parents retain their original rows."""
    if not 0 <= threshold <= 1 or not np.isfinite(threshold):
        raise ValueError("Invalid threshold")
    if threshold == 1:
        return list(upstream), []
    if probabilities.shape != (len(windows), 3) or len(parents) != len(windows):
        raise ValueError("One probability row and parent per window required")
    labels = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(labels)), labels]
    labels[confidence < threshold] = 0
    result, changes = [], []
    for parent, source in enumerate(upstream):
        indices = np.flatnonzero(parents == parent)
        if not len(indices):
            raise ValueError("An upstream interval has no windows")
        current = [TimedChord(windows[i].start, windows[i].end,
                             label_with_seventh(source.chord, labels[i])) for i in indices]
        if all(row.chord == source.chord for row in current):
            result.append(source)
            continue
        merged = []
        for i, row in zip(indices, current):
            if row.chord != source.chord:
                changes.append(dict(start=row.start, end=row.end, original=source.chord, chord=row.chord,
                                    score=float(confidence[i]), score_kind='uncalibrated_extension_evidence'))
            if merged and merged[-1].chord == row.chord:
                merged[-1] = TimedChord(merged[-1].start, row.end, row.chord)
            else:
                merged.append(row)
        result.extend(merged)
    return result, changes
