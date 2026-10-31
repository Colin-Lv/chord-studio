"""Lossless six-head evidence from the original ISMIR2019 ensemble.

Frame centers and responsibility cells are different concepts: native centered
CQT frame k is at k * hop, while the unchanged vendor HMM assigns its label to
[k * hop, (k + 1) * hop). Aggregation uses those cells, clipped to real audio.
No reference, anchor label, or key is an acoustic evidence input.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import NamedTuple

import numpy as np

SCHEMA = "ismir2019_six_head_posteriors_v2"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
HEAD_NAMES = ("triad", "bass", "seventh", "ninth", "eleventh", "thirteenth")
# complex_chord.py NUM_TO_ABS_SCALE and Chord.to_numpy; chordnet_ismir_naive
# chord_limit=(6,3,3,2,2). Negative unknown/not-care sentinels are not classes.
PITCH_CLASSES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
HEAD_LABELS = {
    "triad": ["N"] + [f"{root}:{quality}" for quality in
        ("maj", "min", "sus4", "sus2", "dim", "aug") for root in PITCH_CLASSES],
    "bass": ["none"] + PITCH_CLASSES,
    "seventh": ["none", "7", "b7", "bb7"],
    "ninth": ["none", "9", "#9", "b9"],
    "eleventh": ["none", "11", "#11"],
    "thirteenth": ["none", "13", "b13"],
}
HEAD_DIMENSIONS = {name: len(HEAD_LABELS[name]) for name in HEAD_NAMES}


class PosteriorEvidence(NamedTuple):
    frame_samples: np.ndarray
    probabilities: dict[str, np.ndarray]
    metadata: dict

    @property
    def times(self):
        return self.frame_samples.astype(np.float64) / SAMPLE_RATE


class AggregatedPosteriors(NamedTuple):
    intervals: np.ndarray
    probabilities: dict[str, np.ndarray]
    coverage: np.ndarray
    metadata: dict


def validate_probabilities(probabilities, frame_count=None):
    if set(probabilities) != set(HEAD_NAMES):
        raise ValueError("Exactly six original probability heads are required")
    count = None
    for name, size in HEAD_DIMENSIONS.items():
        values = np.asarray(probabilities[name])
        if values.ndim != 2 or values.shape[1] != size or not len(values):
            raise ValueError(f"Invalid {name} probability shape; expected nonempty [frames,{size}]")
        if values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError(f"{name} probabilities must be finite floating point values")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{name} probabilities are outside [0,1]")
        if not np.allclose(values.sum(axis=1, dtype=np.float64), 1., rtol=0, atol=2e-5):
            raise ValueError(f"{name} probabilities are not normalized")
        if count is None:
            count = len(values)
        if len(values) != count or (frame_count is not None and len(values) != frame_count):
            raise ValueError("Probability heads have inconsistent frame counts")
    return count


def make_evidence(probabilities, identity, *, decoded_samples):
    if isinstance(decoded_samples, bool) or int(decoded_samples) != decoded_samples or decoded_samples <= 0:
        raise ValueError("Cannot export empty audio; decoded_samples must be a positive integer")
    count = validate_probabilities(probabilities)
    if count != int(decoded_samples) // HOP_LENGTH + 1:
        raise ValueError("Posterior frame count disagrees with the centered CQT sample clock")
    metadata = dict(
        schema=SCHEMA, identity=copy.deepcopy(identity),
        head_labels=copy.deepcopy(HEAD_LABELS), head_dimensions=HEAD_DIMENSIONS.copy(),
        sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, audio_offset=0.,
        decoded_samples=int(decoded_samples), audio_seconds=int(decoded_samples) / SAMPLE_RATE,
        frame_count=count, frame_position="centered CQT frame k at sample k * hop_length",
        aggregation_support="[k*hop,(k+1)*hop) clipped to decoded_samples",
        probability_kind="uncalibrated arithmetic mean of five original softmax heads",
        suffix_conditioning="global head classes, not root-conditioned posterior axes",
        triad_index="0=N; (quality-1)*12+1+root_pc; quality=maj,min,sus4,sus2,dim,aug",
        bass_index="0=none; absolute pitch class C=0 through B=11 maps to pc+1",
        unknown_class=False, reference_inputs=False,
    )
    result = PosteriorEvidence(np.arange(count, dtype=np.int64) * HOP_LENGTH,
        {name: np.asarray(probabilities[name]).copy() for name in HEAD_NAMES}, metadata)
    validate_evidence(result)
    return result


def validate_evidence(evidence, expected_identity=None):
    meta = evidence.metadata
    if meta.get("schema") != SCHEMA:
        raise ValueError("Unsupported posterior schema; v1 caches cannot be reused as v2")
    if expected_identity is not None and meta.get("identity") != expected_identity:
        raise ValueError("Posterior cache identity differs from the requested source/model/frontend")
    if not isinstance(meta.get("identity"), dict) or not meta["identity"]:
        raise ValueError("Posterior evidence requires explicit source identity")
    if meta.get("head_labels") != HEAD_LABELS or meta.get("head_dimensions") != HEAD_DIMENSIONS:
        raise ValueError("Posterior class mapping differs from the original trained heads")
    if meta.get("sample_rate") != SAMPLE_RATE or meta.get("hop_length") != HOP_LENGTH or meta.get("audio_offset") != 0.:
        raise ValueError("Unsupported posterior sample clock")
    samples = meta.get("decoded_samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("Cannot export empty audio or an invalid decoded sample count")
    count = samples // HOP_LENGTH + 1
    if meta.get("frame_count") != count or meta.get("audio_seconds") != samples / SAMPLE_RATE:
        raise ValueError("Posterior duration or frame count disagrees with decoded audio")
    positions = np.asarray(evidence.frame_samples)
    if positions.dtype.kind not in "iu" or not np.array_equal(positions, np.arange(count, dtype=np.int64) * HOP_LENGTH):
        raise ValueError("Posterior frame positions differ from the native sample clock")
    validate_probabilities(evidence.probabilities, count)
    return evidence


def save_posteriors(path, evidence):
    """Atomically publish one self-contained NPZ; never overwrite an old cache."""
    validate_evidence(evidence)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Posterior cache already exists: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **evidence.probabilities, frame_samples=evidence.frame_samples,
                metadata_json=np.array(json.dumps(evidence.metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)))
            stream.flush()
            os.fsync(stream.fileno())
        # The bridge owns a per-output-directory lock; this additional guard
        # protects callers that inadvertently attempt sequential cache reuse.
        if path.exists():
            raise FileExistsError(f"Posterior cache already exists: {path}")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_posteriors(path, expected_identity=None):
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(HEAD_NAMES) | {"frame_samples", "metadata_json"}:
            raise ValueError("Posterior NPZ has missing or unexpected arrays")
        encoded = archive["metadata_json"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("Posterior metadata must be a scalar JSON string")
        metadata = json.loads(str(encoded.item()))
        result = PosteriorEvidence(archive["frame_samples"].copy(),
            {name: archive[name].copy() for name in HEAD_NAMES}, metadata)
    return validate_evidence(result, expected_identity)


def aggregate_posteriors(evidence, intervals, *, anchor_source):
    """Duration-weighted means on unchanged requested second boundaries.

Missing support has coverage=0 and NaN means, never an invented N prediction.
Anchors may overlap; each interval is evaluated independently. Source identity
always describes acoustic evidence, while anchor_source only describes timing.
"""
    validate_evidence(evidence)
    boundaries = np.asarray(intervals, dtype=np.float64)
    if boundaries.size == 0:
        boundaries = np.empty((0, 2), dtype=np.float64)
    if boundaries.ndim != 2 or boundaries.shape[1] != 2 or not np.isfinite(boundaries).all():
        raise ValueError("Intervals must be finite [start,end] second boundaries")
    if np.any(boundaries[:, 0] < 0) or np.any(boundaries[:, 1] <= boundaries[:, 0]):
        raise ValueError("Intervals must have nonnegative starts and positive duration")
    if not isinstance(anchor_source, dict) or not anchor_source:
        raise ValueError("anchor_source must explicitly identify the boundary source")
    left = evidence.times
    right = np.minimum((evidence.frame_samples + HOP_LENGTH) / SAMPLE_RATE, evidence.metadata["audio_seconds"])
    means = {name: np.full((len(boundaries), size), np.nan) for name, size in HEAD_DIMENSIONS.items()}
    coverage = np.zeros(len(boundaries), dtype=np.float64)
    for index, (start, end) in enumerate(boundaries):
        first = np.searchsorted(right, start, side="right")
        last = np.searchsorted(left, end, side="left")
        weights = np.maximum(0., np.minimum(right[first:last], end) - np.maximum(left[first:last], start))
        support = weights.sum()
        coverage[index] = min(1., support / (end - start))
        if support > 0:
            for name in HEAD_NAMES:
                means[name][index] = weights @ evidence.probabilities[name][first:last] / support
    metadata = dict(schema="ismir2019_posterior_interval_means_v2",
        source_identity=copy.deepcopy(evidence.metadata["identity"]), anchor_source=copy.deepcopy(anchor_source),
        source_schema=SCHEMA, head_labels=copy.deepcopy(HEAD_LABELS),
        aggregation_support=evidence.metadata["aggregation_support"],
        missing_support="coverage=0 and NaN means; no prediction is fabricated",
        boundary_changes=False, reference_inputs=False)
    return AggregatedPosteriors(boundaries.copy(), means, coverage, metadata)
