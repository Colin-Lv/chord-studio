"""Masked reference-only supervision for independent v2 candidate heads.

These are decomposed training targets, NOT replacements for the project's
standard mir_eval metrics. Bass is separated from harmonic triad/tetrad.
Unannotated extensions never become negative examples under the default policy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Mapping, Sequence
import math

import numpy as np

from chord_semantics_v2 import ChordSemantics, EXTENSIONS, parse_chord, same_chord
from multisource_candidates_v2 import Candidate, CandidatePool, interval_values

SCHEMA = "masked_candidate_targets_v2"
HEADS = ("root", "triad", "tetrad", "bass", "ext9", "ext11", "ext13")


@dataclass(frozen=True)
class ReferenceCoveragePolicy:
    valid_fields: tuple[str, ...] = HEADS
    extension_absence_known: tuple[int, ...] = ()
    implicit_bass_is_root: bool = False
    include_no_chord: bool = True

    def __post_init__(self):
        if set(self.valid_fields) - set(HEADS):
            raise ValueError("Unknown target fields")
        if set(self.extension_absence_known) - set(EXTENSIONS):
            raise ValueError("Only extensions 9, 11, 13 can be covered")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MaskedTargets:
    values: np.ndarray
    valid_seconds: np.ndarray
    mask: np.ndarray
    interval_seconds: float
    policy: ReferenceCoveragePolicy
    heads: tuple[str, ...] = HEADS

    @property
    def coverage(self) -> np.ndarray:
        return self.valid_seconds / self.interval_seconds


@dataclass(frozen=True)
class MaskedUtility:
    values: np.ndarray
    mask: np.ndarray
    observed_weight: np.ndarray
    coverage: np.ndarray


def reference_field_mask(reference: ChordSemantics, policy: ReferenceCoveragePolicy) -> tuple[bool, ...]:
    if reference.state in {"X", "missing"}:
        return (False,) * len(HEADS)
    if reference.state == "N":
        return tuple(policy.include_no_chord and head in policy.valid_fields and head in HEADS[:3]
                     for head in HEADS)
    return tuple(head in policy.valid_fields and (
        head in HEADS[:3] or (head == "bass" and (reference.explicit_bass or policy.implicit_bass_is_root))
        or (head.startswith("ext") and (int(head[3:]) in reference.mentioned_extensions
                                       or int(head[3:]) in policy.extension_absence_known))) for head in HEADS)


def component_matches(candidate: ChordSemantics, reference: ChordSemantics,
                      policy: ReferenceCoveragePolicy) -> tuple[float, ...]:
    """Per-head correctness; callers MUST apply reference_field_mask."""
    basic = [float(same_chord(candidate, reference, level)) for level in HEADS[:3]]
    # A missing candidate bass is unknown, never an implicit root prediction.
    ref_bass = (reference.bass_pc if reference.explicit_bass else reference.root_pc
                if policy.implicit_bass_is_root and reference.state == "chord" else None)
    bass = float(candidate.state == reference.state == "chord" and candidate.explicit_bass
                 and ref_bass is not None and candidate.bass_pc == ref_bass)
    extensions = [float(candidate.state == reference.state == "chord"
                        and candidate.root_pc == reference.root_pc
                        and candidate.extension(degree) == reference.extension(degree)) for degree in EXTENSIONS]
    return tuple([*basic, bass, *extensions])


def reference_mask_for_row(reference: ChordSemantics, row,
                           policy: ReferenceCoveragePolicy) -> np.ndarray:
    """Per-row masks only remove supervision; they cannot invent annotations."""
    valid = np.asarray(reference_field_mask(reference, policy), dtype=bool)
    get = row.get if isinstance(row, Mapping) else lambda key: getattr(row, key, None)
    fields = get("valid_fields")
    if fields is not None:
        if isinstance(fields, str) or set(fields) - set(HEADS):
            raise ValueError("Per-reference valid_fields must contain recognized field names")
        valid &= np.asarray([head in fields for head in HEADS])
    mask = get("field_mask")
    if mask is not None:
        if not isinstance(mask, Mapping) or set(mask) - set(HEADS):
            raise ValueError("Per-reference field_mask must be a named field-to-bool mapping")
        if any(not isinstance(value, bool) for value in mask.values()):
            raise ValueError("Per-reference mask values must be booleans")
        valid &= np.asarray([mask.get(head, True) for head in HEADS])
    return valid


def candidate_targets(interval, candidates: CandidatePool | Sequence[Candidate | ChordSemantics | str | None],
                      references: Sequence, policy: ReferenceCoveragePolicy | None = None) -> MaskedTargets:
    """Exact-duration soft targets on the unchanged candidate interval.

    References are supervision only. Disjoint reference gaps remain masked;
    overlapping references are rejected rather than double counted. Optional
    per-row ``valid_fields`` can only REMOVE global policy validity, never
    invent absent bass or extension annotations.
    """
    policy = policy or ReferenceCoveragePolicy()
    start, end, _ = interval_values(interval)
    pool = candidates.candidates if isinstance(candidates, CandidatePool) else candidates
    semantics = [candidate.semantics if isinstance(candidate, Candidate) else candidate
                 if isinstance(candidate, ChordSemantics) else parse_chord(candidate) for candidate in pool]
    totals = np.zeros((len(semantics), len(HEADS)), dtype=np.float64)
    durations = np.zeros_like(totals)
    rows = sorted([(interval_values(row), row) for row in references], key=lambda pair: pair[0][:2])
    previous_end = -float("inf")
    for (left, right, label), row in rows:
        if left < previous_end:
            raise ValueError("Overlapping reference intervals cannot supply valid-duration targets")
        previous_end = right
        amount = min(end, right) - max(start, left)
        if amount <= 0:
            continue
        reference = parse_chord(label)
        valid = reference_mask_for_row(reference, row, policy)
        for index, candidate in enumerate(semantics):
            totals[index] += amount * valid * np.asarray(component_matches(candidate, reference, policy))
            durations[index] += amount * valid
    mask = durations > 0
    values = np.full_like(totals, np.nan)
    np.divide(totals, durations, out=values, where=mask)
    return MaskedTargets(values, durations, mask, end - start, policy)


def aggregate_utility(targets: MaskedTargets, weights: Mapping[str, float]) -> MaskedUtility:
    """Explicit weighted average over observed heads, never masked-as-negative.

    There are no default weights. Coverage is the weighted observed duration
    divided by weighted full interval duration. It is reported separately and
    is not silently multiplied into the utility or chosen as a sample weight.
    """
    if not isinstance(weights, Mapping) or set(weights) - set(HEADS):
        raise ValueError("Utility weights must map recognized heads to numbers")
    vector = np.asarray([float(weights.get(head, 0.0)) for head in HEADS])
    if not np.isfinite(vector).all() or np.any(vector < 0) or vector.sum() <= 0:
        raise ValueError("Utility weights must be finite, nonnegative and have positive sum")
    observed = targets.mask * vector
    denominator = observed.sum(axis=1)
    values = np.full(len(denominator), np.nan)
    mask = denominator > 0
    numerator = (np.where(targets.mask, targets.values, 0.0) * vector).sum(axis=1)
    np.divide(numerator, denominator, out=values, where=mask)
    coverage = (targets.valid_seconds * vector).sum(axis=1) / (targets.interval_seconds * vector.sum())
    return MaskedUtility(values, mask, denominator, coverage)


def annotation_coverage(references: Sequence, policy: ReferenceCoveragePolicy | None = None) -> dict:
    """Duration of actually supervised fields, including reference gaps in span."""
    policy = policy or ReferenceCoveragePolicy()
    if not references:
        return {"span_seconds": 0.0, "labelled_seconds": 0.0, "gap_seconds": 0.0,
                "state_seconds": {}, "valid_seconds": dict.fromkeys(HEADS, 0.0),
                "coverage": dict.fromkeys(HEADS, 0.0), "policy": policy.to_dict()}
    rows = sorted([(interval_values(row), row) for row in references], key=lambda pair: pair[0][:2])
    start, end = rows[0][0][0], max(values[1] for values, _ in rows)
    seconds = np.zeros(len(HEADS))
    labelled = 0.0
    state_seconds = {}
    previous_end = -math.inf
    for (left, right, label), row in rows:
        if left < previous_end:
            raise ValueError("Overlapping reference intervals cannot supply annotation coverage")
        previous_end = right
        amount = right - left
        labelled += amount
        chord = parse_chord(label)
        state_seconds[chord.state] = state_seconds.get(chord.state, 0.0) + amount
        seconds += amount * reference_mask_for_row(chord, row, policy)
    return {"span_seconds": end - start, "labelled_seconds": labelled,
            "gap_seconds": end - start - labelled, "state_seconds": state_seconds,
            "valid_seconds": dict(zip(HEADS, seconds.tolist())),
            "coverage": dict(zip(HEADS, (seconds / (end - start)).tolist())),
            "policy": policy.to_dict()}
