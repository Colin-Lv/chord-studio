"""Reference-free v2 candidate pools with named, union-based source support."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from chord_semantics_v2 import ChordSemantics, STATES, parse_chord, same_chord

SCHEMA = "named_multisource_candidates_v2"
SOURCE_NAMES = ("raw_madmom", "chordino", "largevocab")


@dataclass(frozen=True)
class TimedChord:
    start: float
    end: float
    chord: str | None


def interval_values(row) -> tuple[float, float, str | None]:
    start, end, label = ((row["start"], row["end"], row.get("chord")) if isinstance(row, Mapping)
                         else (row.start, row.end, row.chord))
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"Invalid interval [{start}, {end}]")
    if label is not None and not isinstance(label, str):
        raise TypeError("Chord labels must be strings or None")
    return start, end, label


def union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    right = -math.inf
    for start, end in sorted(intervals):
        if end > max(start, right):
            total += end - max(start, right)
        right = max(right, end)
    return total


def _sources(sources: Mapping) -> None:
    if not isinstance(sources, Mapping):
        raise TypeError("Sources must be a named mapping, never a positional source list")
    unexpected = set(sources) - set(SOURCE_NAMES)
    if unexpected:
        raise ValueError(f"Unknown sources: {sorted(unexpected)}")


def _overlapping(interval, rows):
    left, right, _ = interval_values(interval)
    for row in rows:
        start, end, label = interval_values(row)
        start, end = max(left, start), min(right, end)
        if end > start:
            yield start, end, parse_chord(label)


@dataclass(frozen=True)
class Candidate:
    label: str | None
    canonical_label: str | None
    semantics: ChordSemantics
    source_support: tuple[float, ...]
    raw_labels: Mapping[str, tuple[str | None, ...]]


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[Candidate, ...]
    raw_index: int
    source_coverage: tuple[float, ...]
    source_state_coverage: Mapping[str, Mapping[str, float]]


def source_support_for(interval, candidate, sources: Mapping, level: str = "full") -> tuple[float, ...]:
    """Union all matching intervals per source; duplicates never add evidence.

    Root/triad support is unioned AFTER grouping matching full candidates. It
    must not be reconstructed by adding full-candidate supports, because those
    intervals may overlap. Each component is in [0,1]. No reference is accepted.
    """
    _sources(sources)
    if level not in {"root", "triad", "tetrad", "full"}:
        raise ValueError(f"Unknown support level {level!r}")
    semantic = (candidate.semantics if isinstance(candidate, Candidate) else candidate
                if isinstance(candidate, ChordSemantics) else parse_chord(candidate))
    start, end, _ = interval_values(interval)
    return tuple(union_seconds([(left, right) for left, right, observed in
                                _overlapping(interval, sources.get(source, ()))
                                if same_chord(semantic, observed, level)]) / (end - start)
                 for source in SOURCE_NAMES)


def candidate_pool(interval, sources: Mapping) -> CandidatePool:
    """Raw candidate first; remaining identities sorted independently of input order.

    Missing source keys mean absent evidence. An interval with a missing label
    is not covered evidence; N and X are covered observations with distinct
    states. Conflicting source-state intervals may overlap, so state coverages
    are individual unions, not a partition whose sum is necessarily one.
    """
    _sources(sources)
    start, end, raw_label = interval_values(interval)
    raw = parse_chord(raw_label)
    observed = {source: list(_overlapping(interval, sources.get(source, ()))) for source in SOURCE_NAMES}
    semantics = {raw.identity: [raw]}
    for rows in observed.values():
        for _, _, chord in rows:
            if chord.state != "missing":
                semantics.setdefault(chord.identity, []).append(chord)
    keys = [raw.identity] + sorted((key for key in semantics if key != raw.identity), key=repr)
    candidates = []
    for key in keys:
        choices = semantics[key]
        representative = raw if key == raw.identity else min(
            choices, key=lambda item: (item.canonical_label or "", item.raw_label or ""))
        canonical = min((item.canonical_label for item in choices if item.canonical_label is not None), default=None)
        raw_labels = {source: tuple(sorted({chord.raw_label for _, _, chord in rows if chord.identity == key},
                                          key=lambda label: (label is not None, label or "")))
                      for source, rows in observed.items()}
        candidates.append(Candidate(raw_label if key == raw.identity else canonical,
                                    canonical, representative,
                                    source_support_for(interval, representative, sources), raw_labels))
    coverage = tuple(union_seconds([(left, right) for left, right, chord in observed[source]
                                    if chord.state != "missing"]) / (end - start) for source in SOURCE_NAMES)
    state_coverage = {source: {state: union_seconds([(left, right) for left, right, chord in observed[source]
                                                   if chord.state == state]) / (end - start)
                              for state in STATES} for source in SOURCE_NAMES}
    # Gaps have missing state even when there is no explicit None row.
    for source, covered in zip(SOURCE_NAMES, coverage):
        state_coverage[source]["missing"] = 1.0 - covered
    return CandidatePool(tuple(candidates), 0, coverage, state_coverage)
