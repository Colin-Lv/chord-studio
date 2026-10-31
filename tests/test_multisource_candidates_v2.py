import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from chord_semantics_v2 import parse_chord
from multisource_candidates_v2 import TimedChord as T, candidate_pool, source_support_for, SOURCE_NAMES


def test_named_sources_are_order_independent_and_raw_is_first():
    event = T(0, 4, "C7")
    sources = {"raw_madmom": [event], "chordino": [T(0, 4, "C9")],
               "largevocab": [T(0, 2, "C:7(9)"), T(2, 4, "Cmaj7")]}
    first = candidate_pool(event, sources)
    second = candidate_pool(event, {k: list(reversed(v)) for k, v in reversed(list(sources.items()))})
    assert first == second
    assert SOURCE_NAMES == ("raw_madmom", "chordino", "largevocab")
    assert first.raw_index == 0 and first.candidates[0].label == "C7"
    assert len(first.candidates) == 3
    ninth = next(c for c in first.candidates if c.semantics.extension(9))
    assert ninth.source_support == (0, 1, .5)
    assert ninth.raw_labels["chordino"] == ("C9",)
    assert ninth.raw_labels["largevocab"] == ("C:7(9)",)


def test_duplicate_and_overlapping_same_source_do_not_inflate_support():
    event = T(10, 14, "C")
    sources = {"raw_madmom": [event], "chordino": [T(9, 12, "C9"), T(11, 13, "C:7(9)"),
                                                     T(9, 12, "C9")], "largevocab": []}
    pool = candidate_pool(event, sources)
    ninth = next(c for c in pool.candidates if c.semantics.extension(9))
    assert ninth.source_support == (0, .75, 0)
    assert pool.source_coverage == (1, .75, 0)


def test_root_and_triad_support_union_different_overlapping_candidates():
    event = T(0, 4, "C")
    sources = {"chordino": [T(0, 3, "C7"), T(1, 4, "C9"), T(2, 4, "Cmaj7")]}
    assert source_support_for(event, "C", sources, "root") == (0, 1, 0)
    assert source_support_for(event, "C", sources, "triad") == (0, 1, 0)
    assert source_support_for(event, "C7", sources, "full") == (0, .75, 0)


def test_missing_unknown_and_no_chord_coverage_are_distinct():
    event = T(0, 4, None)
    sources = {"chordino": [T(0, 1, "N"), T(1, 2, "X"), T(2, 3, None)]}
    pool = candidate_pool(event, sources)
    assert pool.candidates[0].semantics.state == "missing"
    assert {c.semantics.state for c in pool.candidates} == {"missing", "N", "X"}
    assert pool.source_coverage == (0, .5, 0)
    assert pool.source_state_coverage["chordino"] == {"chord": 0, "N": .25, "X": .25, "missing": .5}


def test_no_slash_and_explicit_root_bass_are_separate_candidates():
    pool = candidate_pool(T(0, 1, "C"), {"chordino": [T(0, 1, "C/C")]})
    assert len(pool.candidates) == 2
    assert pool.candidates[0].semantics.bass_pc is None
    assert pool.candidates[1].semantics.bass_pc == 0


def test_no_reference_or_positional_source_api():
    event = T(0, 1, "C")
    with pytest.raises(TypeError):
        candidate_pool(event, [[], []])
    with pytest.raises(ValueError):
        candidate_pool(event, {"reference": [event]})
    with pytest.raises(ValueError):
        candidate_pool(T(1, 1, "C"), {})
