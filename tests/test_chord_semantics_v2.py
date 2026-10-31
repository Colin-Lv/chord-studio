"""Hand-written music-semantic expectations, independent of v2 implementation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from chord_semantics_v2 import parse_chord, same_chord


def pcs(chord, field="chord_bitmap"):
    return {i for i, present in enumerate(getattr(parse_chord(chord), field)) if present}


@pytest.mark.parametrize("label,expected", [
    ("C", {0, 4, 7}), ("Cm", {0, 3, 7}), ("C7", {0, 4, 7, 10}),
    ("C9", {0, 2, 4, 7, 10}), ("C11", {0, 2, 4, 5, 7, 10}),
    ("C13", {0, 2, 4, 5, 7, 9, 10}), ("Cadd9", {0, 2, 4, 7}),
    ("Cmaj7", {0, 4, 7, 11}), ("Cmaj9", {0, 2, 4, 7, 11}),
    ("CmMaj7", {0, 3, 7, 11}), ("C:7(b9)", {0, 1, 4, 7, 10}),
    ("C:min(*5)", {0, 3}), ("C:(b3,b5)", {0, 3, 6})])
def test_full_pitch_sets(label, expected):
    assert pcs(label) == expected


@pytest.mark.parametrize("left,right", [("C9", "C:7(9)"), ("Cadd9", "C:maj(9)"),
                                        ("CM7", "C:maj7"), ("Cmaj9", "C:maj7(9)"),
                                        ("C/E", "C:maj/3"), ("Cb/Eb", "B:maj/3"),
                                        ("C13", "C:7(9,11,13)")])
def test_equivalent_notations_merge(left, right):
    assert parse_chord(left).identity == parse_chord(right).identity


@pytest.mark.parametrize("left,right", [("C7", "C9"), ("C9", "C11"), ("C11", "C13"),
                                        ("C", "Cadd9"), ("Cmaj7", "Cmaj9"),
                                        ("C", "C/C"), ("C/E", "C/G"),
                                        ("Cadd9", "C/D"), ("C:maj(2)", "Cadd9")])
def test_distinct_full_semantics_are_not_merged(left, right):
    assert parse_chord(left).identity != parse_chord(right).identity


def test_pedal_is_separate_from_chord_tone_and_extension():
    value = parse_chord("C/D")
    assert pcs("C/D") == {0, 4, 7}
    assert pcs("C/D", "full_bitmap") == {0, 2, 4, 7}
    assert value.explicit_bass and value.bass_pc == 2 and value.bass_relative == 2
    assert value.extension(9) == () and value.mentioned_extensions == ()
    assert parse_chord("C").bass_pc is None
    assert parse_chord("C").bass_relative is None


def test_raw_spelling_and_structured_omission_are_retained():
    value = parse_chord("Db:7(*9,#11)/b3")
    assert value.raw_label == "Db:7(*9,#11)/b3"
    assert [(d.number, d.alteration, d.omitted) for d in value.degrees] == [(9, 0, True), (11, 1, False)]
    assert value.extension(9) == () and value.extension(11) == (1,)
    assert value.mentioned_extensions == (9, 11)
    assert value.bass_pc == 4


def test_explicit_omission_overrides_quality_implied_extension():
    value = parse_chord("C:9(*9)")
    assert pcs("C:9(*9)") == {0, 4, 7, 10}
    assert value.extension(9) == ()
    assert value.mentioned_extensions == (9,)


@pytest.mark.parametrize("label,state", [(None, "missing"), ("", "missing"), ("N", "N"),
                                        ("X", "X"), ("H:maj", "X"), ("C::maj", "X"),
                                        ("C:maj//E", "X"), ("not-a-chord", "X")])
def test_absence_unknown_and_malformed_are_not_silence(label, state):
    value = parse_chord(label)
    assert value.state == state
    assert value.raw_label == label
    assert value.bass_pc is None


def test_harmonic_lower_heads_are_separate_from_extensions_and_bass():
    assert same_chord(parse_chord("C7"), parse_chord("C13"), "tetrad")
    assert same_chord(parse_chord("C"), parse_chord("C/D"), "triad")
    assert not same_chord(parse_chord("C7"), parse_chord("Cmaj7"), "tetrad")
    assert parse_chord("C13").extension(9) == (0,)
    assert parse_chord("C13").extension(11) == (0,)
    assert parse_chord("C13").extension(13) == (0,)
