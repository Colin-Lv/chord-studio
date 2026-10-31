import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from chord_recognizer import build_templates
from chord_studio import chord_member_pcs, chord_interval_pcs, should_show_slash_bass, template_roots_and_members, chord_display_name, chord_to_roman, chord_to_roman_display


@pytest.mark.parametrize("chord,expected", [
    ("Cdim7", {0, 3, 6, 9}), ("Cm7b5", {0, 3, 6, 10}),
    ("CmMaj7", {0, 3, 7, 11}), ("Cmadd9", {0, 2, 3, 7}),
    ("Cadd9", {0, 2, 4, 7}), ("C9sus4", {0, 2, 5, 7, 10}),
    ("F#dim7", {0, 3, 6, 9}), ("Dbmaj9", {0, 1, 3, 5, 8}),
])
def test_chord_tones(chord, expected):
    assert chord_member_pcs(chord) == expected


def test_all_templates_and_bass_scoring_agree_on_pitch_content():
    for notation in ("sharps", "flats"):
        templates = build_templates("rich", notation)
        _, _, members = template_roots_and_members([t.name for t in templates])
        for template, member in zip(templates, members):
            assert chord_member_pcs(template.name) == set(template.pitch_classes)
            assert set(np.flatnonzero(member)) == set(template.pitch_classes)


def test_extension_does_not_invent_or_omit_slash_bass():
    assert should_show_slash_bass("Cdim7", "A", .95, .95)
    assert not should_show_slash_bass("Cdim7", "Bb", .95, .95)
    assert not should_show_slash_bass("Cmadd9", "Bb", .95, .95)
    assert should_show_slash_bass("CmMaj7", "B", .95, .95)


def test_structural_tones_of_diminished_and_augmented_chords():
    assert chord_interval_pcs("Cdim") == (0, 3, 6)
    assert chord_interval_pcs("Caug") == (0, 4, 8)


@pytest.mark.parametrize("chord,key,expected", [("C/E", "G", "IV/VI"), ("F#m7/A", "A", "VIm7/I"),
                                               ("C/D", "C", "I/II")])
def test_explicit_inversion_or_pedal_survives_conflicting_bass_estimate(chord, key, expected):
    assert chord_display_name(chord, "G", .99, .99) == chord
    assert chord_to_roman(chord, key) == expected
    assert chord_to_roman_display(chord, key, "G", .99, .99) == expected
