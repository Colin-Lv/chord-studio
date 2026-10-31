import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from build_model_disagreements import compare_timelines, difference_kind


def analysis(events, duration=4, offset=0):
    return {"duration": duration, "audio_offset": offset,
            "events": [{"start": a, "end": b, "chord": chord} for a, b, chord in events]}


def test_offbeat_intersections_preserved_without_mutating_sources():
    first = analysis([(0, 1.125, "C"), (1.125, 4, "G")])
    second = analysis([(0, 1.375, "C"), (1.375, 4, "G7")])
    before = copy.deepcopy((first, second))
    result = compare_timelines(first, second)
    assert result["total_count"] == 2
    assert result["all_segments"][0] == dict(start=1.125, end=1.375, primary_label="G", comparison_label="C", kind="root")
    assert result["segments"] == [dict(start=1.375, end=4, primary_label="G", comparison_label="G7", kind="quality")]
    assert (first, second) == before


def test_gaps_are_not_confirmed_no_chord():
    result = compare_timelines(analysis([(1, 4, "C")]), analysis([(0, 1, "N"), (1, 4, "C")]))
    assert result["segments"] == [dict(start=0, end=1, primary_label=None, comparison_label="N", kind="coverage")]


def test_enharmonic_quality_and_explicit_bass():
    assert difference_kind("C#maj7", "Db:maj7") is None
    assert difference_kind("C/E", "C/3") is None
    assert difference_kind("C/E", "C") == "bass"
    assert difference_kind("Cadd9", "C") == "quality"
    assert difference_kind("unsupported", "other unsupported") == "label"


@pytest.mark.parametrize("offset", [math.nan, math.inf, 1])
def test_mismatched_or_nonfinite_offsets_rejected(offset):
    with pytest.raises(ValueError, match="offset"):
        compare_timelines(analysis([(0, 4, "C")]), analysis([(0, 4, "G")], offset=offset))


def test_overlapping_predictions_require_resolution():
    with pytest.raises(ValueError, match="Overlapping"):
        compare_timelines(analysis([(0, 3, "C"), (2, 4, "G")]), analysis([(0, 4, "G")]))


def test_cropped_playback_uses_original_audio_offset():
    cropped = analysis([(0, 4, "C")])
    cropped["original_offset_seconds"] = 30
    with pytest.raises(ValueError, match="offset"):
        compare_timelines(analysis([(0, 4, "C")]), cropped)
    assert compare_timelines(analysis([(0, 4, "C")], offset=30), cropped)["segments"] == []
