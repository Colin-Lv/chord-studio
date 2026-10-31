"""V2 file-boundary tests using temporary text only; no audio/cache/model work."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from candidate_data_v2 import read_annotation, read_source, standard_reference
from candidate_targets_v2 import annotation_coverage, candidate_targets
from chord_semantics_v2 import parse_chord
from multisource_candidates_v2 import TimedChord, candidate_pool


def write_text(tmp_path, suffix, text):
    path = tmp_path / ("input" + suffix)
    path.write_text(text, encoding="utf-8")
    return path


def triples(rows):
    return [(row.start, row.end, row.chord) for row in rows]


def test_lab_source_keeps_raw_spelling_unknown_and_missing(tmp_path):
    path = write_text(tmp_path, ".lab", "# source observation\n0 1 N\n1 2 X\n2 3 H:maj\n3 4 C:maj/3\n4 5\n")
    before = path.read_bytes()
    rows, metadata = read_source(path)
    assert triples(rows) == [(0, 1, "N"), (1, 2, "X"), (2, 3, "H:maj"),
                             (3, 4, "C:maj/3"), (4, 5, None)]
    assert [parse_chord(row.chord).state for row in rows] == ["N", "X", "X", "chord", "missing"]
    assert metadata["rounding_overlap_repairs"] == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("wrapped", [False, True])
def test_json_source_preserves_missing_null_blank_and_invalid_labels(tmp_path, wrapped):
    data = [{"start": 4, "end": 5, "chord": "Cb:maj/Eb"},
            {"start": 0, "end": 1, "chord": "X"},
            {"start": 1, "end": 2, "chord": "H:maj"},
            {"start": 2, "end": 3, "chord": None},
            {"start": 3, "end": 4},
            {"start": 5, "end": 6, "chord": ""}]
    path = write_text(tmp_path, ".json", json.dumps({"events": data} if wrapped else data))
    rows, _ = read_source(path)
    assert [row.chord for row in rows] == ["X", "H:maj", None, None, "Cb:maj/Eb", ""]
    assert triples(rows)[4] == (4, 5, "Cb:maj/Eb")


def test_csv_source_keeps_blank_label_and_parses_timestamp(tmp_path):
    path = write_text(tmp_path, ".csv", "start,end,label\n00:00,00:01,X\n00:01,00:02,H:maj\n00:02,00:03,\n")
    rows, _ = read_source(path)
    assert triples(rows) == [(0, 1, "X"), (1, 2, "H:maj"), (2, 3, "")]
    assert parse_chord(rows[-1].chord).state == "missing"


def test_auxiliary_duplicates_and_overlaps_reach_union_support(tmp_path):
    path = write_text(tmp_path, ".lab", "0 3 C9\n1 4 C:7(9)\n0 3 C9\n")
    rows, metadata = read_source(path)
    assert triples(rows) == [(0, 3, "C9"), (0, 3, "C9"), (1, 4, "C:7(9)")]
    assert metadata["rounding_overlap_repairs"] == 0
    pool = candidate_pool(TimedChord(0, 4, "C7"), {"chordino": rows})
    ninth = next(candidate for candidate in pool.candidates if candidate.semantics.extension(9))
    assert ninth.source_support == (0, 1, 0)
    assert pool.source_coverage == (0, 1, 0)


@pytest.mark.parametrize("text", ["0 1 C\n0 1 C\n", "0 1.001 C\n1 2 G\n", "0 2 C\n1 3 G\n"])
def test_raw_anchors_reject_duplicates_and_even_small_overlap(tmp_path, text):
    path = write_text(tmp_path, ".lab", text)
    with pytest.raises(ValueError, match="Raw anchors must be disjoint"):
        read_source(path, anchor=True)
    # The exact same rows are permitted for an auxiliary observation channel.
    assert len(read_source(path, anchor=False)[0]) == 2


def test_raw_gaps_remain_gaps_and_labels_are_not_filled(tmp_path):
    path = write_text(tmp_path, ".lab", "2 3 X\n0 1 C\n")
    rows, _ = read_source(path, anchor=True)
    assert triples(rows) == [(0, 1, "C"), (2, 3, "X")]


@pytest.mark.parametrize("text", ["-1 0 C\n", "0 0 C\n", "2 1 C\n", "nan 1 C\n", "0 inf C\n"])
def test_invalid_source_times_are_rejected_for_any_channel(tmp_path, text):
    path = write_text(tmp_path, ".lab", text)
    with pytest.raises(ValueError):
        read_source(path)


def test_reference_json_preserves_masks_into_targets(tmp_path):
    data = {"annotations": [{"start": 0, "end": 2, "chord": "C9/E",
                             "valid_fields": ["root", "bass", "ext9"],
                             "field_mask": {"bass": False}}]}
    path = write_text(tmp_path, ".json", json.dumps(data))
    rows, metadata = read_annotation(path)
    assert rows[0].valid_fields == ("root", "bass", "ext9")
    assert rows[0].field_mask == {"bass": False}
    assert metadata["rounding_overlap_repairs"] == 0
    result = candidate_targets(TimedChord(0, 2, "C"), ["C9"], rows)
    assert result.mask[0].tolist() == [True, False, False, False, True, False, False]
    assert result.valid_seconds[0].tolist() == [2, 0, 0, 0, 2, 0, 0]
    assert result.values[0, [0, 4]].tolist() == [1, 1]


def test_reference_json_X_and_missing_are_masked_not_silence(tmp_path):
    path = write_text(tmp_path, ".json", json.dumps([
        {"start": 0, "end": 1, "chord": "X"},
        {"start": 1, "end": 2, "chord": None},
        {"start": 2, "end": 3, "chord": "H:maj"}]))
    rows, _ = read_annotation(path)
    result = candidate_targets(TimedChord(0, 3, "C"), ["N"], rows)
    assert not result.mask.any()
    assert np.isnan(result.values).all()
    standard = standard_reference(rows)
    assert [row.chord for row in standard] == ["X", "X", "H:maj"]


def test_masked_reference_json_does_not_silently_retime_overlaps(tmp_path):
    path = write_text(tmp_path, ".json", json.dumps([
        {"start": 0, "end": 1.001, "chord": "C", "field_mask": {"bass": False}},
        {"start": 1, "end": 2, "chord": "G"}]))
    with pytest.raises(ValueError, match="nonoverlapping"):
        read_annotation(path)


def test_Point_is_only_filtered_from_reference_not_source(tmp_path):
    path = write_text(tmp_path, ".lab", "0.432 86400 Point\n0 1 N\n1 2 C:maj/3\n2 3 X\n")
    refs, _ = read_annotation(path)
    assert triples(refs) == [(0, 1, "N"), (1, 2, "C:maj/3"), (2, 3, "X")]
    assert annotation_coverage(refs)["span_seconds"] == 3
    source, _ = read_source(path)
    assert any(row.chord == "Point" for row in source)


def test_reference_rounding_repair_is_explicit_and_original_file_unchanged(tmp_path):
    path = write_text(tmp_path, ".lab", "0 1.001 C\n1 2 G\n")
    original = path.read_bytes()
    rows, metadata = read_annotation(path)
    assert triples(rows) == [(0, 1, "C"), (1, 2, "G")]
    assert metadata["rounding_overlap_repairs"] == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("suffix,text", [
    (".lab", "0 1 C\n1 3 H:maj\n"),
    (".csv", "start,end,chord\n0,1,C\n1,3,H:maj\n")])
def test_unknown_reference_tail_remains_in_annotation_coverage(tmp_path, suffix, text):
    path = write_text(tmp_path, suffix, text)
    rows, _ = read_annotation(path)
    assert triples(rows) == [(0, 1, "C"), (1, 3, "H:maj")]
    inventory = annotation_coverage(rows)
    assert inventory["span_seconds"] == 3
    assert inventory["valid_seconds"]["root"] == 1
    assert inventory["coverage"]["root"] == pytest.approx(1 / 3)
