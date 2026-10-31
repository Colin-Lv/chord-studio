"""Preserve label protocols and reject silently excluded rich references."""
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from benchmark_guitarset_madmom import checked_reference, selection_rows, summarize
from evaluate_chords import evaluate, TimedChord


def test_performed_and_instructed_are_scored_as_distinct_protocols():
    prediction = [TimedChord(0, 4, "C:maj")]
    performed = evaluate([TimedChord(0, 4, "C:maj7")], prediction)
    instructed = evaluate([TimedChord(0, 4, "C:maj")], prediction)
    result = summarize([dict(status="ok", metrics=dict(performed=performed, instructed=instructed))])
    assert result["performed"]["macro_triad_accuracy"] == 1
    assert result["performed"]["macro_tetrad_accuracy"] == 0
    assert result["instructed"]["macro_tetrad_accuracy"] == 1


def test_rich_labels_supported_but_invalid_labels_cannot_disappear(tmp_path):
    path = tmp_path / "reference.lab"
    path.write_text("0 2 G#:maj6(2,b5,*5)/1\n2 4 D#:sus2(7)/1\n")
    assert len(checked_reference(path)) == 2
    path.write_text("0 2 C:not_a_quality\n")
    with pytest.raises(ValueError, match="Unsupported labels"):
        checked_reference(path)
    path.write_text("0 2 C:maj\n2 4 invalid_label\n")
    with pytest.raises(ValueError, match="silently skipped"):
        checked_reference(path)


def test_selection_rejects_completed_subset(tmp_path):
    path = tmp_path / "subset.csv"
    from benchmark_guitarset_madmom import SELECTION_FIELDS
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerow({key: "" for key in SELECTION_FIELDS})
    with pytest.raises(ValueError, match="exactly 10"):
        selection_rows(path)
