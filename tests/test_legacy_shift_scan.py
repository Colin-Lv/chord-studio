"""Prediction-shift and reference-shift audits require opposite conventions."""
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import build_aligned_manifest_from_shift_scan as builder


@pytest.mark.parametrize("target,expected", [(None, 1.), ("prediction", 1.), ("reference", 5.)])
def test_shift_target_is_explicit_and_legacy_default_is_preserved(tmp_path, monkeypatch, target, expected):
    reference = tmp_path / "song.lab"
    reference.write_text("3 5 C:maj\n", encoding="utf-8")
    (tmp_path / "song.beats").write_text("3 1\n4 2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio,prediction,chord_reference,beat_reference\nsong,song.wav,old.json,song.lab,song.beats\n")
    scan = tmp_path / "scan.csv"
    scan.write_text("id,best_triad,best_time_shift\nsong,0.8,2\n")
    accepted = tmp_path / "accepted.csv"
    command = ["builder", str(manifest), str(scan), "--output-accepted", str(accepted),
               "--output-rejected", str(tmp_path / "rejected.csv"), "--aligned-dir", str(tmp_path / "aligned")]
    if target:
        command += ["--scan-target", target]
    monkeypatch.setattr(sys, "argv", command)
    assert builder.main() == 0
    row = next(csv.DictReader(accepted.open(encoding="utf-8-sig")))
    assert float((tmp_path / row["chord_reference"]).read_text().split()[0]) == expected
    assert row["alignment_scan_target"] == (target or "prediction")
    assert row["alignment_method"] == "prediction_optimized_time_shift"
    assert reference.read_text() == "3 5 C:maj\n"
