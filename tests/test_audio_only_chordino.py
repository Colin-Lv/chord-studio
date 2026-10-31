from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from generate_audio_only_chordino_labs import (
    identity, intervals_from_points, load_audio_manifest, verify_existing,
)


def test_points_preserve_leading_uncovered_time_and_explicit_labels():
    text, coverage = intervals_from_points('0.125 "C:maj7/E"\n1.5 "N"\n2.0 "N"\n', 2.0)
    assert text == "0.125000000 1.500000000 C:maj7/E\n1.500000000 2.000000000 N\n"
    assert coverage["covered_seconds"] == 1.875
    assert not coverage["full_coverage"]
    assert not coverage["synthetic_leading_N"]


def test_full_interval_coverage_and_identical_successive_labels_merge():
    text, coverage = intervals_from_points('0 C\n1 C\n2 G7\n', 3.012345678)
    assert text == "0.000000000 2.000000000 C\n2.000000000 3.012345678 G7\n"
    assert coverage["full_coverage"]
    assert coverage["point_count"] == 3


@pytest.mark.parametrize("text", ["", "0\n", "0 ''\n", "0 C\n0 G\n", "1 C\n0 G\n", "nan C\n", "-0.1 C\n", "4 C\n"])
def test_invalid_prediction_points_fail_instead_of_fabricating_N(text):
    with pytest.raises(ValueError):
        intervals_from_points(text, 3.0)


def test_audio_only_manifest_rejects_reference_columns_without_following_them(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"not decoded by the manifest loader")
    manifest = tmp_path / "input.csv"
    manifest.write_text("id,audio,reference\ntrack,audio.wav,missing_reference.lab\n")
    with pytest.raises(ValueError, match="exactly id,audio"):
        load_audio_manifest(manifest)
    manifest.write_text("id,audio\ntrack,audio.wav\n")
    assert load_audio_manifest(manifest) == [{"id": "track", "audio": str(source.resolve())}]
    manifest.write_text("id,audio\n../escape,audio.wav\n")
    with pytest.raises(ValueError, match="Unsafe"):
        load_audio_manifest(manifest)


def test_resume_verifies_source_config_and_lab_and_never_overwrites(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"original")
    lab = tmp_path / "track.lab"
    sidecar = tmp_path / "track.provenance.json"
    assert verify_existing(lab, sidecar, identity(source), "config") is None
    lab.write_text("0 1 C\n")
    with pytest.raises(FileExistsError, match="Incomplete"):
        verify_existing(lab, sidecar, identity(source), "config")
    sidecar.write_text(json.dumps(dict(status="complete", source_audio=identity(source),
                                     configuration_sha256="config", output_lab=identity(lab))))
    assert verify_existing(lab, sidecar, identity(source), "config")["status"] == "complete"
    with pytest.raises(FileExistsError, match="mismatch"):
        verify_existing(lab, sidecar, identity(source), "other config")
    lab.write_text("0 1 G\n")
    with pytest.raises(FileExistsError, match="mismatch"):
        verify_existing(lab, sidecar, identity(source), "config")
    assert lab.read_text() == "0 1 G\n"
