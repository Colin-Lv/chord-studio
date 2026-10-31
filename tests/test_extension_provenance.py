import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from extension_provenance import EXTENSION_SCHEMA, read_extension_provenance


def fixture_files(tmp_path):
    audio, lab = tmp_path / "song.wav", tmp_path / "derived.lab"
    audio.write_bytes(b"original audio")
    lab.write_text("0 1 Cmaj7\n")
    sidecar = lab.with_name(lab.name + ".provenance.json")
    data = dict(schema=EXTENSION_SCHEMA,
                audio=dict(sha256=hashlib.sha256(audio.read_bytes()).hexdigest()),
                output_lab=dict(sha256=hashlib.sha256(lab.read_bytes()).hexdigest()),
                model=dict(path="must_not_load_or_open.joblib", sha256="a"*64),
                upstream_lab=dict(path="must_not_open.lab", sha256="b"*64),
                feature_schema="fixed_root_triad_seventh_centered_fast_v2", threshold=.97, change_count=1)
    sidecar.write_text(json.dumps(data))
    return audio, lab, sidecar


def test_matching_files_are_verified_without_opening_declared_model_paths(tmp_path):
    audio, lab, sidecar = fixture_files(tmp_path)
    result = read_extension_provenance(audio, lab)
    assert result["source_content_verified"]
    assert result["sidecar"]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert result["model"]["sha256"] == "a"*64
    assert result["upstream_lab"]["sha256"] == "b"*64
    assert result["threshold"] == .97 and result["change_count"] == 1
    assert not result["declared_model_and_upstream_identities_verified"]


def test_different_audio_keeps_declared_metadata_but_fails_verification(tmp_path):
    audio, lab, _ = fixture_files(tmp_path)
    audio.write_bytes(b"different recording")
    result = read_extension_provenance(audio, lab)
    assert not result["source_content_verified"]
    assert "audio SHA256" in result["verification"]
    assert result["threshold"] == .97


def test_modified_output_lab_fails_verification(tmp_path):
    audio, lab, _ = fixture_files(tmp_path)
    lab.write_text("0 1 Dmaj7\n")
    result = read_extension_provenance(audio, lab)
    assert not result["source_content_verified"]
    assert "output LAB SHA256" in result["verification"]


def test_bad_json_records_failure_without_blocking_explicit_lab(tmp_path):
    audio, lab, sidecar = fixture_files(tmp_path)
    sidecar.write_text("{ broken")
    result = read_extension_provenance(audio, lab)
    assert not result["source_content_verified"]
    assert "Cannot read extension provenance" in result["verification"]


def test_raw_cache_sidecar_is_not_reclassified_as_an_extension(tmp_path):
    audio, lab, sidecar = fixture_files(tmp_path)
    sidecar.write_text(json.dumps(dict(schema="madmom_chord_cache_v1")))
    assert read_extension_provenance(audio, lab) is None
