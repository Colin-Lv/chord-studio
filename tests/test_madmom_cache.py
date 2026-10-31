import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from madmom_bridge import cache_matches_audio, file_identity, main, provenance_path
import madmom_bridge
import numpy as np
import pytest
from types import SimpleNamespace


def test_cache_rejects_changed_audio_lab_and_cropped_source(tmp_path):
    audio = tmp_path / "song.mp3"
    lab = tmp_path / "song.lab"
    audio.write_bytes(b"first audio")
    lab.write_text("0 10 C:maj\n")
    assert not cache_matches_audio(lab, audio)[0]
    record = {"schema": "madmom_chord_cache_v1", "audio": file_identity(audio), "lab": file_identity(lab),
              "offset": 0, "requested_duration": None, "sample_rate": 44100}
    provenance_path(lab).write_text(json.dumps(record), encoding="utf-8")
    assert cache_matches_audio(lab, audio)[0]
    audio.write_bytes(b"second audio")
    assert cache_matches_audio(lab, audio) == (False, "audio content changed")
    audio.write_bytes(b"first audio")
    lab.write_text("0 10 G:maj\n")
    assert cache_matches_audio(lab, audio) == (False, "LAB content changed")
    lab.write_text("0 10 C:maj\n")
    record["offset"] = 30
    provenance_path(lab).write_text(json.dumps(record), encoding="utf-8")
    assert not cache_matches_audio(lab, audio)[0]


def test_pretrained_frontend_rejects_wrong_sample_rate_before_decoding():
    assert main(["missing.mp3", "--sample-rate", "22050"]) == 2


def test_source_change_during_inference_does_not_publish_old_audio_labels(tmp_path, monkeypatch):
    audio, lab, python = tmp_path / "audio.mp3", tmp_path / "chords.lab", tmp_path / "python.exe"
    audio.write_bytes(b"audio A")
    python.touch()
    lab.write_text("previous result")
    monkeypatch.setattr(madmom_bridge, "WORKSPACE_TMP", tmp_path / "temp")
    monkeypatch.setattr(madmom_bridge, "decode_audio", lambda *args, **kwargs: np.zeros(44100, np.float32))
    def run(command, **kwargs):
        Path(command[-1]).write_text("0 1 C:maj\n")
        audio.write_bytes(b"audio B")
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(madmom_bridge.subprocess, "run", run)
    assert main([str(audio), "-o", str(lab), "--madmom-python", str(python)]) == 2
    assert lab.read_text() == "previous result"
    assert not provenance_path(lab).exists()
    assert not list((tmp_path / "temp").glob("*.npy"))
    assert not list(tmp_path.glob(".madmom_*"))


def test_failed_temporary_audio_write_cleans_partial_file(tmp_path, monkeypatch):
    audio, python = tmp_path / "audio.mp3", tmp_path / "python.exe"
    audio.write_bytes(b"audio A")
    python.touch()
    monkeypatch.setattr(madmom_bridge, "WORKSPACE_TMP", tmp_path / "temp")
    monkeypatch.setattr(madmom_bridge, "decode_audio", lambda *args, **kwargs: np.zeros(100, np.float32))
    def fail(*args, **kwargs):
        raise OSError("simulated failed write")
    monkeypatch.setattr(madmom_bridge.np, "save", fail)
    with pytest.raises(OSError, match="simulated"):
        main([str(audio), "--madmom-python", str(python)])
    assert not list((tmp_path / "temp").glob("*.npy"))
