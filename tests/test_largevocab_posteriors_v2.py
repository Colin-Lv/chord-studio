"""Semantic contracts for the opt-in six-head export (no acoustic inference)."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import largevocab_posteriors_v2 as posterior
import generate_audio_only_largevocab_labs_v2 as bridge


def evidence():
    # Two complete native frame cells and half of the third.
    count = 1280
    heads = {name: np.eye(size, dtype=np.float32)[np.arange(3) % size]
             for name, size in posterior.HEAD_DIMENSIONS.items()}
    identity = {"source_audio": {"path": "source.wav", "sha256": "a" * 64},
                "recognizer": {"models": [{"sha256": "b" * 64}]}}
    return posterior.make_evidence(heads, identity, decoded_samples=count)


def test_actual_vendor_class_semantics():
    labels = posterior.HEAD_LABELS
    assert [len(labels[name]) for name in posterior.HEAD_NAMES] == [73, 13, 4, 4, 3, 3]
    assert labels["triad"][0] == "N"
    assert labels["triad"][1] == "C:maj"
    assert labels["triad"][12] == "B:maj"
    assert labels["triad"][13] == "C:min"
    assert labels["triad"][72] == "B:aug"
    assert labels["bass"][0] == "none"
    assert labels["bass"][1] == "C" and labels["bass"][12] == "B"
    assert labels["seventh"] == ["none", "7", "b7", "bb7"]
    assert labels["ninth"] == ["none", "9", "#9", "b9"]
    assert labels["eleventh"] == ["none", "11", "#11"]
    assert labels["thirteenth"] == ["none", "13", "b13"]


def test_roundtrip_and_stale_identity_rejected(tmp_path):
    data = evidence()
    target = tmp_path / "evidence.npz"
    posterior.save_posteriors(target, data)
    loaded = posterior.load_posteriors(target, expected_identity=data.metadata["identity"])
    np.testing.assert_array_equal(loaded.frame_samples, [0, 512, 1024])
    for name in posterior.HEAD_NAMES:
        np.testing.assert_array_equal(loaded.probabilities[name], data.probabilities[name])
    stale = copy.deepcopy(data.metadata["identity"])
    stale["source_audio"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="identity"):
        posterior.load_posteriors(target, expected_identity=stale)
    with pytest.raises(FileExistsError):
        posterior.save_posteriors(target, data)


@pytest.mark.parametrize("damage", ["nan", "negative", "normalization", "shape", "clock", "classes"])
def test_invalid_cache_is_rejected(tmp_path, damage):
    data = evidence()
    arrays = {name: values.copy() for name, values in data.probabilities.items()}
    samples = data.frame_samples.copy()
    metadata = copy.deepcopy(data.metadata)
    if damage == "nan": arrays["seventh"][0, 0] = np.nan
    if damage == "negative": arrays["bass"][0, :2] = [-0.1, 1.1]
    if damage == "normalization": arrays["triad"][0] = 0
    if damage == "shape": arrays["thirteenth"] = np.zeros((3, 4), dtype=np.float32)
    if damage == "clock": samples[1] += 1
    if damage == "classes": metadata["head_labels"]["triad"][1] = "C:min"
    target = tmp_path / "bad.npz"
    np.savez(target, **arrays, frame_samples=samples,
             metadata_json=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError):
        posterior.load_posteriors(target)


def test_aggregation_uses_exact_old_boundaries_and_separate_source_identity():
    data = evidence()
    hop = 512 / 22050
    intervals = np.array([[0.5 * hop, 1.5 * hop], [2 * hop, 3 * hop], [3 * hop, 4 * hop]])
    result = posterior.aggregate_posteriors(data, intervals, anchor_source={"model": "raw_madmom"})
    np.testing.assert_array_equal(result.intervals, intervals)
    np.testing.assert_allclose(result.probabilities["seventh"][0], [.5, .5, 0, 0])
    np.testing.assert_allclose(result.probabilities["seventh"][1], [0, 0, 1, 0])
    np.testing.assert_allclose(result.coverage, [1, .5, 0])
    assert np.isnan(result.probabilities["triad"][2]).all()
    assert result.metadata["source_identity"] == data.metadata["identity"]
    assert result.metadata["anchor_source"] == {"model": "raw_madmom"}
    assert data.metadata["identity"]["source_audio"]["path"] == "source.wav"


def test_empty_audio_and_frame_count_are_explicit():
    data = evidence()
    with pytest.raises(ValueError, match="empty"):
        posterior.make_evidence(data.probabilities, {}, decoded_samples=0)
    with pytest.raises(ValueError, match="frame"):
        posterior.make_evidence(data.probabilities, {}, decoded_samples=2048)


def test_decoder_tap_passes_same_objects_and_does_not_change_lab():
    data = evidence()
    arrays = list(data.probabilities.values())
    class Decoder:
        def decode_to_chordlab(self, entry, values, *args, **kwargs):
            assert values is arrays
            assert entry.prop.hop_length == 512
            return [(0., 1280 / 22050, "C:maj/3")]
    decoder = Decoder()
    entry = SimpleNamespace(prop=SimpleNamespace(sr=22050, hop_length=512))
    baseline = decoder.decode_to_chordlab(entry, arrays, False, use_beats=False)
    tap = bridge.PosteriorDecoderTap(decoder)
    actual = tap.decode_to_chordlab(entry, arrays, False, use_beats=False)
    assert baseline == actual
    for name, original in zip(posterior.HEAD_NAMES, arrays):
        np.testing.assert_array_equal(tap.probabilities[name], original)
        assert tap.probabilities[name] is not original


def test_synthetic_original_predict_export_on_off_equal(monkeypatch):
    heads = evidence().probabilities
    class FakeDecoder:
        def decode_to_chordlab(self, entry, values, *args, **kwargs):
            return [(0., 1280 / 22050, "C:maj/3")]
    def original_predict(self, audio, duration=None):
        arrays = list(heads.values())
        rows = self.decoder.decode_to_chordlab(SimpleNamespace(prop=SimpleNamespace(sr=22050, hop_length=512)), arrays, False)
        return rows, {"decoded_samples": 1280, "cqt_frames": 3, "audio_seconds": 1280 / 22050}
    monkeypatch.setattr(bridge.LargeVocabularyRecognizer, "predict", original_predict)
    model = object.__new__(bridge.LargeVocabularyRecognizerV2)
    original_decoder = model.decoder = FakeDecoder()
    model.identity = {"models": []}
    baseline, _ = model.predict("fake.wav")
    exported, _, captured = model.predict_with_posteriors("fake.wav", identity={"source": "fake"})
    assert baseline == exported
    assert model.decoder is original_decoder
    np.testing.assert_array_equal(captured.probabilities["triad"], heads["triad"])
    def failing(*args, **kwargs): raise RuntimeError("synthetic failure")
    monkeypatch.setattr(bridge.LargeVocabularyRecognizer, "predict", failing)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        model.predict_with_posteriors("fake.wav", identity={})
    assert model.decoder is original_decoder


class SyntheticRecognizer(bridge.LargeVocabularyRecognizerV2):
    def __init__(self, *args):
        self.identity = {"schema": "synthetic_test_only", "models": []}

    def predict(self, audio, duration=None):
        if Path(audio).name == "bad.wav":
            raise ValueError("synthetic empty/invalid audio")
        return [(0., 1280 / 22050, "C:maj/3")], {
            "decoded_samples": 1280, "cqt_frames": 3, "audio_seconds": 1280 / 22050,
            "intervals": 1, "requested_duration_seconds": duration}

    def predict_with_posteriors(self, audio, duration=None, *, identity):
        rows, statistics = self.predict(audio, duration)
        return rows, statistics, posterior.make_evidence(evidence().probabilities, identity, decoded_samples=1280)


def test_synthetic_runner_lab_equivalence_cache_reuse_and_staleness(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "LargeVocabularyRecognizerV2", SyntheticRecognizer)
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"synthetic audio identity only")
    disabled, enabled = tmp_path / "off", tmp_path / "on"
    assert bridge.main(["--audio", str(audio), "--output-dir", str(disabled)]) == 0
    assert bridge.main(["--audio", str(audio), "--output-dir", str(enabled), "--export-posteriors"]) == 0
    assert (disabled / "labs/audio.lab").read_bytes() == (enabled / "labs/audio.lab").read_bytes()
    assert bridge.main(["--audio", str(audio), "--output-dir", str(enabled), "--export-posteriors"]) == 0
    report = json.loads((enabled / "report.json").read_text())
    assert report["songs"][0]["status"] == "reused_verified_output"
    audio.write_bytes(b"changed content at the same path")
    assert bridge.main(["--audio", str(audio), "--output-dir", str(enabled), "--export-posteriors"]) == 1
    report = json.loads((enabled / "report.json").read_text())
    assert report["status"] == "incomplete" and "identity" in report["failures"][0]["error"]


def test_partial_batch_failure_is_not_reported_complete(tmp_path, monkeypatch):
    import csv
    monkeypatch.setattr(bridge, "LargeVocabularyRecognizerV2", SyntheticRecognizer)
    manifest = tmp_path / "audio.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "audio"])
        for sid in ("good", "bad"):
            audio = tmp_path / (sid + ".wav")
            audio.write_bytes(b"synthetic fixture")
            writer.writerow([sid, str(audio.resolve())])
    output = tmp_path / "output"
    assert bridge.main(["--manifest", str(manifest), "--output-dir", str(output), "--export-posteriors"]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "incomplete" and report["completed_count"] == 1
    assert report["failures"][0]["id"] == "bad"
    assert not (output / "labs/bad.provenance.json").exists()
    assert not (output / ".v2-extraction.lock").exists()


def test_incomplete_bundle_and_v1_cache_are_not_reused(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"synthetic fixture")
    model = SyntheticRecognizer()
    output = tmp_path / "output"
    (output / "labs").mkdir(parents=True)
    (output / "labs/audio.lab").write_text("0 1 C\n")
    with pytest.raises(ValueError, match="Incomplete"):
        bridge.run_item(model, "audio", audio, output, export_posteriors=True)
    (output / "labs/audio.provenance.json").write_text(json.dumps({"schema": "v1"}))
    with pytest.raises(ValueError, match="v1 cache"):
        bridge.run_item(model, "audio", audio, output)


def test_failed_npz_staging_publishes_no_completed_lab(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"synthetic fixture")
    output = tmp_path / "output"
    def fail(*args): raise OSError("synthetic disk full")
    monkeypatch.setattr(bridge, "save_posteriors", fail)
    with pytest.raises(OSError, match="disk full"):
        bridge.run_item(SyntheticRecognizer(), "audio", audio, output, export_posteriors=True)
    assert not (output / "labs/audio.lab").exists()
    assert not (output / "labs/audio.provenance.json").exists()
