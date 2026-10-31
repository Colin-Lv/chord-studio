import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from compare_evaluation_reports import compare
from replay_fusion_evaluation import refresh_structure, validate_frozen_cnn_evidence


def test_functional_key_evidence_cannot_be_reported_as_cnn():
    payload = dict(key_source="madmom_cnn_key", beat_source="madmom", key="C", source="song.wav")
    validate_frozen_cnn_evidence(payload, copy.deepcopy(payload))
    with pytest.raises(ValueError, match="automatic"):
        validate_frozen_cnn_evidence({**payload, "key_source": "functional"}, payload)
    with pytest.raises(ValueError, match="provided"):
        validate_frozen_cnn_evidence({**payload, "fusion_provenance": {"key_source": "provided"}}, payload)


def test_lightweight_refresh_preserves_manual_section_boundaries(monkeypatch):
    import chord_recognizer
    import song_structure
    monkeypatch.setattr(chord_recognizer, "decode_audio", lambda *a, **kw: np.zeros(22050 * 8))
    monkeypatch.setattr(song_structure.Path, "is_file", lambda self: True)
    monkeypatch.setattr(song_structure, "extract_bar_features", lambda y, sr, bars: {
        "chroma": np.tile([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], (4, 1)),
        "timbre": np.ones((4, 13)), "rhythm": np.ones((4, 8)), "energy": np.full((4, 1), -20.),
    })
    payload = dict(source="song.wav", duration=8, bars=[dict(bar=i + 1, start=2 * i, end=2 * i + 2,
        spans=[dict(start=2 * i, end=2 * i + 2, chord="C")]) for i in range(4)])
    manual = [dict(start_bar=1, end_bar=2, start=0, end=3.25, name="人工主歌", role="verse"),
              dict(start_bar=2, end_bar=4, start=3.25, end=8, name="人工副歌", role="chorus")]
    previous = dict(structure=dict(method="bar_multifeature_novelty", manual_overrides=manual))
    original = copy.deepcopy(previous)
    refresh_structure(payload, previous)
    assert [section["name"] for section in payload["structure"]["sections"]] == ["人工主歌", "人工副歌"]
    assert payload["structure"]["sections"][1]["start"] == 3.25
    assert all(section["source"] == "manual" for section in payload["structure"]["sections"])
    assert previous == original


def report():
    scores = dict(root_accuracy=.8, triad_accuracy=.7, tetrad_accuracy=.6,
                  tetrad_inv_accuracy=.5, prediction_coverage=1., evaluated_seconds=10., non_no_chord_seconds=9.)
    return dict(tracks=[dict(id=str(i), audio=f"song{i}.wav", status="complete", chord_metrics=scores.copy()) for i in range(2)],
                scoring_parameters={"chord_time_integration": "exact"}, annotation_alignment="raw",
                manifest_signature={"sha256": "fixed"})


@pytest.mark.parametrize("field,value", [("annotation_alignment", "oracle aligned"),
    ("scoring_parameters", {"chord_time_integration": "sampled"}), ("manifest_signature", {"sha256": "different"})])
def test_report_comparison_rejects_different_reference_protocol(tmp_path, field, value):
    baseline, final = report(), report()
    final[field] = value
    paths = [tmp_path / "before.json", tmp_path / "after.json"]
    for path, data in zip(paths, (baseline, final)):
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        compare(*paths, resamples=1000, seed=1)
