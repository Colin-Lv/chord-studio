import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from acoustic_chord_model import candidate_features, decode_emissions, emission_scores


def test_candidates_are_transposition_equivariant():
    rng = np.random.default_rng(1)
    chroma, bass = rng.random((5, 12)), rng.random((5, 12))
    before = candidate_features(chroma, bass).reshape(5, 12, 2, -1)
    for shift in range(12):
        after = candidate_features(np.roll(chroma, shift, axis=1), np.roll(bass, shift, axis=1)).reshape(5, 12, 2, -1)
        np.testing.assert_allclose(after, np.roll(before, shift, axis=1), atol=1e-6)


def test_zero_transition_leaves_strong_changes_in_place():
    scores = np.full((4, 24), -10., dtype=np.float32)
    scores[np.arange(4), [0, 5, 18, 0]] = 0.
    assert decode_emissions(scores, 0).tolist() == [0, 5, 18, 0]


def test_duration_independent_decoder_can_sustain_ambiguous_chord():
    scores = np.full((3, 24), -10., dtype=np.float32)
    scores[:, 0] = 0.
    scores[1, 2] = 0.1
    assert decode_emissions(scores, .5).tolist() == [0, 0, 0]


def test_schema_mismatch_fails_before_using_model():
    with pytest.raises(ValueError, match="schema"):
        emission_scores({"feature_schema": "legacy"}, np.ones((1, 12)), np.ones((1, 12)))


def test_empty_features_have_stable_shape():
    assert candidate_features(np.empty((0, 12)), np.empty((0, 12))).shape == (0, 87)


def test_training_labels_accept_display_notation_and_validate_timeline(tmp_path):
    from train_acoustic_chords import reference_rows
    reference = tmp_path / "reference.lab"
    reference.write_text("0 1.001 Cm\n1 2 G:maj7\n2 3 N\n", encoding="utf-8")
    assert reference_rows(reference) == [(0., 1., 1), (1., 2., 14)]
    reference.write_text("0 1.5 C:maj\n1 2 G:maj\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        reference_rows(reference)


def test_training_cache_migration_reuses_spectra_but_rebuilds_targets(tmp_path, monkeypatch):
    import hashlib
    import json
    import chord_studio as studio
    from train_acoustic_chords import song_data
    from acoustic_chord_model import FEATURE_SCHEMA
    audio, beats, reference = [tmp_path / name for name in ("audio.wav", "beats.txt", "reference.lab")]
    audio.write_bytes(b"fixture")
    beats.write_text("0 1\n1 2\n2 3\n3 4\n4 1\n", encoding="utf-8")
    reference.write_text("0 1.001 C:maj\n1 4 D:min\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    signature = json.dumps([FEATURE_SCHEMA, "fast", "half-bar", str(audio.resolve()), audio.stat().st_size,
                            audio.stat().st_mtime_ns, hashlib.sha256(beats.read_bytes()).hexdigest(),
                            hashlib.sha256(reference.read_bytes()).hexdigest()])
    legacy = cache_dir / (hashlib.sha256(signature.encode()).hexdigest()[:24] + ".npz")
    vectors = np.ones((2, 12), dtype=np.float32)
    np.savez_compressed(legacy, vectors=vectors, bass=vectors, targets=np.full((2, 24), 99.),
                        duration=np.array([2., 2.], dtype=np.float32))
    monkeypatch.setattr(studio, "decode_audio", lambda path: np.zeros(studio.SAMPLE_RATE * 4))
    def no_reextract(*args, **kwargs):
        raise AssertionError("Compatible cached spectra must be reused")
    monkeypatch.setattr(studio, "extract_chroma", no_reextract)
    monkeypatch.setattr(studio, "extract_bass_chroma", no_reextract)
    row = dict(id="fixture", audio="audio.wav", beat_reference="beats.txt", chord_reference="reference.lab")
    data = song_data(row, tmp_path, cache_dir, "fast", "half-bar")
    np.testing.assert_array_equal(data["vectors"], vectors)
    assert data["targets"][0, 0] == 1.
    assert data["targets"].sum() == 4.
    assert len(list(cache_dir.glob("*.npz"))) == 2


def test_benchmark_compatibility_scorer_uses_full_reference_and_rejects_overlaps():
    from benchmark_acoustic_frontend import score
    from evaluate_chords import TimedChord
    result = score([(0., 10., "C:maj7")], [TimedChord(0., 1., "C")])
    assert result["triad_accuracy"] == .1
    assert result["seconds"] == 10.
    with pytest.raises(ValueError, match="overlapping"):
        score([(0., 10., "C")], [TimedChord(0., 5., "C"), TimedChord(1., 10., "G")])
