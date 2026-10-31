import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import chord_studio as studio


@pytest.mark.parametrize("text,expected", [("A minor", "Am"), ("F#:min", "F#m"), ("Bb:maj", "Bb"),
                                         ("D:aeolian", "Dm"), ("A♭ major", "Ab"), ("E", "E"), ("Bm", "Bm")])
def test_external_key_labels_keep_major_minor_semantics(text, expected):
    assert studio.normalize_key_label(text) == expected


def test_zero_transition_means_independent_acoustic_decisions():
    rng = np.random.default_rng(17)
    vectors = rng.random((24, 12), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    bass = rng.random((24, 12), dtype=np.float32)
    result = studio.functional_sequence(vectors, bass, np.ones(24), "C", "sharps", 0.)
    assert result.path.tolist() == np.argmax(result.state_scores, axis=1).tolist()


def test_zero_bass_weight_ignores_conflicting_bass_evidence():
    rng = np.random.default_rng(17)
    vectors = rng.random((16, 12), dtype=np.float32)
    bass = np.zeros((16, 12), dtype=np.float32)
    bass[:, 0] = 1
    first = studio.functional_sequence(vectors, bass, np.ones(16), "C", "sharps", .8, bass_weight=0.)
    second = studio.functional_sequence(vectors, np.roll(bass, 7, axis=1), np.ones(16), "C", "sharps", .8, bass_weight=0.)
    np.testing.assert_array_equal(first.state_scores, second.state_scores)


@pytest.mark.parametrize("option,value", [("--offset", "-1"), ("--duration", "0"), ("--bpm", "nan"),
                                             ("--transition-strength", "inf"), ("--bass-weight", "nan")])
def test_invalid_numeric_parameters_fail_before_audio_work(option, value):
    assert studio.main(["missing.wav", option, value]) == 2


def test_offset_shifts_source_annotations_and_preserves_provenance(tmp_path):
    import soundfile as sf
    sr = studio.SAMPLE_RATE
    t = np.arange(sr * 16) / sr
    audio = sum(.15 * np.sin(2 * np.pi * f * t) for f in [261.6256, 329.6276, 391.9954])
    source = tmp_path / "chords.wav"
    sf.write(source, audio, sr)
    beats = tmp_path / "source.beats"
    beats.write_text("\n".join(f"{i * .5} {i % 4 + 1}" for i in range(32)))
    target = tmp_path / "nested" / "clip"
    assert studio.main([str(source), "-o", str(target), "--offset", "10", "--duration", "4",
                        "--beat-grid", str(beats), "--bpm", "93", "--key", "C", "--feature-mode", "fast",
                        "--no-html", "--no-structure"]) == 0
    payload = json.loads(target.with_name("clip_analysis.json").read_text(encoding="utf-8"))
    assert payload["audio_offset"] == 10
    assert payload["beat_source"] == "reference_grid"
    assert payload["ignored_fixed_bpm"] == 93
    assert payload["key_source"] == "provided"
    assert payload["beat_period"] == pytest.approx(.5)
    assert payload["bars"][0]["start"] == 0
    assert payload["bars"][-1]["end"] == 4
    assert all(event["chord"] == "C" for event in payload["events"])
