"""Time alignment regressions: test evidence location, not implementation detail."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from audio_features import centered_magnitude_spectrum
from chord_recognizer import extract_chroma
from chord_studio import extract_bass_chroma


def test_impulse_has_its_peak_at_the_declared_frame_time():
    hop = 2048
    audio = np.zeros(hop * 10, dtype=np.float32)
    audio[hop * 4] = 1
    magnitude = centered_magnitude_spectrum(audio, hop_length=hop)
    assert int(np.argmax(magnitude.sum(axis=0))) == 4


@pytest.mark.parametrize("length", [1, 100, 4096, 4097, 22050])
def test_short_audio_and_last_partial_frame_are_present(length):
    audio = np.zeros(length, dtype=np.float32)
    audio[-1] = 1
    magnitude = centered_magnitude_spectrum(audio)
    assert magnitude.shape == (2049, 1 + length // 2048)
    assert magnitude[:, -1].max() > 0


def test_treble_and_bass_share_the_same_time_grid():
    audio = np.random.default_rng(0).normal(size=22051).astype(np.float32)
    treble = extract_chroma(audio, 22050, 2048, mode="fast")
    bass = extract_bass_chroma(audio, 22050, 2048, mode="fast")
    assert treble.shape == bass.shape == (12, 11)
    assert np.all(np.isfinite(treble)) and np.all(np.isfinite(bass))


def test_silence_and_empty_audio_do_not_invent_pitch_energy():
    for size in (0, 1, 22050):
        for extractor in (extract_chroma, extract_bass_chroma):
            features = extractor(np.zeros(size), 22050, 2048, mode="fast")
            assert np.all(np.isfinite(features))
            assert not np.any(features)
