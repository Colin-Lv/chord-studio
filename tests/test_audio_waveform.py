import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from audio_waveform import attach_waveform, peak_envelope


def test_peak_envelope_preserves_clip_time_and_impulse_bins():
    audio = np.zeros(1000, dtype=np.float32)
    audio[101], audio[909] = -.5, 1.
    waveform = peak_envelope(audio, 100, offset=30, points=10)
    assert waveform["duration"] == 10
    assert waveform["audio_offset"] == 30
    assert waveform["peaks"] == [0, .5, 0, 0, 0, 0, 0, 0, 0, 1]


def test_short_silent_and_empty_envelopes_are_finite():
    assert peak_envelope(np.zeros(3), 100)["peaks"] == [0, 0, 0]
    assert peak_envelope(np.array([]), 100)["peaks"] == []
    with pytest.raises(ValueError):
        peak_envelope(np.array([np.nan]), 100)


def test_wrong_window_is_not_attached():
    payload = {"duration": 2., "audio_offset": 30.}
    assert not attach_waveform(payload, samples=np.zeros(100), sample_rate=100)
    assert "audio_waveform" not in payload
    assert attach_waveform(payload, samples=np.zeros(200), sample_rate=100)
    assert payload["audio_waveform"]["audio_offset"] == 30


@pytest.mark.parametrize("bad", [None, [], {"schema": "peak_envelope_v1", "peaks": [None]},
                                  {"schema": "peak_envelope_v1", "peaks": [float("nan")]},
                                  {"schema": "peak_envelope_v1", "peaks": [-1]}])
def test_invalid_old_waveform_metadata_is_discarded_without_crashing(bad):
    payload = {"duration": 2, "audio_offset": 0, "audio_waveform": bad}
    assert not attach_waveform(payload)
    assert "audio_waveform" not in payload


def test_same_duration_replacement_audio_invalidates_old_envelope(tmp_path):
    import soundfile as sf
    source = tmp_path / "audio.wav"
    first = np.zeros(20000, dtype=np.float32)
    first[2000] = 1
    sf.write(source, first, 10000, subtype="FLOAT")
    payload = {"source": str(source), "duration": 2, "audio_offset": 0}
    assert attach_waveform(payload, sample_rate=10000)
    first_peak = int(np.argmax(payload["audio_waveform"]["peaks"]))
    second = np.zeros_like(first)
    second[18000] = 1
    sf.write(source, second, 10000, subtype="FLOAT")
    assert attach_waveform(payload, sample_rate=10000)
    second_peak = int(np.argmax(payload["audio_waveform"]["peaks"]))
    assert first_peak < 300
    assert second_peak > 1300


def test_valid_cached_numbers_are_normalized_for_javascript():
    payload = {"duration": 2, "audio_offset": 0, "audio_waveform": {
        "schema": "peak_envelope_v1", "duration": 2, "audio_offset": 0, "peaks": ["0", True, .5]}}
    assert attach_waveform(payload)
    assert all(type(value) is float for value in payload["audio_waveform"]["peaks"])
