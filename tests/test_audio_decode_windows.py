from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from chord_recognizer import decode_audio, find_ffmpeg


def test_mp3_excerpt_matches_the_same_full_decode_samples(tmp_path):
    rate = 44100
    time = np.arange(rate * 6) / rate
    audio = (.4 * np.sin(2*np.pi*220*time) + .2 * np.sin(2*np.pi*329.6276*time)) * (.5 + .3*np.sin(time*3.7))
    wave, mp3 = tmp_path / "source.wav", tmp_path / "source.mp3"
    sf.write(wave, audio, rate)
    subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(wave),
                    "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)
    target_rate = 24000
    full = decode_audio(mp3, target_rate)
    for offset in (1., 3., 3.12345):
        actual = decode_audio(mp3, target_rate, offset=offset, duration=1.5)
        start = round(offset * target_rate)
        expected = full[start:start+36000]
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"offset": float('nan')}, {"duration": 0}, {"sample_rate": 0}])
def test_invalid_audio_windows_fail_before_decoding(kwargs):
    with pytest.raises(ValueError):
        decode_audio(Path("missing.wav"), **kwargs)
