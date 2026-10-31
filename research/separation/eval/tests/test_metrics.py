"""Metric tests against synthetic signals whose answers are known exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

import stem_eval as se

SR = 8000


def _tone(
    frequency_hz: float,
    seconds: float = 1.0,
    sample_rate: int = SR,
    amplitude: float = 0.2,
    phase: float = 0.0,
) -> np.ndarray:
    """A deterministic zero-mean tone; integer Hz over whole seconds."""

    frames = int(round(seconds * sample_rate))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * math.pi * frequency_hz * time + phase)


def _write_wav(path, audio: np.ndarray, sample_rate: int = SR) -> None:
    # DOUBLE preserves the synthetic samples exactly, so reconstruction
    # residual tests can check the mathematical result rather than WAV
    # quantisation noise.  The harness also reads ordinary PCM/FLOAT WAVs.
    sf.write(str(path), audio, sample_rate, subtype="DOUBLE")


# ---------------------------------------------------------------------------
# Required exact-answer cases
# ---------------------------------------------------------------------------


def test_perfect_estimate_scores_very_high() -> None:
    reference = _tone(440.0)
    estimate = reference.copy()

    si_sdr = se.si_sdr_db(reference, estimate)
    sdr = se.sdr_db(reference, estimate)

    assert si_sdr is not None and si_sdr > 60.0
    assert sdr is not None and sdr > 60.0
    # Degenerate zero-error ratios are clipped to a finite JSON-safe value.
    assert si_sdr == se.DB_CEILING
    assert sdr == se.DB_CEILING


def test_known_noise_snr_is_recovered_within_half_db() -> None:
    known_snr_db = 20.0
    reference = _tone(440.0, amplitude=0.1)

    rng = np.random.default_rng(20261101)
    noise = rng.standard_normal(reference.size)
    noise -= noise.mean()
    # Explicit finite-sample orthogonalisation makes the theoretical SI-SDR
    # exactly the requested SNR instead of merely approximately it.
    noise -= reference * (np.dot(noise, reference) / np.dot(reference, reference))
    noise_energy = float(np.dot(noise, noise))
    reference_energy = float(np.dot(reference, reference))
    target_noise_energy = reference_energy / (10.0 ** (known_snr_db / 10.0))
    noise *= math.sqrt(target_noise_energy / noise_energy)

    estimate = reference + noise
    measured_si_sdr = se.si_sdr_db(reference, estimate)
    measured_sdr = se.sdr_db(reference, estimate)

    assert measured_si_sdr is not None
    assert measured_sdr is not None
    assert abs(measured_si_sdr - known_snr_db) <= 0.5, (
        f"SI-SDR {measured_si_sdr:.4f} dB differs from expected "
        f"{known_snr_db:.1f} dB"
    )
    assert abs(measured_sdr - known_snr_db) <= 0.5


def test_fifty_percent_leakage_is_visible_in_scores() -> None:
    sr = SR
    source = _tone(440.0, sample_rate=sr)
    interferer = _tone(550.0, sample_rate=sr)

    # Over an integer number of cycles at integer frequencies the discrete
    # tones are orthogonal, so the leaked energy ratio is exactly 0.25.
    covariance = float(np.dot(source, interferer))
    assert abs(covariance) < 1e-12

    leaked = source + 0.5 * interferer
    clean_si_sdr = se.si_sdr_db(source, source.copy())
    leaked_si_sdr = se.si_sdr_db(source, leaked)
    leaked_sdr = se.sdr_db(source, leaked)

    expected_db = 20.0 * math.log10(2.0)  # ~6.0206 dB
    assert clean_si_sdr is not None and clean_si_sdr > 60.0
    assert leaked_si_sdr is not None and leaked_sdr is not None
    assert abs(leaked_si_sdr - expected_db) < 0.1
    assert abs(leaked_sdr - expected_db) < 0.1
    assert leaked_si_sdr < clean_si_sdr - 50.0
    assert leaked_sdr < 60.0


def test_identical_stems_have_correlation_near_one(tmp_path) -> None:
    seconds = 2.0
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    gate = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2.0 * math.pi * 1.7 * t)) ** 2
    shared = 0.15 * gate * np.sin(2.0 * math.pi * 440.0 * t)
    independent = 0.12 * (0.4 + 0.6 * np.sin(2.0 * math.pi * 2.3 * t) ** 2) * np.sin(
        2.0 * math.pi * 110.0 * t
    )

    truth_dir = tmp_path / "truth"
    estimate_dir = tmp_path / "estimates"
    truth_dir.mkdir()
    estimate_dir.mkdir()

    # Identical content in two different stem slots, plus a third stem.
    stems = {"vocals": shared, "drums": shared.copy(), "bass": independent}
    for name, audio in stems.items():
        _write_wav(truth_dir / f"{name}.wav", audio)
        _write_wav(estimate_dir / f"{name}.wav", audio.copy())

    mix = sum(stems.values())
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path, mix)

    result = se.evaluate_reference_free(mix_path, estimate_dir)
    correlation = result["envelope_correlation"]
    labels = correlation["labels"]
    matrix = correlation["matrix"]
    i_vocals = labels.index("vocals")
    i_drums = labels.index("drums")

    assert correlation["labels"] == ["vocals", "drums", "bass"]
    assert matrix[i_vocals][i_drums] == pytest.approx(1.0, abs=1e-6)
    assert matrix[i_drums][i_vocals] == pytest.approx(1.0, abs=1e-6)
    assert matrix[i_vocals][i_vocals] == 1.0
    assert result["reconstruction"]["residual_db_rel_mix"] == se.DB_FLOOR


# ---------------------------------------------------------------------------
# Additional guards for edge cases and supporting metrics
# ---------------------------------------------------------------------------


def test_silent_estimate_is_not_rewarded_by_si_sdr() -> None:
    reference = _tone(440.0)
    silent = np.zeros_like(reference)

    assert se.si_sdr_db(reference, silent) == se.DB_FLOOR
    # Plain SDR has no target projection: a silent estimate is 0 dB, not
    # "perfect", because the full reference is error.
    assert se.sdr_db(reference, silent) == pytest.approx(0.0, abs=1e-9)


def test_active_ratio_tracks_half_active_signal() -> None:
    sample_rate = SR
    seconds = 2.0
    signal = np.zeros(int(seconds * sample_rate), dtype=np.float64)
    # Second half is a tone; use whole cycles to keep it zero-mean.
    signal[sample_rate:] = _tone(440.0, seconds=1.0, sample_rate=sample_rate)

    ratio = se.active_ratio(signal, sample_rate)
    assert 0.45 <= ratio <= 0.55, f"unexpected active ratio: {ratio}"


def test_rms_dbfs_matches_known_sine_level() -> None:
    amplitude = 0.25
    signal = _tone(440.0, amplitude=amplitude)
    expected_rms = amplitude / math.sqrt(2.0)
    expected_dbfs = 20.0 * math.log10(expected_rms)

    assert se.rms_dbfs(signal) == pytest.approx(expected_dbfs, abs=1e-6)
    assert se.rms_dbfs(np.zeros(1000)) == se.DB_FLOOR


def test_reconstruction_residual_matches_known_ratio(tmp_path) -> None:
    mix = _tone(220.0, amplitude=0.2)
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path, mix)
    # One estimate at 90% gain leaves a residual of exactly 10% of the mix.
    _write_wav(stems_dir / "vocals.wav", 0.9 * mix)

    result = se.evaluate_reference_free(mix_path, stems_dir)
    expected_db = 20.0 * math.log10(0.1)
    assert result["reconstruction"]["residual_db_rel_mix"] == pytest.approx(
        expected_db, abs=1e-3
    )
    assert result["n_stems"] == 1


def test_stereo_is_supported_and_si_sdr_is_gain_invariant() -> None:
    left = _tone(440.0)
    right = _tone(550.0, phase=0.2)
    reference = np.stack([left, right], axis=1)

    assert se.si_sdr_db(reference, reference) == se.DB_CEILING
    assert se.sdr_db(reference, reference) == se.DB_CEILING
    scaled = 0.5 * reference
    assert se.si_sdr_db(reference, scaled) == se.DB_CEILING
    # Plain SDR, by contrast, sees the gain mismatch as about 6.02 dB of error.
    assert se.sdr_db(reference, scaled) == pytest.approx(
        20.0 * math.log10(2.0), abs=1e-9
    )


def test_shape_mismatch_is_rejected() -> None:
    reference = _tone(440.0)
    with pytest.raises(se.EvalError, match="shapes differ"):
        se.si_sdr_db(reference, reference[:-1])
