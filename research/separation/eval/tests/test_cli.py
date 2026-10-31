"""End-to-end checks for both CLI modes and strict JSON output."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import stem_eval as se

SR = 8000


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SR) -> None:
    sf.write(str(path), audio, sample_rate, subtype="DOUBLE")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(Path(se.__file__)), *args]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_cli_reports_valid_json_for_both_modes(tmp_path) -> None:
    seconds = 2.0
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    vocals = 0.12 * (0.5 + 0.5 * np.sin(2.0 * math.pi * 0.7 * t)) * np.sin(
        2.0 * math.pi * 440.0 * t
    )
    drums = 0.10 * np.sin(2.0 * math.pi * 3.0 * t) * (
        np.mod(t, 0.25) < 0.05
    )
    mix = vocals + drums

    truth_dir = tmp_path / "truth"
    estimate_dir = tmp_path / "estimates"
    truth_dir.mkdir()
    estimate_dir.mkdir()
    _write_wav(truth_dir / "vocals.wav", vocals)
    _write_wav(truth_dir / "drums.wav", drums)
    # Deliberate, deterministic contamination makes the reference scores
    # finite and clearly worse than perfect.
    _write_wav(estimate_dir / "vocals.wav", vocals + 0.01 * drums)
    _write_wav(estimate_dir / "drums.wav", drums)

    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path, mix)

    reference_run = _run_cli("--truth", str(truth_dir), "--stems", str(estimate_dir))
    assert reference_run.returncode == 0, reference_run.stderr
    reference = json.loads(reference_run.stdout)
    assert reference["mode"] == "reference"
    assert reference["n_stems"] == 2
    assert set(reference["stems"]) == {"vocals", "drums"}
    assert reference["stems"]["drums"]["si_sdr_db"] > 60.0
    assert reference["stems"]["vocals"]["si_sdr_db"] < 60.0
    assert reference["mean_si_sdr_db"] is not None
    assert reference["mean_sdr_db"] is not None

    free_run = _run_cli("--mix", str(mix_path), "--stems", str(estimate_dir))
    assert free_run.returncode == 0, free_run.stderr
    reference_free = json.loads(free_run.stdout)
    assert reference_free["mode"] == "reference_free"
    assert set(reference_free["stems"]) == {"vocals", "drums"}
    assert reference_free["envelope_correlation"]["labels"] == ["vocals", "drums"]
    correlation = reference_free["envelope_correlation"]["matrix"]
    assert len(correlation) == 2 and len(correlation[0]) == 2
    assert correlation[0][0] == 1.0
    assert correlation[1][1] == 1.0
    reconstruction = reference_free["reconstruction"]
    # The estimates deliberately contain a little extra drums energy, so the
    # reconstruction residual is finite and negative rather than the floor.
    assert reconstruction["residual_db_rel_mix"] is not None
    assert reconstruction["residual_db_rel_mix"] < 0.0


def test_cli_rejects_missing_stem_matches(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    estimate_dir = tmp_path / "estimates"
    truth_dir.mkdir()
    estimate_dir.mkdir()
    _write_wav(truth_dir / "vocals.wav", np.ones(800) * 0.01)
    _write_wav(estimate_dir / "drums.wav", np.ones(800) * 0.01)

    run = _run_cli("--truth", str(truth_dir), "--stems", str(estimate_dir))
    assert run.returncode == 2
    assert "stem sets differ" in run.stderr
    assert run.stdout == ""
