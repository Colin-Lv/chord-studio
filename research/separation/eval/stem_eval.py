#!/usr/bin/env python3
"""Small, dependency-light evaluation harness for music stem separation.

Two modes
---------
reference
    ``--truth DIR --stems DIR``
    Matched ground-truth and estimated WAV files are compared with SI-SDR and
    plain SDR.

reference-free
    ``--mix mix.wav --stems DIR``
    Only the original mix and estimated stems are needed.  The tool reports
    reconstruction residual, per-stem level/activity, and a cross-stem
    envelope-correlation matrix.

Stems are matched by canonical filename (case-insensitive), e.g.
``vocals.wav``.  Supported names are vocals, drums, bass, other, guitar and
piano.  Inputs must have identical sample rates, channel counts and lengths.

Only numpy + soundfile are required.  The output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SUPPORTED_STEMS: tuple[str, ...] = (
    "vocals",
    "drums",
    "bass",
    "other",
    "guitar",
    "piano",
    "instrumental",
    "accompaniment",
)

# Degenerate ratios (for example an exactly perfect reconstruction) are
# clipped to this range so that the CLI can always emit strictly valid JSON.
DB_CEILING = 200.0
DB_FLOOR = -200.0

# Activity is measured on short frames.  A frame is active when its RMS is no
# more than 40 dB below the stem's loudest frame and above the absolute
# -60 dBFS floor.
ACTIVITY_FRAME_MS = 20.0
ACTIVITY_HOP_MS = 10.0
ACTIVITY_ABS_FLOOR_DBFS = -60.0
ACTIVITY_PEAK_RANGE_DB = 40.0

# Amplitude envelopes used for cross-stem correlation.
ENVELOPE_FRAME_MS = 50.0
ENVELOPE_HOP_MS = 25.0

_DB_DIGITS = 4
_RATIO_DIGITS = 6


class EvalError(ValueError):
    """A user-facing input or evaluation error."""


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def _as_2d(signal: Any, label: str = "signal") -> np.ndarray:
    """Return a finite ``(frames, channels)`` float64 array."""

    try:
        arr = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvalError(f"{label} is not a numeric array") from exc

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise EvalError(
            f"{label} must be 1-D or 2-D (frames, channels); got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise EvalError(f"{label} contains no samples")
    if not np.all(np.isfinite(arr)):
        raise EvalError(f"{label} contains NaN or infinite values")
    return arr


def _paired_arrays(reference: Any, estimate: Any) -> tuple[np.ndarray, np.ndarray]:
    ref = _as_2d(reference, "reference")
    est = _as_2d(estimate, "estimate")
    if ref.shape != est.shape:
        raise EvalError(
            "reference and estimate shapes differ: "
            f"{ref.shape} vs {est.shape}"
        )
    return ref, est


def _clamp_db(value: float) -> float:
    return float(min(DB_CEILING, max(DB_FLOOR, value)))


def _amplitude_to_dbfs(amplitude: float) -> float:
    """Convert a linear amplitude/RMS to dBFS (0 dBFS == full scale 1.0)."""

    if amplitude <= 0.0:
        return DB_FLOOR
    return float(max(DB_FLOOR, 20.0 * math.log10(amplitude)))


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    rounded = round(value, digits)
    return 0.0 if rounded == 0.0 else rounded


# ---------------------------------------------------------------------------
# Reference metrics
# ---------------------------------------------------------------------------


def si_sdr_db(reference: Any, estimate: Any) -> float | None:
    """Scale-invariant SDR in dB.

    Each channel is mean-centred, channels are flattened, and the estimate is
    projected onto the reference:

        alpha = <s_hat, s> / ||s||^2
        SI-SDR = 10 log10(||alpha*s||^2 / ||s_hat - alpha*s||^2)

    Returns ``None`` when the reference has no energy (undefined).  Scores are
    clipped to [-200, 200] dB so perfect/enormously bad cases stay JSON-safe.
    """

    ref, est = _paired_arrays(reference, estimate)
    ref = ref - np.mean(ref, axis=0, keepdims=True)
    est = est - np.mean(est, axis=0, keepdims=True)
    s = ref.ravel()
    e = est.ravel()

    reference_energy = float(np.dot(s, s))
    if reference_energy <= 0.0:
        return None

    alpha = float(np.dot(e, s) / reference_energy)
    target_energy = alpha * alpha * reference_energy
    error = e - alpha * s
    error_energy = float(np.dot(error, error))

    # A zero/constant estimate has no component along the reference: that is
    # a total failure, not a perfect score.  Check this before the zero-error
    # case (for a silent estimate both energies are zero).
    if target_energy <= 0.0:
        return DB_FLOOR
    if error_energy <= 0.0:
        return DB_CEILING

    return _clamp_db(10.0 * math.log10(target_energy / error_energy))


def sdr_db(reference: Any, estimate: Any) -> float | None:
    """Plain (unfiltered) SDR in dB.

        SDR = 10 log10(sum(s^2) / sum((s - s_hat)^2))

    This is intentionally *not* BSS Eval v4: no time-invariant distortion
    filter, window, or source permutation is applied.  Returns ``None`` when
    the reference has no energy.  Scores are clipped to [-200, 200] dB.
    """

    ref, est = _paired_arrays(reference, estimate)
    s = ref.ravel()
    e = est.ravel()

    reference_energy = float(np.dot(s, s))
    if reference_energy <= 0.0:
        return None

    error_energy = float(np.dot(s - e, s - e))
    if error_energy <= 0.0:
        return DB_CEILING

    return _clamp_db(10.0 * math.log10(reference_energy / error_energy))


def rms_dbfs(signal: Any) -> float:
    """RMS level in dBFS over every sample and channel.

    Silence is reported at the -200 dBFS floor rather than ``-inf``.
    """

    arr = _as_2d(signal)
    rms = float(np.sqrt(np.mean(arr * arr)))
    return _amplitude_to_dbfs(rms)


# ---------------------------------------------------------------------------
# Reference-free level, activity and envelope metrics
# ---------------------------------------------------------------------------


def _frame_rms(
    signal: Any,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    """Frame RMS on the mono-summed channel power.

    Full frames only (the short tail is dropped), except that a signal shorter
    than one frame is zero-padded to one frame so a value can still be
    reported.
    """

    arr = _as_2d(signal)
    if sample_rate <= 0:
        raise EvalError(f"sample rate must be positive; got {sample_rate}")
    if frame_ms <= 0.0 or hop_ms <= 0.0:
        raise EvalError("frame and hop lengths must be positive")

    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    power = np.mean(arr * arr, axis=1)
    n_samples = power.shape[0]

    if n_samples < frame_length:
        power = np.pad(power, (0, frame_length - n_samples), mode="constant")
        return np.array([math.sqrt(float(np.mean(power)))], dtype=np.float64)

    windows = np.lib.stride_tricks.sliding_window_view(power, frame_length)
    windows = windows[::hop_length]
    return np.sqrt(np.mean(windows, axis=1))


def active_ratio(
    signal: Any,
    sample_rate: int,
    frame_ms: float = ACTIVITY_FRAME_MS,
    hop_ms: float = ACTIVITY_HOP_MS,
) -> float:
    """Fraction of short frames considered active.

    Threshold = max(-60 dBFS, stem peak frame - 40 dB).  Using the stem's own
    peak makes the value comparable between loud and quiet stems while the
    absolute floor prevents near-silent numerical noise being called active.
    """

    frame_rms = _frame_rms(signal, sample_rate, frame_ms, hop_ms)
    if frame_rms.size == 0:
        return 0.0

    frame_db = np.full(frame_rms.shape, DB_FLOOR, dtype=np.float64)
    positive = frame_rms > 0.0
    frame_db[positive] = 20.0 * np.log10(frame_rms[positive])
    peak_db = float(np.max(frame_db))
    threshold_db = max(
        ACTIVITY_ABS_FLOOR_DBFS, peak_db - ACTIVITY_PEAK_RANGE_DB
    )
    return float(np.mean(frame_db >= threshold_db))


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Zero-lag Pearson correlation, or None when either input is constant."""

    if a.size != b.size or a.size < 2:
        return None

    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denominator = math.sqrt(
        float(np.dot(a_centered, a_centered))
        * float(np.dot(b_centered, b_centered))
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None

    value = float(np.dot(a_centered, b_centered) / denominator)
    if not math.isfinite(value):
        return None
    return float(np.clip(value, -1.0, 1.0))


def envelope_correlation_matrix(
    stem_signals: Mapping[str, Any],
    sample_rate: int,
    frame_ms: float = ENVELOPE_FRAME_MS,
    hop_ms: float = ENVELOPE_HOP_MS,
) -> dict[str, Any]:
    """Pearson correlation of short-time RMS envelopes for every stem pair.

    Envelopes are linear RMS (not dB).  Correlations use zero lag, so a fixed
    time offset between two stems can reduce the value.  Pairs containing a
    silent or perfectly constant envelope are reported as ``None``; a usable
    diagonal is 1.0.
    """

    labels = list(stem_signals.keys())
    if not labels:
        raise EvalError("at least one stem is required for correlation")

    envelopes = [
        _frame_rms(stem_signals[label], sample_rate, frame_ms, hop_ms)
        for label in labels
    ]
    frame_count = min(env.size for env in envelopes)
    matrix: list[list[float | None]] = [
        [None for _ in labels] for _ in labels
    ]

    for i, label_i in enumerate(labels):
        env_i = envelopes[i][:frame_count]
        for j in range(i, len(labels)):
            env_j = envelopes[j][:frame_count]
            value = _pearson(env_i, env_j)
            if i == j and value is not None:
                value = 1.0
            matrix[i][j] = value
            matrix[j][i] = value

    return {
        "frame_ms": float(frame_ms),
        "hop_ms": float(hop_ms),
        "labels": labels,
        "matrix": matrix,
    }


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float64 ``(frames, channels)`` soundfile data."""

    path = Path(path)
    if not path.is_file():
        raise EvalError(f"audio file not found: {path}")

    try:
        audio, sample_rate = sf.read(
            str(path), dtype="float64", always_2d=True
        )
    except Exception as exc:  # soundfile uses several exception subclasses
        raise EvalError(f"could not read audio {path}: {exc}") from exc

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise EvalError(f"audio file has no samples: {path}")
    if audio.shape[1] == 0:
        raise EvalError(f"audio file has no channels: {path}")
    if not np.all(np.isfinite(audio)):
        raise EvalError(f"audio file contains NaN/infinite samples: {path}")
    if sample_rate <= 0:
        raise EvalError(f"audio file has an invalid sample rate: {path}")
    return audio, int(sample_rate)


def discover_stems(directory: str | Path) -> dict[str, Path]:
    """Find supported stem WAV files directly inside *directory*.

    Matching is case-insensitive on the filename stem.  Other files and other
    WAV names are ignored.  Results are returned in canonical stem order.
    """

    root = Path(directory)
    if not root.is_dir():
        raise EvalError(f"stems directory not found: {root}")

    found: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda p: (p.name.lower(), p.name)):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        name = path.stem.lower()
        if name not in SUPPORTED_STEMS:
            continue
        if name in found:
            raise EvalError(
                f"duplicate stem filename (case-insensitive): {name}.wav"
            )
        found[name] = path

    if not found:
        expected = ", ".join(f"{name}.wav" for name in SUPPORTED_STEMS)
        raise EvalError(
            f"no supported stem WAVs in {root}; expected one or more of: {expected}"
        )

    return {name: found[name] for name in SUPPORTED_STEMS if name in found}


def _check_pair(
    name: str,
    reference: np.ndarray,
    reference_sr: int,
    estimate: np.ndarray,
    estimate_sr: int,
) -> None:
    if reference_sr != estimate_sr:
        raise EvalError(
            f"{name}: sample rates differ "
            f"({reference_sr} Hz truth vs {estimate_sr} Hz estimate)"
        )
    if reference.shape != estimate.shape:
        raise EvalError(
            f"{name}: shapes differ "
            f"({reference.shape} truth vs {estimate.shape} estimate); "
            "trim/resample inputs before evaluation"
        )


def _average_defined(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------


def evaluate_reference(
    truth_dir: str | Path,
    stems_dir: str | Path,
) -> dict[str, Any]:
    """Reference mode: SDR / SI-SDR per matched stem plus unweighted means."""

    truth_files = discover_stems(truth_dir)
    estimate_files = discover_stems(stems_dir)

    missing = [
        name
        for name in truth_files
        if name not in estimate_files
    ]
    extra = [
        name
        for name in estimate_files
        if name not in truth_files
    ]
    if missing or extra:
        details = []
        if missing:
            details.append("missing estimates: " + ", ".join(missing))
        if extra:
            details.append("unmatched estimates: " + ", ".join(extra))
        raise EvalError(
            "truth and estimate stem sets differ (" + "; ".join(details) + ")"
        )

    names = [name for name in SUPPORTED_STEMS if name in truth_files]
    if not names:
        raise EvalError("no matched stems to evaluate")

    per_stem: dict[str, dict[str, float | None]] = {}
    raw_scores: dict[str, tuple[float | None, float | None]] = {}
    first_shape: tuple[int, int] | None = None
    sample_rate: int | None = None

    for name in names:
        truth_audio, truth_sr = load_audio(truth_files[name])
        estimate_audio, estimate_sr = load_audio(estimate_files[name])
        _check_pair(name, truth_audio, truth_sr, estimate_audio, estimate_sr)

        if sample_rate is None:
            sample_rate = truth_sr
            first_shape = truth_audio.shape
        else:
            if truth_sr != sample_rate:
                raise EvalError(
                    f"{name}: inconsistent sample rate across stems "
                    f"({truth_sr} Hz vs {sample_rate} Hz)"
                )
            if truth_audio.shape != first_shape:
                raise EvalError(
                    f"{name}: inconsistent shape across stems "
                    f"({truth_audio.shape} vs {first_shape})"
                )

        raw_si_sdr = si_sdr_db(truth_audio, estimate_audio)
        raw_sdr = sdr_db(truth_audio, estimate_audio)
        raw_scores[name] = (raw_si_sdr, raw_sdr)
        per_stem[name] = {
            "si_sdr_db": _round_or_none(raw_si_sdr, _DB_DIGITS),
            "sdr_db": _round_or_none(raw_sdr, _DB_DIGITS),
        }

    assert sample_rate is not None and first_shape is not None
    frames, channels = first_shape
    scored = [
        raw_si_sdr
        for raw_si_sdr, _raw_sdr in raw_scores.values()
        if raw_si_sdr is not None
    ]

    return {
        "mode": "reference",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": round(frames / sample_rate, _RATIO_DIGITS),
        "n_stems": len(names),
        "n_scored_stems": len(scored),
        "stems": per_stem,
        "mean_si_sdr_db": _round_or_none(
            _average_defined(
                [raw_si_sdr for raw_si_sdr, _raw_sdr in raw_scores.values()]
            ),
            _DB_DIGITS,
        ),
        "mean_sdr_db": _round_or_none(
            _average_defined(
                [raw_sdr for _raw_si_sdr, raw_sdr in raw_scores.values()]
            ),
            _DB_DIGITS,
        ),
    }


def evaluate_reference_free(
    mix_path: str | Path,
    stems_dir: str | Path,
) -> dict[str, Any]:
    """Reference-free mode using only the mix and estimated stems."""

    stem_files = discover_stems(stems_dir)
    mix, sample_rate = load_audio(mix_path)
    stem_signals: dict[str, np.ndarray] = {}

    for name, path in stem_files.items():
        signal, stem_sr = load_audio(path)
        if stem_sr != sample_rate:
            raise EvalError(
                f"{name}: sample rate {stem_sr} Hz does not match mix "
                f"{sample_rate} Hz"
            )
        if signal.shape != mix.shape:
            raise EvalError(
                f"{name}: shape {signal.shape} does not match mix "
                f"{mix.shape}; trim/resample inputs before evaluation"
            )
        stem_signals[name] = signal

    stem_sum = np.zeros_like(mix)
    for signal in stem_signals.values():
        stem_sum += signal
    residual = mix - stem_sum

    mix_energy = float(np.sum(mix * mix))
    residual_energy = float(np.sum(residual * residual))
    if mix_energy <= 0.0:
        residual_db_rel_mix = None
    elif residual_energy <= 0.0:
        residual_db_rel_mix = DB_FLOOR
    else:
        residual_db_rel_mix = _clamp_db(
            10.0 * math.log10(residual_energy / mix_energy)
        )

    per_stem: dict[str, dict[str, float]] = {}
    for name, signal in stem_signals.items():
        per_stem[name] = {
            "rms_dbfs": _round_or_none(rms_dbfs(signal), _DB_DIGITS),
            "active_ratio": _round_or_none(
                active_ratio(signal, sample_rate), _DB_DIGITS
            ),
        }

    correlation = envelope_correlation_matrix(stem_signals, sample_rate)
    rounded_matrix = [
        [_round_or_none(value, _RATIO_DIGITS) for value in row]
        for row in correlation["matrix"]
    ]
    correlation["matrix"] = rounded_matrix

    frames, channels = mix.shape
    return {
        "mode": "reference_free",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": round(frames / sample_rate, _RATIO_DIGITS),
        "n_stems": len(stem_signals),
        "stems": per_stem,
        "reconstruction": {
            "residual_db_rel_mix": _round_or_none(
                residual_db_rel_mix, _DB_DIGITS
            ),
            "residual_rms_dbfs": _round_or_none(
                rms_dbfs(residual), _DB_DIGITS
            ),
            "stem_sum_rms_dbfs": _round_or_none(
                rms_dbfs(stem_sum), _DB_DIGITS
            ),
            "mix_rms_dbfs": _round_or_none(rms_dbfs(mix), _DB_DIGITS),
        },
        "envelope_correlation": correlation,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/inf with None so strict JSON can be emitted."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _json_dumps(result: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(result), indent=2, allow_nan=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate music stem separation. Use --truth for reference "
            "metrics or --mix for reference-free metrics."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--truth",
        type=Path,
        metavar="DIR",
        help="directory containing ground-truth stem WAVs",
    )
    source.add_argument(
        "--mix",
        type=Path,
        metavar="MIX.WAV",
        help="original mixture WAV for reference-free evaluation",
    )
    parser.add_argument(
        "--stems",
        type=Path,
        required=True,
        metavar="DIR",
        help="directory containing estimated stem WAVs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.truth is not None:
            result = evaluate_reference(args.truth, args.stems)
        else:
            result = evaluate_reference_free(args.mix, args.stems)
    except EvalError as exc:
        print(f"stem_eval.py: error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"stem_eval.py: error: {exc}", file=sys.stderr)
        return 2

    print(_json_dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
