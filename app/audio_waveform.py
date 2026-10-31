"""Small, time-aligned peak envelopes embedded in standalone workbenches."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
import numpy as np


def source_identity(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError("Audio changed while validating its waveform")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def peak_envelope(samples, sample_rate: int, *, offset: float = 0., points: int = 1600) -> dict:
    y = np.asarray(samples, dtype=np.float32)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Waveform audio must be a finite mono array")
    if not isinstance(sample_rate, int) or sample_rate <= 0 or points <= 0 or not math.isfinite(offset) or offset < 0:
        raise ValueError("Invalid waveform rate, size or offset")
    count = min(int(points), len(y))
    if count:
        starts = np.linspace(0, len(y), count + 1, dtype=np.int64)[:-1]
        peaks = np.maximum.reduceat(np.abs(y), starts)
        maximum = float(peaks.max())
        if maximum > 0:
            peaks = peaks / maximum
        values = np.round(peaks, 6).tolist()
    else:
        values = []
    return {"schema": "peak_envelope_v1", "peaks": values, "duration": len(y) / sample_rate,
            "audio_offset": offset, "sample_rate": sample_rate, "normalization": "global_peak", "source": "decoded_audio"}


def attach_waveform(payload: dict, *, samples=None, sample_rate: int = 22050) -> bool:
    """Add the analysis window's envelope; source JSON is never written here."""
    offset = float(payload.get("audio_offset") or 0)
    duration = float(payload.get("duration") or 0)
    source = Path(str(payload.get("source") or ""))
    identity = source_identity(source) if source.is_file() else None
    if samples is None:
        existing = payload.get("audio_waveform")
        if isinstance(existing, dict):
            try:
                peaks = np.asarray(existing.get("peaks", []), dtype=float)
                if (existing.get("schema") == "peak_envelope_v1" and peaks.ndim == 1 and 0 < len(peaks) <= 10000
                        and np.isfinite(peaks).all() and np.all((peaks >= 0) & (peaks <= 1))
                        and abs(float(existing.get("duration", -1)) - duration) < .05
                        and abs(float(existing.get("audio_offset", -1)) - offset) < 1e-6
                        and (identity is None or existing.get("audio_identity") == identity)):
                    existing["peaks"] = peaks.tolist()
                    return True
            except (TypeError, ValueError):
                pass
        payload.pop("audio_waveform", None)
        if not source.is_file() or duration <= 0:
            return False
        from chord_recognizer import decode_audio
        samples = decode_audio(source, sample_rate=sample_rate, offset=offset, duration=duration)
    envelope = peak_envelope(samples, sample_rate, offset=offset)
    if not envelope["peaks"] or abs(envelope["duration"] - duration) > .05:
        return False
    if identity is not None:
        if source_identity(source) != identity:
            raise OSError("Audio changed while generating its waveform")
        envelope["audio_identity"] = identity
    payload["audio_waveform"] = envelope
    return True
