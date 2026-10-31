"""Shared, explicitly centered spectral features for offline audio analysis."""

from __future__ import annotations

import numpy as np


def centered_magnitude_spectrum(
    y: np.ndarray, n_fft: int = 4096, hop_length: int = 2048,
) -> np.ndarray:
    """Return magnitudes with frame t centered at sample t * hop_length.

    This is the same time convention as librosa's centered CQT/STFT. The old
    fast extractor started the window at t * hop, shifting its evidence almost
    93 ms into the future at 22050 Hz and silently dropping the song's tail.
    Batching the FFT bounds temporary memory for long recordings.
    """
    samples = np.asarray(y, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("Audio must be a one-dimensional mono array.")
    if n_fft < 2 or n_fft % 2 or hop_length < 1:
        raise ValueError("n_fft must be positive and even; hop_length must be positive.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Audio contains non-finite samples.")
    if samples.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak
    padded = np.pad(samples, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(n_fft).astype(np.float32)
    magnitude = np.empty((n_fft // 2 + 1, len(frames)), dtype=np.float32)
    for start in range(0, len(frames), 256):
        block = frames[start:start + 256] * window
        magnitude[:, start:start + len(block)] = np.abs(np.fft.rfft(block, axis=1)).T
    return magnitude
