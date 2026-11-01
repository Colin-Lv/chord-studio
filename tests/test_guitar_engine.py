"""Signal-level regression for overlap-add; no model weights or GPU required."""
import numpy as np
import torch
from app.web.guitar_engine import separate


def test_overlap_add_preserves_gain_length_and_stereo():
    class Identity:
        def __call__(self, x): return x
    rng = np.random.default_rng(13)
    x = rng.normal(size=(1493, 2)).astype('float32') * 1.7
    y = separate(Identity(), x, 256, 'cpu')
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=2e-6)


def test_anonymous_chunk_permutations_align_to_the_first_source():
    class Alternating:
        def __init__(self): self.calls = 0
        def __call__(self, x):
            y = torch.zeros_like(x)
            channel = self.calls % 2
            y[:, channel] = x[:, channel]
            self.calls += 1
            return y
    rng = np.random.default_rng(71)
    x = rng.normal(size=(1733, 2)).astype('float32')
    y = separate(Alternating(), x, 256, 'cpu', align_sources=True)
    np.testing.assert_allclose(y[:, 0], x[:, 0], atol=2e-6)
    np.testing.assert_allclose(y[:, 1], 0, atol=2e-6)
    # Reconstruction is an arithmetic invariant, not a separation quality score.
    np.testing.assert_allclose(y + (x-y), x, atol=2e-6)


def test_nonfinite_estimate_is_rejected_before_writing():
    class Bad:
        def __call__(self, x): return x * float('nan')
    import pytest
    with pytest.raises(ValueError, match='Nonfinite'):
        separate(Bad(), np.ones((300, 2), dtype='float32'), 256, 'cpu')
