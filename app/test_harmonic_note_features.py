"""Only local feature-coordinate and frame-selection mechanism checks."""
import unittest
from types import SimpleNamespace

import numpy as np

from harmonic_note_features import (HOP_LENGTH, SAMPLE_RATE, NoteEvidence,
                                    FEATURE_DIMENSION, features_for_intervals)


def evidence(midi):
    times = np.arange(5) * HOP_LENGTH / SAMPLE_RATE
    values = np.zeros((84, 5), dtype=np.float32)
    values[midi - 21, :3] = 1
    return NoteEvidence(times, values, dict(duration_seconds=5 * HOP_LENGTH / SAMPLE_RATE,
                                          decoded_samples=5 * HOP_LENGTH))


class NoteFeatureMechanisms(unittest.TestCase):
    def test_root_relative_coordinates_and_register_mass(self):
        end = 5 * HOP_LENGTH / SAMPLE_RATE
        c = features_for_intervals(evidence(60), [SimpleNamespace(start=0, end=end, chord="C")])[0]
        d = features_for_intervals(evidence(62), [SimpleNamespace(start=0, end=end, chord="D")])[0]
        np.testing.assert_array_equal(c, d)
        self.assertEqual(c.shape, (FEATURE_DIMENSION,))
        self.assertAlmostEqual(float(c[0]), .6)
        np.testing.assert_allclose(c[:12], c[48:60] + c[60:72] + c[72:84])
        self.assertAlmostEqual(float(c[84]), .6)

    def test_half_open_interval_and_subhop_selection(self):
        frame = HOP_LENGTH / SAMPLE_RATE
        intervals = [SimpleNamespace(start=0, end=3 * frame, chord="C"),
                     SimpleNamespace(start=3 * frame, end=5 * frame, chord="C"),
                     SimpleNamespace(start=.2 * frame, end=.3 * frame, chord="Cm")]
        features = features_for_intervals(evidence(60), intervals)
        self.assertEqual(features[0, 0], 1)
        self.assertEqual(features[1, 0], 0)
        self.assertEqual(features[2, 0], 1)
        self.assertEqual(features[2, -1], 1)

    def test_missing_tail_or_outside_audio_fails(self):
        original = evidence(60)
        incomplete = NoteEvidence(original.times[:-1], original.activations[:, :-1], original.metadata)
        with self.assertRaisesRegex(ValueError, "entire decoded audio"):
            features_for_intervals(incomplete, [])
        with self.assertRaisesRegex(ValueError, "exceeds decoded audio"):
            features_for_intervals(original, [SimpleNamespace(start=0, end=1, chord="C")])


if __name__ == "__main__":
    unittest.main()
