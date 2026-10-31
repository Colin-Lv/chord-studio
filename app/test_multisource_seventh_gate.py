import unittest

import numpy as np

from evaluate_chords import TimedChord, evaluate
from train_multisource_seventh_gate import overlap_features, gate_rows


class FixedHead:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, features):
        return np.tile([.02, .96, .02], (len(features), 1))


class ExternalSeventhGateMechanisms(unittest.TestCase):
    def test_duration_partition_and_missing_is_not_no_chord(self):
        upstream = [TimedChord(0, 8, "C")]
        external = [TimedChord(0, 1, "C"), TimedChord(1, 2, "C7"),
                    TimedChord(2, 3, "Cmaj7"), TimedChord(3, 4, "D"),
                    TimedChord(4, 5, "Cm"), TimedChord(5, 6, "N"),
                    TimedChord(6, 7, "X")]
        np.testing.assert_allclose(overlap_features(upstream, external), np.full((1, 8), .125))
        np.testing.assert_array_equal(overlap_features(upstream, []), [[0, 0, 0, 0, 0, 0, 0, 1]])

    def test_partial_window_and_same_triad_richer_chords(self):
        upstream = [TimedChord(1, 3, "C")]
        np.testing.assert_allclose(overlap_features(upstream,
            [TimedChord(0, 2, "C9"), TimedChord(2, 4, "Cmaj13")]), [[0, .5, .5, 0, 0, 0, 0, 0]])

    def test_gate_keeps_roots_triads_edges_bass_and_rich_labels(self):
        original = [TimedChord(i, i + 1, label) for i, label in enumerate(
            ["C/E", "Dm", "G9", "N", "Csus4", "Cmaj7"])]
        artifact = dict(model=FixedHead(), threshold=.8)
        transformed = gate_rows(original, np.zeros((len(original), 166)), artifact)
        self.assertEqual([row.chord for row in transformed], ["C7/E", "Dm7", "G9", "N", "Csus4", "Cmaj7"])
        score = evaluate(original, transformed)
        self.assertEqual(score["root_accuracy"], 1)
        self.assertEqual(score["triad_accuracy"], 1)
        self.assertEqual([(r.start, r.end) for r in transformed], [(r.start, r.end) for r in original])
        self.assertEqual(gate_rows(original, np.zeros((len(original), 166)), artifact, 1), original)


if __name__ == "__main__":
    unittest.main()
