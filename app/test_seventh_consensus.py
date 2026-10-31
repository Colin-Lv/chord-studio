import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from seventh_consensus import ConsensusSeventhModel


class FixedHead:
    def __init__(self, probabilities, classes=(0, 1, 2), dimension=148):
        self.probabilities = np.asarray(probabilities)
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = dimension
        self.seen_shape = None

    def predict_proba(self, X):
        self.seen_shape = X.shape
        return np.repeat(self.probabilities[None, :], len(X), axis=0)


class ConsensusTests(unittest.TestCase):
    def test_agreement_minimum_and_feature_prefix(self):
        a = FixedHead([.1, .8, .1], dimension=87)
        b = FixedHead([.25, .65, .1])
        model = ConsensusSeventhModel([(a, 87), (b, 148)])
        np.testing.assert_allclose(model.predict_proba(np.zeros((2, 148))), [[.35, .65, 0]] * 2)
        self.assertEqual(a.seen_shape, (2, 87))
        self.assertEqual(b.seen_shape, (2, 148))

    def test_reordered_and_missing_classes(self):
        a = FixedHead([.8, .1, .1], classes=[2, 0, 1])
        b = FixedHead([.7, .3], classes=[2, 0], dimension=87)
        model = ConsensusSeventhModel([a, b], [148, 87])
        np.testing.assert_allclose(model.predict_proba(np.zeros((1, 148))), [[.3, 0, .7]])

    def test_disagreement_and_no_seventh_abstain(self):
        for probabilities in ([.1, .1, .8], [.8, .1, .1]):
            model = ConsensusSeventhModel([(FixedHead([.1, .8, .1]), 148), (FixedHead(probabilities), 148)])
            np.testing.assert_array_equal(model.predict_proba(np.zeros((1, 148))), [[1, 0, 0]])

    def test_subhalf_agreement_remains_conservative(self):
        model = ConsensusSeventhModel([(FixedHead([.35, .4, .25]), 148)])
        np.testing.assert_allclose(model.predict_proba(np.zeros((1, 148))), [[.6, .4, 0]])
        np.testing.assert_array_equal(model.predict(np.zeros((1, 148))), [0])

    def test_invalid_inputs_and_empty(self):
        model = ConsensusSeventhModel([(FixedHead([.1, .8, .1]), 148)])
        for X in (np.zeros((1, 87)), np.zeros(148), np.full((1, 148), np.nan)):
            with self.assertRaises(ValueError):
                model.predict_proba(X)
        self.assertEqual(model.predict_proba(np.empty((0, 148))).shape, (0, 3))

    def test_invalid_heads_and_probabilities(self):
        for heads in ([], [(FixedHead([1]), 7)], [(FixedHead([1], classes=[3]), 148)],
                      [(FixedHead([.5, .5], classes=[1, 1]), 148)], [(FixedHead([1]), 87)]):
            with self.assertRaises(ValueError):
                ConsensusSeventhModel(heads)
        for probability in ([np.nan, .5, .5], [-.1, .5, .6], [.1, .2, .3], [1.1, 0, 0], [.5, .5]):
            model = ConsensusSeventhModel([(FixedHead(probability), 148)])
            with self.assertRaises(ValueError):
                model.predict_proba(np.zeros((1, 148)))

    def test_joblib_roundtrip(self):
        model = ConsensusSeventhModel([(FixedHead([.1, .2, .7]), 148)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_proba(np.zeros((1, 148))), [[.3, 0, .7]])
            self.assertEqual(restored.n_features_in_, 148)
            np.testing.assert_array_equal(restored.classes_, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
