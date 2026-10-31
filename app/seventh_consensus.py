"""Conservative agreement evidence from already fitted seventh classifiers.

The returned rows sum to one, but are not calibrated probabilities. A seventh
gets evidence only when every head's most likely class is the same seventh.
No fitting, label lookup, or threshold selection takes place here.
"""
from __future__ import annotations

import numpy as np


class ConsensusSeventhModel:
    """Combine ``[(fitted_head, input_dimension), ...]`` on v2 features.

    Alternatively pass ``heads=[head1, head2]`` and
    ``input_dimensions=[87, 148]``. Each head sees only its declared prefix.
    Classes are 0 (no seventh), 1 (minor seventh), and 2 (major seventh).
    Missing head classes have zero evidence. The object is joblib serializable
    whenever its fitted heads are serializable.
    """

    def __init__(self, heads, input_dimensions=None):
        if input_dimensions is None:
            pairs = list(heads)
            if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
                raise ValueError("Supply (fitted_head, dimension) pairs or input_dimensions")
            self.heads = [pair[0] for pair in pairs]
            self.input_dimensions = [pair[1] for pair in pairs]
        else:
            self.heads = list(heads)
            self.input_dimensions = list(input_dimensions)
        if not self.heads or len(self.heads) != len(self.input_dimensions):
            raise ValueError("At least one fitted head and one dimension per head are required")
        self.n_features_in_ = 148
        self.classes_ = np.array([0, 1, 2], dtype=np.int64)
        for head, dimension in zip(self.heads, self.input_dimensions):
            if dimension not in (87, 148):
                raise ValueError("Head input dimensions must be 87 or 148")
            if not callable(getattr(head, "predict_proba", None)):
                raise ValueError("Each fitted head must implement predict_proba")
            if getattr(head, "n_features_in_", dimension) != dimension:
                raise ValueError("Declared dimension differs from fitted head n_features_in_")
            self._head_classes(head)

    @staticmethod
    def _head_classes(head):
        classes = np.asarray(getattr(head, "classes_", []))
        if (classes.ndim != 1 or not len(classes)
                or not np.all(np.isin(classes, [0, 1, 2]))
                or len(np.unique(classes)) != len(classes)):
            raise ValueError("Head classes_ must be distinct members of [0, 1, 2]")
        return classes.astype(np.int64)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Consensus input must have shape (n_samples, 148)")
        if not np.all(np.isfinite(X)):
            raise ValueError("Consensus input must be finite")
        result = np.zeros((len(X), 3), dtype=np.float64)
        result[:, 0] = 1.0
        if not len(X):
            return result
        votes, aligned_heads = [], []
        for head, dimension in zip(self.heads, self.input_dimensions):
            classes = self._head_classes(head)
            probability = np.asarray(head.predict_proba(X[:, :dimension]), dtype=np.float64)
            if probability.shape != (len(X), len(classes)):
                raise ValueError("Head predict_proba shape does not match samples/classes_")
            if (not np.all(np.isfinite(probability)) or np.any(probability < 0)
                    or np.any(probability > 1)
                    or not np.allclose(probability.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7)):
                raise ValueError("Head probabilities must be finite, within [0, 1], and sum to one")
            aligned = np.zeros_like(result)
            aligned[:, classes] = probability
            aligned_heads.append(aligned)
            # Canonical class order also makes tie handling deterministic.
            votes.append(np.argmax(aligned, axis=1))
        votes = np.asarray(votes)
        agreed = np.all(votes == votes[0], axis=0) & (votes[0] != 0)
        indices = np.flatnonzero(agreed)
        if len(indices):
            agreed_classes = votes[0, indices]
            confidence = np.min(
                [probability[indices, agreed_classes] for probability in aligned_heads], axis=0
            )
            result[indices, 0] = 1.0 - confidence
            result[indices, agreed_classes] = confidence
        return result

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
