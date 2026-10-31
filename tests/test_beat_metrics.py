import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from evaluate_beats import BeatMark, evaluate, greedy_match


def test_matching_does_not_consume_the_next_beats_only_partner():
    hits, _ = greedy_match([1., 1.1], [.94, 1.04], .07)
    assert hits == 2


def test_missing_predicted_downbeats_count_as_zero_recall():
    metrics = evaluate([BeatMark(1., 1), BeatMark(2., 2)], [BeatMark(1., 2), BeatMark(2., 3)], .07)
    assert metrics["beat_f1"] == 1.
    assert metrics["downbeat_f1"] == 0.
    assert metrics["downbeat_reference_count"] == 1.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_invalid_tolerance_is_rejected(bad):
    with pytest.raises(ValueError):
        greedy_match([1.], [1.], bad)
