import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from candidate_diagnostics_v2 import candidate_diagnostics
from diagnose_boundaries_v2 import boundary_oracle, checked_identity
from experiment_protocol_v2 import signature
from multisource_candidates_v2 import TimedChord as T


def test_one_raw_interval_cannot_follow_two_reference_roots():
    result = boundary_oracle([T(0, 2, "C")], [T(0, 1, "C"), T(1, 2, "D")])
    root = result["fields"]["root"]
    assert root["correct_seconds"] == 1 and root["valid_reference_seconds"] == 2
    assert root["accuracy"] == .5 and root["fixed_boundary_mixture_loss_seconds"] == 1
    assert result["oracle_only"] and result["interval_count"] == 1
    assert result["intervals"][0]["start"] == 0 and result["intervals"][0]["end"] == 2


def test_raw_gaps_never_shrink_the_known_reference_denominator():
    reference = [T(0, 1, "C"), T(1, 2, "D")]
    complete = boundary_oracle([T(0, 2, "G")], reference)["fields"]["root"]
    missing = boundary_oracle([T(0, 1, "G")], reference)["fields"]["root"]
    assert complete["accuracy"] == missing["accuracy"] == .5
    assert missing["valid_reference_seconds"] == 2
    assert missing["raw_uncovered_valid_seconds"] == 1
    empty = boundary_oracle([], reference)["fields"]["root"]
    assert empty["accuracy"] == 0 and empty["raw_uncovered_valid_seconds"] == 2


@pytest.mark.parametrize("target", ["raw", "reference"])
def test_overlapping_scoring_intervals_are_rejected(target):
    raw, reference = [T(0, 2, "C")], [T(0, 2, "C")]
    (raw if target == "raw" else reference).append(T(1, 2, "C"))
    with pytest.raises(ValueError, match="disjoint"):
        boundary_oracle(raw, reference)


def test_reference_masks_and_unknown_regions_survive_oracle_selection():
    reference = [{"start": 0, "end": 1, "chord": "C9/E", "field_mask": {"bass": False}},
                 {"start": 1, "end": 2, "chord": "D9/F#", "valid_fields": ["root", "bass"]},
                 T(2, 3, "X"), T(4, 5, "N")]
    result = boundary_oracle([T(0, 5, "G")], reference)
    fields = result["fields"]
    assert fields["root"]["valid_reference_seconds"] == 3
    assert fields["root"]["correct_seconds"] == 1
    assert fields["bass"]["valid_reference_seconds"] == 1
    assert fields["bass"]["accuracy"] == 1
    assert fields["ext9"]["valid_reference_seconds"] == 1
    assert fields["ext9"]["accuracy"] == 1
    assert fields["ext11"]["accuracy"] is None and fields["ext11"]["valid_reference_seconds"] == 0
    assert result["annotation_coverage"]["gap_seconds"] == 1


def test_candidate_loss_is_separate_from_raw_support_and_boundary_mixing():
    raw = [T(0, 2, "G")]
    reference = [T(0, 1, "C"), T(1, 3, "D")]
    candidates = candidate_diagnostics(raw, {"raw_madmom": raw}, reference)["oracle"]
    root = boundary_oracle(raw, reference, candidates)["fields"]["root"]
    assert root["valid_reference_seconds"] == 3
    assert root["raw_uncovered_valid_seconds"] == 1
    assert root["fixed_boundary_mixture_loss_seconds"] == 1
    for variant in ("v1", "v2"):
        assert root["candidate_oracles"][variant]["candidate_absence_loss_seconds"] == 1
        assert root["candidate_oracles"][variant]["total_loss_seconds"] == 3
        assert root["candidate_oracles"][variant]["accuracy"] == 0
    inflated = copy.deepcopy(candidates)
    inflated["v2"]["root"]["correct_seconds"] = 4
    with pytest.raises(ValueError, match="exceeds"):
        boundary_oracle(raw, reference, inflated)


def test_source_bytes_are_verified_before_reference_diagnosis(tmp_path):
    path = tmp_path / "reference.lab"
    path.write_text("0 1 C\n", encoding="utf-8")
    frozen = signature(path)
    assert checked_identity(frozen) == frozen
    path.write_text("0 1 D\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        checked_identity(frozen)
