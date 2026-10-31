import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from candidate_targets_v2 import (HEADS, ReferenceCoveragePolicy, candidate_targets,
                                  aggregate_utility, annotation_coverage)
from multisource_candidates_v2 import TimedChord as T


def test_unknown_extensions_and_unannotated_bass_are_masked():
    result = candidate_targets(T(0, 2, "C"), ["C", "C9", "C/E"], [T(0, 2, "C")])
    assert result.mask.tolist() == [[True, True, True, False, False, False, False]] * 3
    assert np.isnan(result.values[:, 3:]).all()
    assert result.values[:, 0].tolist() == [1, 1, 1]
    assert result.values[:, 2].tolist() == [1, 0, 1]


def test_explicit_extensions_bass_and_alteration_have_independent_targets():
    candidates = ["C:9/E", "C7/E", "C:7(b9)/E", "C9", "D9/F#"]
    result = candidate_targets(T(0, 1, "C"), candidates, [T(0, 1, "C:9/E")])
    assert result.mask[0].tolist() == [True, True, True, True, True, False, False]
    assert result.values[:, HEADS.index("ext9")].tolist() == [1, 0, 0, 1, 0]
    assert result.values[:, HEADS.index("bass")].tolist() == [1, 1, 1, 0, 0]


def test_extension_omission_is_known_negative_but_bare_triad_is_not():
    omitted = candidate_targets(T(0, 1, "C"), ["C7", "C9"], [T(0, 1, "C:7(*9)")])
    assert omitted.mask[:, 4].tolist() == [True, True]
    assert omitted.values[:, 4].tolist() == [1, 0]
    unspecified = candidate_targets(T(0, 1, "C"), ["C7", "C9"], [T(0, 1, "C7")])
    assert not unspecified.mask[:, 4].any()


def test_closed_world_policy_is_explicit_and_records_coverage():
    policy = ReferenceCoveragePolicy(extension_absence_known=(9, 11, 13), implicit_bass_is_root=True)
    result = candidate_targets(T(0, 1, "C"), ["C", "C/C", "Cadd9"], [T(0, 1, "C")], policy)
    assert result.mask.all()
    assert result.values[:, 3].tolist() == [0, 1, 0]
    assert result.values[:, 4].tolist() == [1, 1, 0]
    assert result.policy.to_dict()["implicit_bass_is_root"] is True


def test_reference_N_X_gaps_and_per_row_mask_exact_durations():
    refs = [T(0, 1, "N"), T(1, 2, "X"),
            {"start": 3, "end": 4, "chord": "C9/E", "field_mask": {"bass": False, "tetrad": False}}]
    result = candidate_targets(T(0, 4, "C"), ["N", "C9/E"], refs)
    assert result.valid_seconds[0].tolist() == [2, 2, 1, 0, 1, 0, 0]
    assert result.values[:, 0].tolist() == [.5, .5]
    assert result.values[:, 2].tolist() == [1, 0]
    assert result.coverage[0].tolist() == [.5, .5, .25, 0, .25, 0, 0]
    coverage = annotation_coverage(refs)
    assert coverage["span_seconds"] == 4 and coverage["gap_seconds"] == 1
    assert coverage["valid_seconds"]["bass"] == 0
    assert coverage["state_seconds"] == {"N": 1, "X": 1, "chord": 1}


def test_per_reference_valid_fields_removes_and_cannot_invent_annotation():
    refs = [{"start": 0, "end": 1, "chord": "C", "valid_fields": ("root", "bass", "ext9")}]
    result = candidate_targets(T(0, 1, "C"), ["C"], refs)
    assert result.mask[0].tolist() == [True, False, False, False, False, False, False]


def test_soft_targets_integrate_reference_changes_without_moving_candidates():
    interval = T(0, 4, "C")
    result = candidate_targets(interval, ["C", "G"], [T(0, 1, "C"), T(1, 4, "G")])
    assert result.values[:, 0].tolist() == [.25, .75]
    assert interval == T(0, 4, "C")
    assert result.valid_seconds[:, :3].tolist() == [[4, 4, 4], [4, 4, 4]]


def test_masked_utility_requires_weights_and_never_imputes_zero():
    result = candidate_targets(T(0, 2, "C"), ["C", "G"], [T(0, 1, "C")])
    utility = aggregate_utility(result, {"root": 1, "bass": 4})
    assert utility.values.tolist() == [1, 0]
    assert utility.observed_weight.tolist() == [1, 1]
    assert utility.coverage.tolist() == [.1, .1]
    no_evidence = aggregate_utility(result, {"bass": 1})
    assert not no_evidence.mask.any() and np.isnan(no_evidence.values).all()
    with pytest.raises(TypeError):
        aggregate_utility(result)
    with pytest.raises(ValueError):
        aggregate_utility(result, {"root": -1})


def test_reference_overlap_invalid_mask_and_unknown_labels_are_rejected_or_masked():
    with pytest.raises(ValueError):
        candidate_targets(T(0, 2, "C"), ["C"], [T(0, 1.1, "C"), T(1, 2, "G")])
    with pytest.raises(ValueError):
        candidate_targets(T(0, 1, "C"), ["C"],
                          [{"start": 0, "end": 1, "chord": "C", "field_mask": {"root": "false"}}])
    result = candidate_targets(T(0, 1, "C"), ["C"], [T(0, 1, "H:maj")])
    assert not result.mask.any() and np.isnan(result.values).all()
