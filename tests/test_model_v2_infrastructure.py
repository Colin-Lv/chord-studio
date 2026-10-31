"""Small v2 infrastructure checks; no audio extraction or real model fitting."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import candidate_diagnostics_v2 as diagnostics
from candidate_targets_v2 import HEADS
import experiment_protocol_v2 as protocol
import multisource_features_v2 as features
import multisource_model_v2 as models
from multisource_candidates_v2 import TimedChord as T


class StubEstimator:
    """Pickleable width/prediction stub, deliberately never fitted."""
    n_features_in_ = features.DIMENSION

    def predict(self, matrix):
        return np.full(len(matrix), .6)


@pytest.fixture
def mocked_notes(monkeypatch):
    calls = []

    def note_features(evidence, windows):
        calls.append(tuple(windows))
        return np.tile(np.arange(97, dtype=float) / 97, (len(windows), 1))

    monkeypatch.setattr(features.notes, "features_for_intervals", note_features)
    return SimpleNamespace(metadata={"duration_seconds": 10.}), calls


def test_feature_dimension_full_extensions_and_explicit_bass(mocked_notes):
    evidence, calls = mocked_notes
    anchor = T(.25, 4.25, "C7")
    sources = {"raw_madmom": [anchor], "chordino": [T(.25, 1.25, "C9/E"),
               T(1.25, 2.25, "C11/G"), T(2.25, 3.25, "C13/Bb")],
               "largevocab": [T(.25, 4.25, "C/C")]}
    matrix, groups = features.automatic_candidate_features([anchor], sources, evidence)
    assert features.DIMENSION == len(features.FEATURE_NAMES) == 139
    assert len(set(features.FEATURE_NAMES)) == 139
    assert matrix.shape == (5, 139) and np.isfinite(matrix).all()
    assert matrix.dtype == np.float32 and len(calls) == 1
    assert all((row.start, row.end) == (.25, 4.25) for row in calls[0])
    assert groups[0]["raw_index"] == 0
    assert groups[0]["candidates"][0]["label"] == "C7"
    names = {name: i for i, name in enumerate(features.FEATURE_NAMES)}
    assert matrix[0, names["bass_unannotated"]] == 1
    assert matrix[0, names["chord_relative_pc_2"]] == 0
    for index, candidate in enumerate(groups[0]["candidates"]):
        semantic = candidate["semantic"]
        for degree, relative_pc in ((9, 2), (11, 5), (13, 9)):
            if degree in semantic.extension_degrees:
                assert matrix[index, names[f"chord_relative_pc_{relative_pc}"]] == 1
        if semantic.explicit_bass:
            assert matrix[index, names[f"explicit_bass_relative_pc_{semantic.bass_relative}"]] == 1
            assert matrix[index, names["bass_unannotated"]] == 0
    # C and C/C are distinct from C7; an explicit root bass is not "unknown".
    explicit_root = next(i for i, c in enumerate(groups[0]["candidates"])
                         if c["semantic"].explicit_bass and c["semantic"].bass_relative == 0)
    assert matrix[explicit_root, names["explicit_bass_relative_pc_0"]] == 1


def test_named_source_features_do_not_follow_dictionary_or_interval_order(mocked_notes):
    evidence, _ = mocked_notes
    anchor = T(0, 4, "C7")
    sources = {"raw_madmom": [anchor], "chordino": [T(0, 2, "C9"), T(2, 4, "D")],
               "largevocab": [T(0, 4, "Cmaj7")]}
    matrix, groups = features.automatic_candidate_features([anchor], sources, evidence)
    reordered = {key: list(reversed(value)) for key, value in reversed(list(sources.items()))}
    other, other_groups = features.automatic_candidate_features([anchor], reordered, evidence)
    np.testing.assert_array_equal(matrix, other)
    assert groups[0]["pool"] == other_groups[0]["pool"]
    column = {name: i for i, name in enumerate(features.FEATURE_NAMES)}
    assert matrix[0, column["raw_madmom_full_support"]] == 1
    assert matrix[0, column["chordino_root_support"]] == .5
    assert matrix[0, column["chordino_full_support"]] == 0
    assert matrix[0, column["largevocab_root_support"]] == 1
    assert matrix[0, column["largevocab_full_support"]] == 0


def test_feature_empty_input_and_distinct_unknown_states(mocked_notes):
    evidence, calls = mocked_notes
    empty, groups = features.automatic_candidate_features([], {}, evidence)
    assert empty.shape == (0, 139) and groups == [] and calls == []
    for label, state in ((None, "missing"), ("X", "X"), ("N", "N")):
        matrix, _ = features.automatic_candidate_features([T(0, 1, label)], {}, evidence)
        for candidate_state in features.STATES:
            index = features.FEATURE_NAMES.index("state_" + candidate_state)
            assert matrix[0, index] == (candidate_state == state)


def test_oracle_includes_reference_without_raw_coverage_and_unions_sources():
    raw = [T(0, 1, "C")]
    sources = {"raw_madmom": raw, "chordino": [T(0, 1, "C9/E"), T(0, 1, "C9/E")],
               "largevocab": [T(0, 1, "C9/E")]}
    report = diagnostics.candidate_diagnostics(raw, sources, [T(0, 4, "C9/E")])
    for variant in ("v1", "v2"):
        root = report["oracle"][variant]["root"]
        assert root["valid_reference_seconds"] == 4
        assert root["candidate_support_valid_seconds"] == 1
        assert root["correct_seconds"] == 1 and root["accuracy"] == .25
        for field in report["oracle"][variant].values():
            assert field["accuracy"] is None or 0 <= field["accuracy"] <= 1
    no_raw = diagnostics.candidate_diagnostics([], {}, [T(0, 4, "C")])
    assert no_raw["oracle"]["v2"]["root"]["accuracy"] == 0


@pytest.mark.parametrize("overlap_target", ["raw", "reference"])
def test_oracle_rejects_overlapping_scoring_support(overlap_target):
    raw, reference = [T(0, 2, "C")], [T(0, 2, "C")]
    if overlap_target == "raw":
        raw.append(T(1, 2, "C"))
    else:
        reference.append(T(1, 2, "C"))
    with pytest.raises(ValueError, match="disjoint"):
        diagnostics.candidate_diagnostics(raw, {}, reference)


def test_dict_reference_masks_survive_alignment_metrics_oracle_and_changes():
    refs = [{"start": 0, "end": 2, "chord": "C9/E", "valid_fields": ["root", "bass"],
             "field_mask": {"bass": False}},
            {"start": 2, "end": 4, "chord": "G9/B", "valid_fields": ["root"]}]
    raw, predicted = [T(0, 4, "D")], [T(0, 4, "C9/E")]
    metric = diagnostics.field_metrics(refs, predicted)
    assert metric["root"]["valid_seconds"] == 4 and metric["root"]["correct_seconds"] == 2
    changes = diagnostics.change_diagnostics(refs, raw, predicted)
    oracle = diagnostics.candidate_diagnostics(raw, {"chordino": predicted}, refs)
    assert changes["fields"]["root"]["helpful_seconds"] == 2
    for head in HEADS[1:]:
        assert metric[head]["valid_seconds"] == 0 and metric[head]["accuracy"] is None
        assert changes["fields"][head] == {"harmful_seconds": 0., "helpful_seconds": 0.}
        assert oracle["oracle"]["v2"][head]["valid_reference_seconds"] == 0
        assert oracle["oracle"]["v2"][head]["accuracy"] is None


def grouped_row(index, split="fit", domain="folk"):
    return dict(id=f"id{index}", work_id=f"work{index}", artist_id=f"artist{index}",
                recording_id=f"recording{index}", audio_sha256=f"audio{index}", domain=domain,
                split=split, grouping_evidence="Synthetic explicit independent provenance")


@pytest.mark.parametrize("axis", ["work_id", "artist_id", "recording_id", "audio_sha256"])
def test_all_grouping_axes_reject_cross_partition_leakage(axis):
    first, second = grouped_row(0), grouped_row(1, "sealed_test")
    second[axis] = first[axis]
    with pytest.raises(ValueError, match="grouping leakage"):
        protocol.validate_grouped_rows([first, second], require_complete=False)


def test_transitive_regression_groups_stay_together_independent_of_input_order():
    rows = [grouped_row(i) for i in range(9)]
    rows[0]["known_regression"] = True
    rows[1]["work_id"] = rows[0]["work_id"]
    rows[2]["artist_id"] = rows[1]["artist_id"]
    assigned = protocol.grouped_assignment(rows)
    reverse = protocol.grouped_assignment(list(reversed(rows)))
    mapping = {row["id"]: row["split"] for row in assigned}
    assert mapping == {row["id"]: row["split"] for row in reverse}
    assert all(mapping[f"id{i}"] == "known_regression" for i in range(3))
    assert set(mapping.values()) == set(protocol.PARTITIONS)
    assert all(row["split"] == "fit" for row in rows)  # caller data untouched
    leaked = grouped_row(10, "sealed_test")
    leaked["known_regression"] = True
    with pytest.raises(ValueError, match="Known regression"):
        protocol.validate_grouped_rows([leaked], require_complete=False)


def test_complete_protocol_requires_cross_style_fit_and_development():
    rows = [grouped_row(0, "fit", "folk"), grouped_row(1, "fit", "classical"),
            grouped_row(2, "development", "folk"), grouped_row(3, "development", "classical"),
            grouped_row(4, "calibration"), grouped_row(5, "sealed_test")]
    assert protocol.validate_grouped_rows(rows)["development"] == 2
    with pytest.raises(ValueError, match="nonempty"):
        protocol.validate_grouped_rows(rows[:-1])
    rows[3]["domain"] = "folk"
    with pytest.raises(ValueError, match="two explicitly documented domains"):
        protocol.validate_grouped_rows(rows)


def test_configuration_binds_exact_split_bytes_and_rejects_invalid_weights(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text('{"frozen": true}\n', encoding="utf-8")
    config = dict(schema=protocol.CONFIG_SCHEMA, split_protocol=protocol.signature(path),
                  weights=dict.fromkeys(protocol.HEADS, 1.), margin=.02,
                  reference_coverage_policy={"implicit_bass_is_root": False})
    assert protocol.validate_configuration(config, path) is config
    for bad_weights in ({"root": 1.}, dict.fromkeys(protocol.HEADS, 0.),
                        dict(config["weights"], bass=float("nan"))):
        with pytest.raises(ValueError, match="weights"):
            protocol.validate_configuration(dict(config, weights=bad_weights), path)
    path.write_text('{"frozen": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bind"):
        protocol.validate_configuration(config, path)


@pytest.fixture
def saved_model(tmp_path):
    path = tmp_path / "stub.joblib"
    models.save_artifact(path, StubEstimator(), objective="synthetic contract only", margin=.02,
        provenance={"ids": ("synthetic",)}, nnls_contract={"synthetic": "no extraction"},
        selection_config={"valid_fields": ("root", "triad")})
    return path


def test_artifact_contract_json_roundtrip_and_no_overwrite(saved_model):
    artifact, contract = models.load_artifact(saved_model)
    assert artifact["feature_dimension"] == 139
    assert contract["feature_names"] == list(features.FEATURE_NAMES)
    np.testing.assert_array_equal(models.score_candidates(artifact, np.zeros((2, 139))), [.6, .6])
    with pytest.raises(FileExistsError):
        models.save_artifact(saved_model, StubEstimator(), objective="overwrite", margin=.02,
                             provenance={}, nnls_contract={})


@pytest.mark.parametrize("field,replacement", [
    ("artifact_schema", "cached_nnls_multisource_relative_candidate_utility_v1"),
    ("feature_schema", "old_semantics_same_width"), ("feature_dimension", 136),
    ("feature_names", list(reversed(features.FEATURE_NAMES)))])
def test_contract_rejects_v1_or_changed_feature_semantics(saved_model, field, replacement):
    path = saved_model.with_suffix(".contract.json")
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract[field] = replacement
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="compatible v2"):
        models.load_artifact(saved_model)


def test_contract_rejects_changed_model_bytes_and_source_implementation(saved_model, monkeypatch):
    identity = copy.deepcopy(models.implementation_identity())
    identity["multisource_features_v2.py"] = "changed semantic implementation"
    monkeypatch.setattr(models, "implementation_identity", lambda: identity)
    with pytest.raises(ValueError, match="source semantics changed"):
        models.load_artifact(saved_model)
    saved_model.write_bytes(saved_model.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Model bytes changed"):
        models.load_artifact(saved_model)


def test_v1_matrix_and_artifact_rejected_before_predict():
    estimator = Mock(n_features_in_=139)
    artifact = dict(schema=models.SCHEMA, model=estimator)
    for matrix in (np.zeros((1, 136)), np.full((1, 139), np.nan)):
        with pytest.raises(ValueError):
            models.score_candidates(artifact, matrix)
    with pytest.raises(ValueError, match="v1"):
        models.score_candidates(dict(artifact, schema="v1"), np.zeros((1, 139)))
    estimator.predict.assert_not_called()


def test_small_fit_input_guard_and_weight_normalization_use_mock_only(monkeypatch):
    estimator = Mock()
    estimator.fit.return_value = estimator
    constructor = Mock(return_value=estimator)
    monkeypatch.setattr(models, "HistGradientBoostingRegressor", constructor)
    matrix = np.zeros((2, 139))
    assert models.fit_hgb(matrix, np.array([0., 1.]), np.array([2., 6.])) is estimator
    np.testing.assert_array_equal(estimator.fit.call_args.kwargs["sample_weight"], [.5, 1.5])
    constructor.assert_called_once_with(**models.HGB_PARAMETERS)
    for invalid in (np.zeros((2, 136)), np.full((2, 139), np.inf)):
        with pytest.raises(ValueError, match="Finite v2"):
            models.fit_hgb(invalid, np.array([0., 1.]), np.array([1., 1.]))
    assert constructor.call_count == 1
