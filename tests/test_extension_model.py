import sys
from pathlib import Path

import numpy as np
import mir_eval
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from chord_extension_model import (FEATURE_SCHEMA, FEATURE_SCHEMA_V2, FEATURE_DIMENSIONS,
                                   extend_rows, features_for_rows, temporal_features)
from evaluate_chords import TimedChord, harte_chord


class SeventhEvidence:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, features):
        return np.tile([.01, .98, .01], (len(features), 1))


def test_extension_head_preserves_boundaries_roots_triads_and_existing_labels():
    labels = ["C:maj", "F#:min", "N", "B:dim", "Cadd9", "Bbmaj7", "D:min/E", "Gsus4"]
    rows = [TimedChord(i * .37, (i + 1) * .37, label) for i, label in enumerate(labels)]
    artifact = dict(feature_schema=FEATURE_SCHEMA, threshold=.7, model=SeventhEvidence())
    result, _ = extend_rows(rows, np.zeros((len(rows), 87)), artifact)
    before, after = [harte_chord(r.chord) for r in rows], [harte_chord(r.chord) for r in result]
    np.testing.assert_equal(mir_eval.chord.root(before, after), 1)
    np.testing.assert_equal(mir_eval.chord.triads(before, after), 1)
    assert [(r.start, r.end) for r in rows] == [(r.start, r.end) for r in result]
    assert [r.chord for r in result] == ["C7", "F#m7", "N", "B:dim", "Cadd9", "Bbmaj7", "Dm7/E", "Gsus4"]


def test_disabled_extension_head_is_identity():
    rows = [TimedChord(0, 1, "C")]
    artifact = dict(feature_schema=FEATURE_SCHEMA, threshold=1., model=SeventhEvidence())
    result, _ = extend_rows(rows, np.zeros((1, 87)), artifact)
    assert result == rows


def test_disabled_head_does_not_depend_on_probability_rounding():
    class ConfidentSeventh:
        classes_ = np.array([0, 1, 2])
        def predict_proba(self, features):
            raise AssertionError("Disabled head must not run the classifier")
    rows = [TimedChord(0, 1, "Db/Eb")]
    artifact = dict(feature_schema=FEATURE_SCHEMA, threshold=1., model=ConfidentSeventh())
    result, evidence = extend_rows(rows, np.zeros((1, 87)), artifact)
    assert result == rows
    assert evidence[0]["score_kind"] == "disabled_identity"


def test_empty_extension_timeline_is_supported():
    features, attributes = features_for_rows(np.array([]), [])
    assert features.shape == (0, 87)
    assert attributes == []
    artifact = dict(feature_schema=FEATURE_SCHEMA, threshold=.7, model=SeventhEvidence())
    assert extend_rows([], features, artifact) == ([], [])


def test_v2_preserves_v1_columns_and_uses_no_reference_key_or_extension_label():
    sr = 22050
    t = np.arange(sr, dtype=np.float32)/sr
    audio = sum(np.sin(2*np.pi*f*t) for f in [261.63,329.63,392.,466.16]).astype(np.float32)
    rows = [TimedChord(0,.4,"C"),TimedChord(.4,1,"Am")]
    baseline, attributes = features_for_rows(audio, rows)
    richer, richer_attributes = features_for_rows(audio, rows, feature_schema="v2")
    assert baseline.shape == (2,87) and richer.shape == (2,148)
    np.testing.assert_array_equal(baseline,richer[:,:87])
    assert attributes == richer_attributes
    np.testing.assert_allclose(richer[:,-1],np.log1p([.4,.6]))
    # An existing seventh changes eligibility, never the feature vector.
    equivalent = [TimedChord(0,.4,"C7"),TimedChord(.4,1,"Am7")]
    other, _ = features_for_rows(audio,equivalent,feature_schema=FEATURE_SCHEMA_V2)
    np.testing.assert_array_equal(richer,other)
    empty, _ = features_for_rows(np.array([]),[],feature_schema="v2")
    assert empty.shape == (0,148)


def test_temporal_support_distinguishes_sustained_seventh_from_transient():
    sustained = np.zeros((12,10),dtype=np.float32)
    sustained[[0,4,7],:] = 1
    transient = sustained.copy()
    sustained[10,:] = .8
    transient[10,4:6] = 4
    row = [TimedChord(0,1,"C")]
    stable = temporal_features(sustained,row,[(0,0,0)],10,1)[0]
    brief = temporal_features(transient,row,[(0,0,0)],10,1)[0]
    assert stable[48+10] == 1 and brief[48+10] == pytest.approx(.2)
    assert stable[12+10] > brief[12+10]  # median seventh support
    assert stable[36+10] < brief[36+10]  # transient variance


def test_temporal_frame_clock_is_centered_and_subhop_fallback_uses_interval_center():
    chroma = np.zeros((12,3),dtype=np.float32)
    chroma[0,:] = 1
    chroma[10,0] = 1
    chroma[11,1] = 1
    rows = [TimedChord(0,.1,"C"),TimedChord(.1,.2,"C"),TimedChord(.052,.06,"C")]
    features = temporal_features(chroma,rows,[(0,0,0)]*3,10,1)
    assert features[0,48+10] == 1 and features[0,48+11] == 0
    assert features[1,48+10] == 0 and features[1,48+11] == 1
    assert features[2,48+11] == 1  # nearest centered frame at .1s


def test_temporal_features_are_transposition_invariant_and_silence_is_finite():
    rng = np.random.default_rng(20260926)
    chroma = rng.random((12,12),dtype=np.float32)
    rows = [TimedChord(0,.6,"C"),TimedChord(.6,1.2,"Am")]
    first = temporal_features(chroma,rows,[(0,0,0),(9,1,0)],10,1)
    shifted = temporal_features(np.roll(chroma,3,axis=0),rows,[(3,0,0),(0,1,0)],10,1)
    np.testing.assert_allclose(first,shifted,atol=1e-7)
    silent = temporal_features(np.zeros_like(chroma),rows,[(0,0,0),(9,1,0)],10,1)
    assert np.isfinite(silent).all() and not silent[:,:60].any()


def test_v2_extension_preserves_upstream_semantics_and_rejects_wrong_schema_dimensions():
    labels = ["C", "F#:min", "N", "B:dim", "Cadd9", "Bbmaj7", "D:min/E", "Gsus4"]
    rows = [TimedChord(i*.37,(i+1)*.37,label) for i,label in enumerate(labels)]
    artifact = dict(feature_schema=FEATURE_SCHEMA_V2,threshold=.7,model=SeventhEvidence())
    features = np.zeros((len(rows),FEATURE_DIMENSIONS[FEATURE_SCHEMA_V2]))
    result, _ = extend_rows(rows,features,artifact,feature_schema="v2")
    assert [r.chord for r in result] == ["C7","F#m7","N","B:dim","Cadd9","Bbmaj7","Dm7/E","Gsus4"]
    assert [(r.start,r.end) for r in result] == [(r.start,r.end) for r in rows]
    np.testing.assert_equal(mir_eval.chord.root([harte_chord(r.chord) for r in rows],[harte_chord(r.chord) for r in result]),1)
    np.testing.assert_equal(mir_eval.chord.triads([harte_chord(r.chord) for r in rows],[harte_chord(r.chord) for r in result]),1)
    with pytest.raises(ValueError,match="schema mismatch"):
        extend_rows(rows,features,artifact,feature_schema="v1")
    with pytest.raises(ValueError,match="148 feature columns"):
        extend_rows(rows,np.zeros((len(rows),87)),artifact)
    features[0,0] = np.nan
    with pytest.raises(ValueError,match="finite"):
        extend_rows(rows,features,artifact)


def test_estimator_dimension_is_checked_without_requiring_it_on_old_mocks():
    estimator = SeventhEvidence()
    estimator.n_features_in_ = 87
    artifact = dict(feature_schema=FEATURE_SCHEMA_V2,threshold=.7,model=estimator)
    with pytest.raises(ValueError,match="classifier feature dimension"):
        extend_rows([TimedChord(0,1,"C")],np.zeros((1,148)),artifact)
