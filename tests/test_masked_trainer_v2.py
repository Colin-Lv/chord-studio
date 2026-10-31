"""Supervision integration checks using synthetic labels only."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"app"))
from candidate_targets_v2 import HEADS,ReferenceCoveragePolicy
from evaluate_chords import TimedChord
from multisource_candidates_v2 import candidate_pool
from train_multisource_candidate_reliability_v2 import masked_training_arrays,selection_utility
from experiment_protocol_v2 import freeze_input_rows,INPUT_NAMES


def test_unknown_second_interval_is_removed_without_negative_extension_targets():
    raw=[TimedChord(0,1,"C7"),TimedChord(1,2,"C13")]
    pools=[candidate_pool(event,{"raw_madmom":[event],"chordino":[TimedChord(event.start,event.end,"C9")]}) for event in raw]
    groups=[dict(pool=pool) for pool in pools]
    total=sum(len(pool.candidates) for pool in pools)
    matrix=np.repeat(np.arange(total)[:,None],139,axis=1)
    refs=[dict(start=0,end=1,chord="C9",valid_fields=["ext9"]),
          dict(start=1,end=2,chord="C13",valid_fields=[])]
    x,y,w=masked_training_arrays([dict(raw=raw,groups=groups,features=matrix,reference=refs)],
                                dict.fromkeys(HEADS,1.),ReferenceCoveragePolicy())
    assert len(x)==len(pools[0].candidates)==2
    np.testing.assert_array_equal(x,matrix[:2])
    np.testing.assert_array_equal(y,[0.,1.])
    np.testing.assert_allclose(w,[.5,.5])


def test_no_implicit_root_bass_training_when_only_bass_is_weighted():
    event=TimedChord(0,1,"C")
    song=dict(raw=[event],groups=[dict(pool=candidate_pool(event,{}))],
              features=np.zeros((1,139)),reference=[event])
    with pytest.raises(ValueError,match="No covered reference"):
        masked_training_arrays([song],dict.fromkeys(HEADS,0.)|{"bass":1.},ReferenceCoveragePolicy())


def test_selection_objective_ignores_unknown_fields_and_reports_exposure():
    fields=dict(root=dict(correct_seconds=8.,valid_seconds=10.),
                bass=dict(correct_seconds=0.,valid_seconds=0.))
    result=selection_utility(fields,{"root":1.,"bass":9.})
    assert result["value"]==.8
    assert result["weighted_observed_seconds"]==10.


def test_freezing_hashes_inputs_without_parsing_sealed_references(tmp_path):
    row=dict(id="synthetic",split="sealed_test")
    for name in INPUT_NAMES:
        path=tmp_path/name
        path.write_bytes(b"Opaque input; deliberately not parseable as annotations")
        row[name]=name
    frozen=freeze_input_rows([row],tmp_path)[0]
    assert set(frozen["input_identities"])==set(INPUT_NAMES)
    assert frozen["audio_sha256"]==frozen["input_identities"]["audio"]["sha256"]
    assert Path(frozen["reference"]).is_absolute()
    with pytest.raises(ValueError,match="stale"):
        freeze_input_rows([row|{"audio_sha256":"invented"}],tmp_path)
