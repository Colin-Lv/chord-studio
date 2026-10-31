import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from evaluate_chords import TimedChord
from largevocab_posteriors_v2 import HEAD_LABELS
from posterior_candidate_proposals_v1 import propose, split_supported_intervals
from posterior_candidate_proposals_v1 import export_candidates
from largevocab_posteriors_v2 import HEAD_DIMENSIONS, make_evidence

def distribution(size,index):
    values=np.zeros(size);values[index]=1.;return values

def test_c7_is_retained_without_a_reference_or_bass_input():
    result=propose(distribution(73,1),distribution(4,2),budget=3)
    assert result[0]['label']=='C:maj(b7)'
    assert len(result)==3
    assert all('/' not in row['label'] for row in result)

def test_major_seventh_and_diminished_seventh_are_distinct():
    assert propose(distribution(73,1),distribution(4,1))[0]['label']=='C:maj(7)'
    dim=HEAD_LABELS['triad'].index('C:dim')
    assert propose(distribution(73,dim),distribution(4,3))[0]['label']=='C:dim(bb7)'

def test_unknown_audio_support_never_fabricates_no_chord():
    assert propose(np.full(73,np.nan),np.full(4,np.nan),coverage=0)==[]

@pytest.mark.parametrize('bad',[np.full(73,np.nan),np.full(73,1.),np.zeros(73),np.zeros(72)])
def test_bad_acoustic_distributions_rejected(bad):
    with pytest.raises(ValueError):propose(bad,distribution(4,0))

@pytest.mark.parametrize('budget',[0,-1,True,2.5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):propose(distribution(73,1),distribution(4,2),budget=budget)

def test_no_chord_has_no_suffix_and_ties_repeat_deterministically():
    result=propose(np.ones(73)/73,np.ones(4)/4,budget=289)
    assert len(result)==289
    assert sum(row['label']=='N' for row in result)==1
    assert not any(row['label'].startswith('N(') for row in result)
    assert result==propose(np.ones(73)/73,np.ones(4)/4,budget=289)

def test_union_does_not_fill_raw_gaps_or_copy_external_labels():
    raw=[TimedChord(1.,2.,'C'),TimedChord(4.,6.,'G')]
    external=[[TimedChord(0.,1.5,'A'),TimedChord(1.5,5.,'F'),TimedChord(5.,8.,'D')]]
    split=split_supported_intervals(raw,external)
    assert [(r.start,r.end,r.chord) for r in split]==[(1.,1.5,'C'),(1.5,2.,'C'),(4.,5.,'G'),(5.,6.,'G')]
    assert sum(r.end-r.start for r in split)==sum(r.end-r.start for r in raw)

@pytest.mark.parametrize('raw',[[TimedChord(1.,2.,'C'),TimedChord(1.5,3.,'G')],
    [TimedChord(-1.,2.,'C')],[TimedChord(1.,float('nan'),'C')],[TimedChord(2.,1.,'C')]])
def test_union_rejects_invalid_raw_timelines(raw):
    with pytest.raises(ValueError):split_supported_intervals(raw,[])

def fixture_evidence():
    probs={name:np.ones((2,size))/size for name,size in HEAD_DIMENSIONS.items()}
    return make_evidence(probs,{'source_audio':{'sha256':'fixture-audio'}},decoded_samples=700)

def test_export_rejects_foreign_audio_cache():
    with pytest.raises(ValueError,match='different declared audio'):
        export_candidates(fixture_evidence(),[TimedChord(0.,.02,'G')],expected_audio_sha256='foreign')

def test_export_marks_timing_identity_unverified_and_preserves_missing_support():
    result=export_candidates(fixture_evidence(),[TimedChord(0.,.02,'G'),TimedChord(1.,2.,'C')],expected_audio_sha256='fixture-audio')
    assert result['status']=='experimental_candidates_not_predictions'
    assert result['anchor_audio_identity_verified'] is False
    assert result['reference_inputs'] is False
    assert len(result['events'][0]['candidates'])==3
    assert result['events'][1]['candidates']==[]
    assert result['events'][1]['acoustic_coverage']==0
