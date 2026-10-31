import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import numpy as np
import pytest
from evaluate_chords import TimedChord
from cached_sequence_constraints_v1 import decode,switch_costs,local_key_adjustment,repetition_adjustment
from cached_sequence_constraints_v1 import soft_snap_to_beats

def fixture_rows(durations=(1.,1.,1.)):
 raw=[];groups=[];left=0
 for i,d in enumerate(durations):
  raw.append(TimedChord(left,left+d,'C'));left+=d
  groups.append(dict(start=2*i,end=2*i+2,candidates=[dict(label='C'),dict(label='C7')]))
 return raw,groups

def test_switch_penalty_suppresses_brief_seventh_but_preserves_sustained_evidence():
 raw,groups=fixture_rows()
 scores=[.5,.4,.5,.54,.5,.4]
 assert [r.chord for r in decode(raw,groups,scores)[0]]==['C','C7','C']
 assert [r.chord for r in decode(raw,groups,scores,transition_costs=[0,.02,.02])[0]]==['C','C','C']
 raw,groups=fixture_rows((1.,3.,1.))
 assert [r.chord for r in decode(raw,groups,scores,transition_costs=[0,.02,.02])[0]]==['C','C7','C']

def test_gap_resets_sequence_influence():
 raw,groups=fixture_rows((1.,1.));raw[1]=TimedChord(2.,3.,'C')
 rows,_=decode(raw,groups,[.5,.8,.7,.71],transition_costs=[0,100])
 assert [r.chord for r in rows]==['C7','C']
 assert [(r.start,r.end) for r in rows]==[(0.,1.),(2.,3.)]

def test_root_and_no_chord_policy_remain_local_and_exact():
 raw=[TimedChord(0.,1.,'C'),TimedChord(1.,2.,'N')]
 groups=[dict(start=0,end=3,candidates=[dict(label='C'),dict(label='G7'),dict(label='C7')]),
         dict(start=3,end=5,candidates=[dict(label='N'),dict(label='C')])]
 assert [r.chord for r in decode(raw,groups,[.4,1.,.7,.1,1.])[0]]==['C7','N']

@pytest.mark.parametrize('costs',[[0,-1,0],[0,np.nan,0],[0,1]])
def test_invalid_transition_costs_rejected(costs):
 raw,groups=fixture_rows()
 with pytest.raises(ValueError):decode(raw,groups,[.5,.4]*3,transition_costs=costs)

def test_beat_and_downbeat_relax_penalty_without_retiming():
 raw,groups=fixture_rows((1.,3.,1.));beats=np.array([[0,1],[1,2],[2,3],[3,4],[4,1],[5,2]])
 costs=switch_costs(raw,beats);base=switch_costs(raw)
 assert costs[1]==pytest.approx(base[1]*.5)
 assert costs[2]==pytest.approx(base[2]*.25)
 assert [(r.start,r.end) for r in raw]==[(0.,1.),(1.,4.),(4.,5.)]

def test_ambiguous_context_adds_no_key_evidence_and_never_excludes_c7():
 raw,groups=fixture_rows();context=np.ones((3,4,12))/12;coverage=np.ones((3,4))
 bonus,details=local_key_adjustment(raw,groups,context,coverage)
 assert np.array_equal(bonus,np.zeros(6))
 assert all(d['confidence']==0 for d in details)
 assert [r.chord for r in decode(raw,groups,[.4,.8]*3)[0]]==['C7']*3

def test_nonlocal_repetition_uses_frozen_predictions_and_rejects_incomplete_context():
 raw=[TimedChord(4.,5.,'C'),TimedChord(24.,25.,'C')]
 groups=[dict(start=0,end=2,candidates=[dict(label='C'),dict(label='C7')]),dict(start=2,end=4,candidates=[dict(label='C'),dict(label='C7')])]
 context=np.ones((2,4,12));coverage=np.ones((2,4))
 bonus,neighbors=repetition_adjustment(raw,groups,context,coverage,['C7','C7'])
 assert neighbors[0][0]['index']==1
 assert bonus.tolist()==pytest.approx([0,.02,0,.02])
 coverage[0,0]=.5
 assert np.array_equal(repetition_adjustment(raw,groups,context,coverage,['C7','C7'])[0],np.zeros(4))

def test_soft_beat_snap_moves_both_shared_edges_and_retains_gaps():
 raw=[TimedChord(0.,1.1,'C'),TimedChord(1.1,2.,'G'),TimedChord(3.1,4.,'F')]
 beats=np.array([[0,1],[1,2],[2,3],[3,4],[4,1]])
 snapped,details=soft_snap_to_beats(raw,beats)
 assert [(r.start,r.end,r.chord) for r in snapped]==[(0.,1.05,'C'),(1.05,2.,'G'),(3.1,4.,'F')]
 assert details['moved_boundaries']==1
 assert sum(r.end-r.start for r in snapped)==pytest.approx(sum(r.end-r.start for r in raw))
 import json
 assert json.loads(json.dumps(details,allow_nan=False))['moved_boundaries']==1

def test_soft_beat_snap_leaves_far_boundaries_and_raw_edges_unchanged():
 raw=[TimedChord(.3,1.4,'C'),TimedChord(1.4,2.3,'G')]
 snapped,details=soft_snap_to_beats(raw,np.array([[0,1],[1,2],[2,3],[3,4]]))
 assert snapped==raw
 assert details['moved_boundaries']==0
