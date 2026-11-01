"""Tonal reasoning contracts; synthetic controls are not real-song accuracy."""
import copy
import json
import numpy as np
import pytest
from app.harmonic_key_v3 import analyze, chord_features, segment_features, KEYS
from app.key_chord_adapter import parse_rows
from app.web.harmonic_key import refresh


def rows(labels,step=2,start=0):
    return [dict(start=start+i*step,end=start+(i+1)*step,label=c) for i,c in enumerate(labels)]


def test_relative_and_parallel_modulations_are_allowed():
    first=['C:maj','F:maj','G:7','C:maj']*3
    for target in [('A:min','D:min','E:7','A:min'),('C:min','F:min','G:7','C:min')]:
        events=rows(first+list(target)*5)
        result=analyze(events,events[-1]['end'])
        assert [s['key'] for s in result['segments']]==['C','Am' if target[0]=='A:min' else 'Cm']


def test_brief_secondary_dominant_does_not_force_modulation():
    events=rows(['C:maj','F:maj','G:7','C:maj']*3+['D:7','G:maj']+['C:maj','F:maj','G:7','C:maj']*3)
    result=analyze(events,events[-1]['end'])
    assert [s['key'] for s in result['segments']]==['C']
    assert any(t['target']=='G' for t in result['tonicizations'])


def test_subdivision_does_not_multiply_evidence_or_changes():
    events=rows(['C:maj','F:maj','G:7','C:maj']*3+['D:maj','G:maj','A:7','D:maj']*4)
    split=[dict(start=r['start']+i/4*(r['end']-r['start']),end=r['start']+(i+1)/4*(r['end']-r['start']),label=r['label']) for r in events for i in range(4)]
    a,b=analyze(events,events[-1]['end']),analyze(split,events[-1]['end'])
    assert a['segments']==b['segments'] and a['changes']==b['changes']
    np.testing.assert_allclose(segment_features(parse_rows(events)[0]),segment_features(parse_rows(split)[0]))


def test_audio_reference_cannot_override_chords():
    events=rows(['C:maj','F:maj','G:7','C:maj']*4)
    result=analyze(events,events[-1]['end'],acoustic={'key':'F#','confidence':1.0})
    assert result['main_key']=='C'
    assert result['acoustic_used_for_decoding'] is False


def test_full_extension_and_slash_semantics():
    events=rows(['B:maj/D#','F#:maj9','B:maj','F#:7','B:maj'])
    f,_=chord_features(parse_rows(events)[0]);key=KEYS.index((11,'major'))
    assert f[0,key,5]==1 # Slash third is not a new tonic.
    assert f[1,key,0]==0 and f[1,key,7]==0 # E#; maj9 is not V7.
    assert f[2,key,8]==0 and f[4,key,8]==1


def test_no_cadence_across_unknown_span_and_coverage_excludes_gaps():
    events=rows(['C:maj','F:maj','G:7','N','C:maj']*3)
    parsed,_=parse_rows(events);f,_=chord_features(parsed)
    key=KEYS.index((0,'major'))
    assert f[:,key,8].sum()==0
    result=analyze(events,events[-1]['end'])
    assert result['coverage']==pytest.approx(.8)
    assert sum(result['key_seconds'].values())==pytest.approx(events[-1]['end']*.8)
    assert len(result['uncovered'])==3


@pytest.mark.parametrize('events,duration',[
    ([dict(start=0,end=2,label='C'),dict(start=1,end=3,label='G')],3),
    ([dict(start=0,end=4,label='C')],3),
    ([dict(start=0,end=float('nan'),label='C')],3),
    ([],0),
])
def test_invalid_clock_rejected(events,duration):
    with pytest.raises(ValueError):analyze(events,duration)


def test_silence_is_not_c_major():
    assert analyze(rows(['N','X']),4)['status']=='insufficient'


@pytest.mark.parametrize('shift',range(12))
def test_transposition_equivariance(shift):
    names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    events=rows([f'{names[(root+shift)%12]}:{q}' for root,q in [(0,'maj'),(5,'maj'),(7,'7'),(0,'maj')]*4])
    result=analyze(events,events[-1]['end'])
    assert result['main_key']==names[shift] and not result['changes']


def test_refresh_keeps_manual_data_and_separate_source_provenance():
    a=rows(['C:maj','F:maj','G:7','C:maj']*3)
    b=rows(['D:maj','G:maj','A:7','D:maj']*3)
    result=dict(duration=24,run_id='run',audio={'sha256':'audio'},default_source='a',
                sources=[dict(id='a',name='A',sha256='a-sha',rows=a),dict(id='b',name='B',sha256='b-sha',rows=b)],
                key_analysis={'key':'F#','source':'cnn'},key_fusion={'key':'B'},manual={'key':'E'})
    original=copy.deepcopy(result);refresh(result)
    assert result['key_analysis']['key']=='C' and result['harmonic_keys']['b']['main_key']=='D'
    assert result['harmonic_keys']['a']['source_sha256']=='a-sha'
    assert result['manual']==original['manual'] and result['sources']==original['sources']
    assert result['key_analysis_before_v3']==original['key_analysis'] and 'key_fusion' not in result
    repeated=copy.deepcopy(result);refresh(result)
    assert result==repeated
    result['default_source']='b';refresh(result,['b'])
    assert result['key_analysis']['key']=='D' and result['key_analysis_before_v3']==original['key_analysis']


def test_incomplete_legacy_publication_preserves_existing_key():
    result={'sources':[],'key_analysis':{'key':'B'}};refresh(result)
    assert result['harmonic_key_status']['status']=='unavailable' and result['key_analysis']['key']=='B'


def test_isolated_chromatic_chord_is_reviewed_without_its_own_key_region():
    progression=['C:maj','F:maj','G:7','C:maj']*4
    events=rows(progression+['G#:maj']+progression)
    result=analyze(events,events[-1]['end'])
    assert all(len(s['evidence'])>=2 for s in result['segments'])
    assert any(e['chords']==['G#:maj'] and not e['independent_progression'] for e in result['excursions'])


def test_fragmented_source_retains_all_observations_with_bounded_boundary_search():
    events=rows(['C:maj','F:maj','G:7','C:maj']*128,step=.1)
    # Avoid adjacency coalescing at the repeat boundary.
    for i in range(3,len(events),4):events[i]['label']='A:min'
    bars=[dict(start=i*.4,end=(i+1)*.4) for i in range(128)]
    result=analyze(events,events[-1]['end'],bars)
    assert result['parse']['tokens']==512 and result['coverage']==pytest.approx(1)
    assert result['boundary_search'].startswith('bar/time')
