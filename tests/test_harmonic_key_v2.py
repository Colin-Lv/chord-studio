"""Hand-verifiable semantics and transposition/rotation properties, not acoustics."""
import copy
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import harmonic_key_v2 as h
from key_profile_baseline import PROFILES,scores
from prepare_harmonic_key_data import isolated_rows
from chord_studio import parse_chord_name


def progression(labels):
    return [dict(start=2*i,end=2*i+2,label=label) for i,label in enumerate(labels)]


def transpose(labels,semitones):
    out=[]
    for label in labels:
        if label in {'N','X'}:out.append(label);continue
        root,suffix=parse_chord_name(label)
        out.append(h.NOTES[(root+semitones)%12]+suffix)
    return out


@pytest.mark.parametrize('tonic',range(12))
@pytest.mark.parametrize('mode',['major','minor'])
def test_every_theoretical_triad_is_equally_compatible(tonic,mode):
    scale=h.MAJOR if mode=='major' else h.NATURAL_MINOR
    for interval,quality in scale.items():
        fit=h.compatibility(interval,quality,mode)
        assert fit['root_compatibility']==fit['quality_compatibility']==1.
    if mode=='minor':
        assert h.compatibility(7,'major',mode)['quality_compatibility']==1.
        assert h.compatibility(11,'dim',mode)['quality_compatibility']==1.


def test_relative_keys_share_compatibility_without_hidden_tonic_bonus():
    a=h.compatibility(0,'minor','minor');c=h.compatibility(9,'minor','major')
    assert a['root_compatibility']==c['root_compatibility']==1
    assert a['quality_compatibility']==c['quality_compatibility']==1
    assert a['functional_degree']=='i' and c['functional_degree']=='vi'
    assert h.compatibility(7,'minor','minor')['quality_compatibility']==1


def test_unknown_quality_is_masked_and_borrowing_is_soft():
    assert not h.compatibility(0,'unknown','minor')['quality_valid']
    assert h.compatibility(0,'sus4','major')['quality_compatibility']==0
    assert 0<h.compatibility(5,'minor','major')['quality_compatibility']<1
    assert h.quality_of('Bm7b5')=='dim'
    assert h.quality_of('Bdim7')=='dim'
    assert h.quality_of('Cmaj7/E')=='major'


@pytest.mark.parametrize('labels',['C G Am F','Am Dm Em Am','Am Dm E Am','Am F C G','C Am Fm Bb Dm G C','Am N F X C G'])
def test_entire_hierarchy_is_transposition_invariant(labels):
    labels=labels.split()*2
    original=h.analyze(progression(labels),len(labels)*2)
    for shift in range(12):
        moved=h.analyze(progression(transpose(labels,shift)),len(labels)*2)
        for tonic in range(12):
            for mode in ['major','minor']:
                a=next(row for row in original['mode_candidates_by_tonic'][h.NOTES[tonic]] if row['mode']==mode)
                b=next(row for row in moved['mode_candidates_by_tonic'][h.NOTES[(tonic+shift)%12]] if row['mode']==mode)
                assert b['components']==pytest.approx(a['components'],abs=1e-12)
                assert b['score']==pytest.approx(a['score'],abs=1e-12)
        original_scores={row['pitch_class']:row['score'] for row in original['tonic_candidates']}
        moved_scores={row['pitch_class']:row['score'] for row in moved['tonic_candidates']}
        assert all(moved_scores[(pc+shift)%12]==pytest.approx(value,abs=1e-12) for pc,value in original_scores.items())


@pytest.mark.parametrize('mode',['major','minor'])
def test_all_known_loop_rotations_have_equal_loop_evidence(mode):
    for family in h.LOOPS[mode]:
        for offset in range(len(family)):
            moved=family[offset:]+family[:offset]
            assert h.loop_evidence(moved,mode)==pytest.approx(1.)
            assert h.loop_evidence(moved*4,mode)==pytest.approx(1.)


def test_major_minor_cadences_and_predominants_are_symmetric():
    a=h.components(h.event_rows(progression(['Dm','G','C']),6),0,'major')
    b=h.components(h.event_rows(progression(['Bdim','E','Am']),6),9,'minor')
    c=h.components(h.event_rows(progression(['Bdim','Em','Am']),6),9,'minor')
    for name in ['cadence','pair','cadence_count']:
        assert a[name]==b[name]==c[name]


def test_schema_contains_independently_auditable_tonic_and_mode():
    labels=['Am','F','C','G']*4
    out=h.analyze(progression(labels),len(labels)*2)
    assert len(out['tonic_candidates'])==12
    assert len(out['mode_candidates_by_tonic'])==12
    assert len(out['final_key_candidates'])==24
    assert out['key']==out['tonic']+('m' if out['mode']=='minor' else '')
    assert out['final_key_candidates'][0]['key']==out['key']


def test_unpitched_and_invalid_inputs():
    assert h.analyze(progression(['N','X']),4)['key'] is None
    with pytest.raises(ValueError):h.analyze([dict(start=2,end=1,label='C')],4)
    with pytest.raises(ValueError):h.analyze(progression(['C']),float('nan'))
    with pytest.raises(ValueError):h.analyze(progression(['C']),2,[dict(start=0,end=1),dict(start=.5,end=2)])


def test_local_window_shift_invariance_and_unique_bar_assignment():
    labels=['C','G','Am','F']*8+['D','A','Bm','G']*4
    bars=[dict(bar=i+1,start=2*i,end=2*i+2) for i in range(len(labels))]
    first=h.analyze(progression(labels),len(labels)*2,bars)
    moved=h.analyze(progression(transpose(labels,2)),len(labels)*2,bars)
    a=first['window_evidence'];b=moved['window_evidence']
    assert len({row['bar'] for row in a['bar_assignments']})==a['valid_bars']
    assert a['decisive_bars']<=a['valid_bars']
    assert [row['stable'] for row in a['windows']]==[row['stable'] for row in b['windows']]
    assert [row['margin'] for row in a['windows']]==pytest.approx([row['margin'] for row in b['windows']])


def test_new_groups_exclude_all_old_connected_tracks():
    rows=[dict(id='old',eligible=True,selected=True,group_tokens=['artist:A','work:old']),
          dict(id='version',eligible=True,selected=False,group_tokens=['artist:B','work:old']),
          dict(id='artist',eligible=True,selected=False,group_tokens=['artist:B','work:new']),
          dict(id='fresh',eligible=True,selected=False,group_tokens=['artist:C','work:fresh'])]
    out=isolated_rows(rows)
    assert [row['id'] for row in out if row['harmonic_selected']]==['fresh']
    assert len({row['group'] for row in out if row['harmonic_selected']}&{row['group'] for row in out if row['selected']})==0


@pytest.mark.parametrize('mode',['major','minor'])
def test_profile_source_scores_and_pitch_shift_property(mode):
    profile=np.array(PROFILES[mode])[:,None]
    base=scores(profile,1.)
    assert base['key']==('C' if mode=='major' else 'Cm')
    for shift in range(12):
        result=scores(np.roll(profile,shift,axis=0),1.)
        assert result['key']==h.NOTES[shift]+('m' if mode=='minor' else '')
        assert result['candidates'][0]['score']==pytest.approx(1.)


def test_profile_missing_uniform_nonfinite_and_time_axis():
    assert scores(np.ones((12,4)),4*512/22050)['key'] is None
    assert scores(np.empty((12,0)),1.)['key'] is None
    with pytest.raises(ValueError):scores(np.full((12,1),float('nan')),1.)
    with pytest.raises(ValueError):scores(np.ones((12,100)),.01)


def test_seeded_arbitrary_progressions_have_shift_invariant_scores():
    rng=np.random.default_rng(2026100602)
    suffixes=['','m','dim','aug','sus2','sus4']
    for _ in range(8):
        labels=[h.NOTES[int(rng.integers(12))]+suffixes[int(rng.integers(len(suffixes)))] for _ in range(12)]
        original=h.analyze(progression(labels),len(labels)*2)
        for shift in range(1,12):
            moved=h.analyze(progression(transpose(labels,shift)),len(labels)*2)
            a={row['pitch_class']:row['score'] for row in original['tonic_candidates']}
            b={row['pitch_class']:row['score'] for row in moved['tonic_candidates']}
            assert all(b[(tonic+shift)%12]==pytest.approx(value,abs=1e-12) for tonic,value in a.items())


def test_selection_reader_never_opens_sealed_predictions(tmp_path):
    import json
    from evaluate_harmonic_key_v2 import records
    split=tmp_path/'split.json'
    split.write_text(json.dumps(dict(rows=[dict(id='sealed',harmonic_selected=True,harmonic_split='harmonic_new_sealed')])) )
    assert records(split,tmp_path/'must_not_read',tmp_path/'must_not_read',{'harmonic_dev','harmonic_calibration'})==[]


def test_tonic_mode_metrics_are_independent_and_cover_missing():
    from evaluate_harmonic_key_v2 import metrics
    rows=[dict(reference='C',test='Cm'),dict(reference='Am',test='C'),
          dict(reference='G',test='G'),dict(reference='Dm',test=None)]
    result=metrics(rows,'test')
    assert result['tonic_accuracy']==.5
    assert result['mode_accuracy']==.25
    assert result['full_key_accuracy']==.25
    assert result['mode_given_correct_tonic']==.5
    assert result['predicted_modes']==dict(major=2,minor=1,missing=1)
    assert result['confusions']['parallel_mode']==1
    assert result['confusions']['relative_major_minor']==1
    assert result['confusions']['exact']==1
    assert result['confusions']['missing']==1


def test_silent_unknown_gaps_do_not_create_cadence_or_loops():
    events=progression(['G','N','C'])
    result=h.analyze(events,6.)
    evidence=result['mode_candidates_by_tonic']['C'][0]['components']
    assert evidence['cadence_count']==0
    assert evidence['contiguous_runs']==2
    labels=['C','G','X','Am','F']
    evidence=h.analyze(progression(labels),10.)['mode_candidates_by_tonic']['C'][0]['components']
    assert evidence['loop']==0


@pytest.mark.parametrize('field,value',[('cadence_weight',float('nan')),('root_weight',-1),('window_bars',0),('step_bars',1.5)])
def test_policy_rejects_invalid_fields(field,value):
    with pytest.raises(ValueError):h.Policy(**{field:value})
