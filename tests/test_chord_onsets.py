import numpy as np
import pytest
from app.web.chord_onsets import align_recovered_onsets,spectral_profile,onset_evidence
from app.web.chord_onsets import attack_chord,resolve_recovered_attacks


def fixture(beat=.75):
    rows=[dict(start=0,end=3.6*beat,label='D#:7'),dict(start=3.6*beat,end=4.6*beat,label='C#:maj'),dict(start=4.6*beat,end=6*beat,label='D#:7')]
    repair=dict(start=rows[1]['start'],end=rows[1]['end'],label='C#:maj',previous='D#:7',minimum_seconds=beat)
    return rows,[repair],np.arange(7)*beat


def profile(v,energy=1):
    v=np.asarray(v,dtype=float);return dict(notes=v/v.sum(),energy=energy)


@pytest.mark.parametrize('beat',[.3,.5,.75,1.1])
@pytest.mark.parametrize('rotation',range(12))
def test_onset_can_precede_stable_label_across_tempos_and_pitches(beat,rotation):
    rows,repairs,beats=fixture(beat)
    old=profile(np.roll([.8,.1,.1]+[0]*9,rotation));new=profile(np.roll([.05,.9,.05]+[0]*9,rotation))
    out,decisions=align_recovered_onsets(rows,repairs,beats,lambda a,b:old if b<=3*beat else new,lambda a,b:0)
    assert decisions[0]['accepted']
    assert out[0]['end']==out[1]['start']==3*beat
    assert out[1]['end']==rows[1]['end'] and out[1]['label']==rows[1]['label']
    assert rows[0]['end']==3.6*beat # Never mutate the source.


@pytest.mark.parametrize('case',['same_context','late_real_onset','different_prefix','quiet','no_chord','missing'])
def test_ambiguous_prefix_does_not_snap_a_genuine_offbeat_change(case):
    rows,repairs,beats=fixture();old=profile([.9,.1,0]);new=profile([.05,.95,0]);other=profile([0,.1,.9])
    def at(a,b):
        if b<=2.25:return new if case=='same_context' else old
        if a>=2.7:return new
        if case=='late_real_onset':return old if a<2.475 else new
        if case=='different_prefix':return other
        if case=='quiet':return profile([.05,.95,0],.001)
        if case=='missing':return None
        return new
    out,decisions=align_recovered_onsets(rows,repairs,beats,at,lambda a,b:1 if case=='no_chord' else 0)
    assert out==rows and not decisions[0]['accepted']


def test_no_recovered_anchor_never_changes_labels_or_inversions():
    rows=[dict(start=0,end=4,label='B:maj/C#')]
    out,decisions=align_recovered_onsets(rows,[],np.arange(5),lambda a,b:None,lambda a,b:0)
    assert out==rows and not decisions


@pytest.mark.parametrize('shift',[-5,0,4])
@pytest.mark.parametrize('late',[False,True])
def test_audio_register_transition_vs_genuine_delayed_onset(shift,late):
    # An audible old bass disappears while the shared upper voice continues.
    # The same setup with a later real change must not be snapped to the beat.
    sr=11025;t=np.arange(int(4.5*sr))/sr
    hz=lambda midi:440*2**((midi-69)/12)
    onset=2.50 if late else 2.25
    old_gate=np.clip((onset-t)/.03,0,1)
    y=.18*np.sin(2*np.pi*hz(61+shift)*t)+old_gate*.28*np.sin(2*np.pi*hz(39+shift)*t)
    evidence=onset_evidence(y,sr)
    args=fixture()
    out,decisions=align_recovered_onsets(*args,lambda a,b:spectral_profile(evidence,a,b),lambda a,b:0)
    assert decisions[0]['accepted']==(not late)
    assert out[1]['start']==(2.25 if not late else args[0][1]['start'])


@pytest.mark.parametrize('root',range(12))
@pytest.mark.parametrize('quality,third',[('maj',4),('min',3)])
def test_struck_chord_is_identified_from_attack_without_future_predictions(root,quality,third):
    sr=11025;t=np.arange(3*sr)/sr;hz=lambda midi:440*2**((midi-69)/12)
    y=.005*np.sin(2*np.pi*hz(38)*t)
    envelope=np.where(t>=1,np.exp(-2*np.maximum(t-1,0)),0)
    for interval,amplitude in [(0,.16),(third,.13),(7,.11)]:
        y+=amplitude*np.sin(2*np.pi*hz(60+root+interval)*t)*envelope
    proposal=attack_chord(onset_evidence(y,sr),1,.75)
    assert proposal['accepted']
    assert proposal['label']==['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][root]+':'+quality


def test_single_note_harmonics_are_not_a_played_major_chord():
    sr=11025;t=np.arange(3*sr)/sr;f=261.625565
    y=np.full_like(t,1e-6)
    for harmonic,amplitude in [(1,.2),(2,.1),(3,.12),(4,.05),(5,.1)]:
        y+=amplitude*np.sin(2*np.pi*f*harmonic*t)*(t>=1)
    proposal=attack_chord(onset_evidence(y,sr),1,.75)
    assert proposal is None or not proposal['accepted']


def test_recovered_region_uses_two_independent_attacks_not_old_label_tail(monkeypatch):
    from app.web import chord_onsets as module
    original=[dict(start=0,end=6,label='D#:7')]
    rows=[dict(start=0,end=2.7,label='D#:7'),dict(start=2.7,end=3.45,label='C#:maj'),dict(start=3.45,end=6,label='D#:7')]
    repair=dict(start=2.7,end=3.45,label='C#:maj',previous='D#:7',minimum_seconds=.75)
    proposals={2.25:dict(time=2.25,label='C#:maj',accepted=True),3.75:dict(time=3.75,label='C:min',accepted=True)}
    monkeypatch.setattr(module,'attack_chord',lambda e,t,b:dict(proposals[t]) if t in proposals else None)
    monkeypatch.setattr(module,'spectral_profile',lambda *args:profile([.05,.9,.05]))
    out,decisions=resolve_recovered_attacks(rows,original,[repair],np.arange(9)*.75,None,lambda a,b:0,
        lambda attack,a,b:(attack['label'],{'resolution':'independent_fixture_confirmation'}))
    assert [(r['start'],r['end'],r['label']) for r in out]==[(0,2.25,'D#:7'),(2.25,3.75,'C#:maj'),(3.75,6,'C:min')]
    assert all(d['applied'] for d in decisions)


def acoustic(label,prob=.7):
    from app.web.full_decode import NAMES,QUALITIES
    p=np.full(120,(1-prob)/119)
    root,quality=label.split(':');p[NAMES.index(root)*10+QUALITIES.index(quality)]=prob
    return p


@pytest.mark.parametrize('score',[.578,.599,.601,.606])
def test_weak_c_template_cannot_override_acoustic_dsharp_and_consistent_bass(score):
    from app.web.chord_onsets import reconcile_attack
    bass=dict(note='D#' if score>=.6 else None,score=score,
              beats=[dict(candidate='D#',score=score),dict(candidate='D#',score=score)])
    label,report=reconcile_attack({'label':'C:min'},acoustic('D#:maj'),bass)
    assert label=='D#:maj'
    assert report['resolution']=='frozen_chord_and_consistent_bass'


@pytest.mark.parametrize('label,bass_note',[('G#:maj','D#'),('C:min','D#'),('B:maj','C#')])
def test_real_inversion_or_pedal_is_not_forced_into_a_bass_root_chord(label,bass_note):
    from app.web.chord_onsets import reconcile_attack
    bass=dict(note=bass_note,beats=[dict(candidate=bass_note),dict(candidate=bass_note)])
    actual,report=reconcile_attack({'label':label},acoustic(label),bass)
    assert actual==label and report['resolution']=='template_and_frozen_agree'


@pytest.mark.parametrize('votes',[['D#','F'],['D#'],['D#',None]])
def test_conflicting_or_incomplete_bass_cannot_confirm_a_root_change(votes):
    from app.web.chord_onsets import reconcile_attack
    actual,report=reconcile_attack({'label':'C:min'},acoustic('D#:maj'),{'beats':[dict(candidate=v) for v in votes]})
    assert actual is None and report['resolution']=='conflicting_chord_evidence'
