"""Semantic and clock regressions for independent bass recognition."""
import json
import numpy as np
import pytest
from app.web import bass_voice as bass
from app.web import stem_bass_harmony as harmony
from app.web.stem_harmony import LABELS


def prior(**labels):
    p=np.full(120,1e-8)
    for label,value in labels.items():p[LABELS.index(label)]=value
    return p/p.sum()


@pytest.mark.parametrize('shift',range(12))
def test_inversion_keeps_chord_root_in_all_transpositions(shift):
    root=bass.NAMES[shift];low=bass.NAMES[(shift+4)%12]
    p=prior(**{f'{root}:maj':.9,f'{low}:min':.1})
    tone=np.zeros(12);tone[(np.array([0,4,7])+shift)%12]=1
    assert harmony.refine_label(p,f'{root}:maj',tone,low)==f'{root}:maj/{low}'


def test_bass_does_not_rewrite_unchanged_seventh_quality():
    p=prior(**{'C:maj':.8,'C:maj7':.2})
    assert harmony.refine_label(p,'C:maj7',np.ones(12),'E')=='C:maj7/E'


def test_nonchord_pedal_is_not_forced_into_a_new_root():
    assert harmony.refine_label(prior(**{'B:maj':1}),'B:maj',np.ones(12),'C#')=='B:maj/C#'


def test_bass_does_not_invent_third_root_against_acoustic_winner():
    p=prior(**{'B:maj':.55,'F#:7':.4,'C#:min7':.05})
    assert harmony.refine_label(p,'B:maj',np.ones(12),'C#')=='B:maj/C#'


@pytest.mark.parametrize('base',[37,49,61,73])
def test_learned_lowest_voice_survives_octave_transposition(base):
    # Same voicing moved from C#2 through C#5; no hard "bass below C4" rule.
    model=bass.load_model();sr=22050;t=np.arange(sr*2)/sr;y=np.zeros_like(t)
    for note,amp in [(base,.2),(base+7,.1),(base+12,.1)]:
        frequency=440*2**((note-69)/12)
        for harmonic in range(1,4):
            y+=amp/harmonic*np.sin(2*np.pi*frequency*harmonic*t)
    result=bass.estimate(model,bass.spectrum(y,sr),.2,1.8)
    assert result['note']=='C#'


def test_moderate_beats_are_combined_before_abstention(monkeypatch):
    class Model:
        bass_min_score_=.6
        bass_min_margin_=.1
    def estimate(m,e,a,b):
        return dict(note=None,candidate='C#' if a<1.54 else 'B' if a<2.28 else 'D#',
                    score=.76 if a<.76 else .576 if a<1.54 else .43,margin=.3)
    monkeypatch.setattr(bass,'estimate',estimate)
    assert bass.summarize_beats(Model(),None,0,3.02,[0,.76,1.54,2.28,3.02])['note']=='C#'


def test_e_chord_is_not_forced_to_bsus4_by_shared_b_note():
    p=prior(**{'E:maj':.65,'B:sus4':.25,'B:maj':.1})
    tone=np.zeros(12);tone[[4,8,11]]=1
    assert harmony.refine_label(p,'B:sus4',tone,'E')=='E:maj'


def test_silence_and_uncertainty_preserve_original():
    assert harmony.refine_label(np.ones(120)/120,'N',np.ones(12),'D')=='N'
    assert harmony.refine_label(np.ones(120)/120,'B:maj',np.ones(12),None)=='B:maj'
    e=bass.spectrum(np.zeros(44100),44100)
    class NeverCalled:
        def predict_proba(self,_):raise AssertionError('Silence must abstain')
    assert bass.estimate(NeverCalled(),e,0,1)['note'] is None


def test_duration_vote_ignores_loud_final_passing_tone(monkeypatch):
    def estimate(model,e,a,b):return dict(note='E' if a<3 else 'B',score=.99)
    monkeypatch.setattr(bass,'estimate',estimate)
    result=bass.summarize_beats(None,None,0,4,[0,1,2,3,4])
    assert result['note']=='E' and result['coverage']==.75


def test_short_pickup_does_not_outvote_long_bass(monkeypatch):
    monkeypatch.setattr(bass,'estimate',lambda m,e,a,b:dict(note='B' if a<.2 else 'D#'))
    assert bass.summarize_beats(None,None,0,2,[0,.1,.2,2])['note']=='D#'


def test_tied_or_unsupported_bass_abstains(monkeypatch):
    monkeypatch.setattr(bass,'estimate',lambda m,e,a,b:dict(note='B' if a<1 else 'D#'))
    assert bass.summarize_beats(None,None,0,2,[0,1,2])['note'] is None


def test_publication_preserves_original_corrections_and_key(tmp_path):
    original=dict(sources=[dict(id='old',rows=[])],default_source='old',key_analysis={'key':'B'},manual={'keep':1})
    (tmp_path/'analysis.json').write_text(json.dumps(original))
    rows=[dict(id=0,start=0,end=1,label='B:maj/D#',candidates=[])]
    first=harmony.publish_result(tmp_path,rows,{},False)
    rows=[dict(rows[0],label='E:maj')]
    second=harmony.publish_result(tmp_path,rows,{},True)
    result=json.loads((tmp_path/'analysis.json').read_text())
    assert result['default_source']==second
    assert result['key_analysis']==original['key_analysis'] and result['manual']==original['manual']
    assert {s['id'] for s in result['sources']}=={'old',first,second}
    assert (tmp_path/(first+'.lab')).read_text().strip().endswith('B:maj/D#')


def test_absolute_slash_is_understood_by_existing_parser():
    from app.chord_semantics_v2 import parse_chord
    chord=parse_chord('B:maj/D#')
    assert chord.root_pc==11 and chord.bass_pc==3 and chord.explicit_bass


@pytest.mark.parametrize('make_default',[False,True])
def test_fusion_job_publishes_bass_revision_without_replacing_earlier_sources(tmp_path,monkeypatch,make_default):
    from app.web import fusion_worker,stem_harmony,stems_worker
    from app.web.storage import signature
    run=tmp_path/'run';(run/'retained10').mkdir(parents=True)
    (run/'retained10/posterior.npz').touch()
    (run/'analysis.json').write_text(json.dumps(dict(sources=[dict(id='old',rows=[])],default_source='old',key_analysis={'key':'B'})))
    audio=tmp_path/'audio.wav';audio.write_bytes(b'original audio identity')
    base=tmp_path/'stems';base.mkdir();(base/'manifest.json').write_text('{}')
    job_path=tmp_path/'job.json';job_path.write_text(json.dumps(dict(project_id='p',model='bs_roformer_sw',make_default=make_default)))
    project=dict(id='p',current_run='r',audio='audio.wav')
    class Store:
        def job_path(self,_):return job_path
        def project(self,_):return dict(project)
        def project_dir(self,_):return tmp_path
        def run_dir(self,*_):return run
        def save_project(self,p):project.update(p)
    def recognize(label):
        def action(*args,**kwargs):
            return [dict(id=0,start=0,end=1,label=label,candidates=[])],dict(source_audio_sha256=signature(audio)['sha256'],segments=1,seconds=.1)
        return action
    monkeypatch.setattr(stems_worker,'stems_dir',lambda *args:base)
    monkeypatch.setattr(stem_harmony,'recognize',recognize('B:maj'))
    monkeypatch.setattr(harmony,'recognize',recognize('B:maj/C#'))
    fusion_worker.execute(Store(),'j')
    result=json.loads((run/'analysis.json').read_text())
    assert len(result['sources'])==3 and result['key_analysis']=={'key':'B'}
    new=result['stem_bass_harmony']['source_id']
    assert result['default_source']==(new if make_default else 'old')
    assert next(s for s in result['sources'] if s['id']==new)['rows'][0]['label']=='B:maj/C#'
