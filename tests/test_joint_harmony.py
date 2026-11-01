"""Default-path wiring, immutable sources, and failures; no acoustic mocks as accuracy."""
import copy,json
from types import SimpleNamespace
import pytest
from app.web import joint_harmony as joint
from app.web import stem_bass_harmony as bass
from app.web.storage import signature


@pytest.fixture
def fixture(tmp_path):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'current recording')
    project=dict(id='p',audio='audio.wav',audio_identity=signature(audio),current_run=None)
    run=tmp_path/'new';run.mkdir()
    result=dict(run_id='new',duration=4,audio=signature(audio),sources=[dict(id='retained10_full',rows=[dict(start=0,end=4,label='B:maj')])],default_source='retained10_full',key_analysis={'key':'B'})
    (run/'analysis.json').write_text(json.dumps(result))
    job_path=tmp_path/'job.json';job_path.write_text(json.dumps(dict(id='job',project_id='p',status='running',profile='retained10')))
    store=SimpleNamespace(project=lambda _:copy.deepcopy(project),project_dir=lambda _:tmp_path,
                          run_dir=lambda *args:run,job_path=lambda _:job_path)
    return store,project,run,job_path,result


def test_first_upload_gets_bass_without_previous_source_or_opt_in(fixture,monkeypatch,tmp_path):
    store,project,run,path,original=fixture;seen=[]
    def separate(s,j,model_override=None,force_override=None):
        seen.append('separate');assert model_override=='bs_roformer_sw' and force_override is False
    monkeypatch.setattr(joint.stems_worker,'execute',separate)
    monkeypatch.setattr(joint.stems_worker,'stems_dir',lambda *args:tmp_path/'stems')
    def recognize(folder,base,out,progress):
        assert folder==run;seen.append('bass')
        return [dict(start=0,end=2,label='B:maj/C#'),dict(start=2,end=4,label='B:maj/D#')],dict(bass_model={'sha256':'bass'},model={'name':'bs_roformer_sw'})
    monkeypatch.setattr(bass,'recognize',recognize)
    result=joint.execute(store,'job',run);after=json.loads((run/'analysis.json').read_text())
    assert seen==['separate','bass']
    assert after['default_source']==result['source_id'] and result['source_id'].startswith('retained10_bass_')
    assert after['sources'][0]==original['sources'][0]
    assert after['joint_harmony']['automatic'] and after['joint_harmony']['status']=='completed'
    assert after['model_identity']['bass_voice']['weights_sha256']=='bass'
    assert after['harmonic_keys'][result['source_id']]['source_id']==result['source_id']
    assert json.loads(path.read_text())['profile']=='retained10'


def test_failure_does_not_publish_baseline_as_success(fixture,monkeypatch):
    store,project,run,path,original=fixture
    def fail(*args,**kwargs):raise RuntimeError('separation failed')
    monkeypatch.setattr(joint.stems_worker,'execute',fail)
    with pytest.raises(RuntimeError,match='separation failed'):joint.execute(store,'job',run)
    assert json.loads((run/'analysis.json').read_text())==original


def test_changed_audio_rejected_before_separation(fixture,monkeypatch):
    store,project,run,path,original=fixture
    (store.project_dir('p')/'audio.wav').write_bytes(b'changed')
    with pytest.raises(ValueError,match='身份不一致'):joint.execute(store,'job',run)


@pytest.mark.parametrize('extra',[False,True])
def test_full_pipeline_always_calls_joint_stage_after_stems(tmp_path,monkeypatch,extra):
    from app.web import pipeline_worker as pipeline
    path=tmp_path/'job.json';path.write_text(json.dumps(dict(id='j',project_id='p',status='running',stages=[],
        pipeline=dict(model='htdemucs',harmony=extra))))
    project={'id':'p','current_run':None};calls=[]
    store=SimpleNamespace(job_path=lambda _:path,project=lambda _:dict(project),save_project=lambda p:project.update(p),
                          run_dir=lambda *args:tmp_path/'new-run')
    def worker(profile):
        def action(*args):
            calls.append(profile)
            if profile=='joint_harmony':
                assert project['current_run'] is None
                assert args[2]==tmp_path/'new-run'
        return action
    monkeypatch.setattr(pipeline,'_worker',worker);pipeline.execute(store,'j')
    assert calls[:3]==['retained10','stems','joint_harmony']
    assert ('fusion' in calls)==extra
    assert project['current_run']=='j'


def test_pipeline_bass_failure_preserves_previous_current_run(tmp_path,monkeypatch):
    from app.web import pipeline_worker as pipeline
    path=tmp_path/'job.json';path.write_text(json.dumps(dict(id='j',project_id='p',status='running',stages=[],pipeline={'harmony':False})))
    project={'id':'p','current_run':'previous'}
    store=SimpleNamespace(job_path=lambda _:path,project=lambda _:dict(project),save_project=lambda p:project.update(p),run_dir=lambda *args:tmp_path/'new')
    def worker(profile):
        def action(*args):
            if profile=='joint_harmony':raise RuntimeError('bass failed')
        return action
    monkeypatch.setattr(pipeline,'_worker',worker)
    with pytest.raises(RuntimeError,match='bass failed'):pipeline.execute(store,'j')
    assert project['current_run']=='previous'


def test_saved_separator_metadata_accepts_object_form(tmp_path):
    previous=dict(default_source='stem_bass_harmony_v2_x',audio={'sha256':'a'},stem_bass_harmony=dict(source_id='stem_bass_harmony_v2_x',model={'name':'bs_roformer_sw'}))
    (tmp_path/'analysis.json').write_text(json.dumps(previous))
    store=SimpleNamespace(run_dir=lambda *args:tmp_path)
    assert bass.selected_bass_model(store,dict(id='p',current_run='old',audio_identity={'sha256':'a'}))=='bs_roformer_sw'
