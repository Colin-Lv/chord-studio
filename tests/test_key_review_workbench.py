import json
from pathlib import Path
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.web.server import create_app
from app.web.storage import Store, atomic_json, signature
from app.web import key_review


class NoJobs:
    def __init__(self, store):pass
    def shutdown(self):pass


@pytest.fixture
def client(tmp_path):
    store=Store(tmp_path); p=store.create("review");pid=p["id"];run="a"*32
    audio=store.project_dir(pid)/"audio.wav";audio.write_bytes(b"real identity fixture")
    p.update(current_run=run,audio="audio.wav",audio_identity=signature(audio));store.save_project(p)
    folder=store.run_dir(pid,run);folder.mkdir(parents=True)
    raw=folder/"raw_madmom.lab";raw.write_text("0\t4\tC:maj\n")
    atomic_json(Path(str(raw)+".provenance.json"),dict(audio=signature(audio),lab=signature(raw),duration=4,offset=0))
    result=dict(project_id=pid,run_id=run,audio=signature(audio),duration=4,bars=[],key_analysis=dict(key="G"),sources=[dict(id="raw_madmom",path=raw.name,sha256=signature(raw)["sha256"],rows=[dict(start_text="0",end_text="4",label="C:maj")])])
    atomic_json(folder/"analysis.json",result)
    with TestClient(create_app(tmp_path,NoJobs)) as c:
        yield c,store,pid,result


def act(c,pid,state,action,**kwargs):
    return c.post(f"/api/projects/{pid}/key-review",headers={"X-Harmony-Client":"studio"},json={k:state[k] for k in ("run_id","audio_sha256","revision")}|dict(action=action,**kwargs))


def test_manual_priority_refresh_restore_original_and_immutable(client):
    c,store,pid,result=client;url=f"/api/projects/{pid}/key-review"
    before=(store.run_dir(pid,result['run_id'])/'analysis.json').read_bytes()
    s=c.get(url).json();assert s['effective_key']=='G'
    s=act(c,pid,s,'custom',key='C').json();assert s['effective_key']=='C' and s['derived_key_analysis_stale']
    assert c.get(url).json()['effective_key']=='C'
    s=act(c,pid,s,'restore').json();assert s['effective_key']=='G' and s['manual'] is None
    assert len(s['history'])==2 and s['history'][0]['prediction_visible']
    assert (store.run_dir(pid,result['run_id'])/'analysis.json').read_bytes()==before


def test_cross_window_and_run_audio_conflicts(client):
    c,_,pid,_=client;s=c.get(f'/api/projects/{pid}/key-review').json()
    assert act(c,pid,s,'custom',key='C').status_code==200
    assert act(c,pid,s,'custom',key='D').status_code==409
    newer=c.get(f'/api/projects/{pid}/key-review').json()
    assert act(c,pid,newer|dict(audio_sha256='bad'),'restore').status_code==409
    assert act(c,pid,newer|dict(run_id='b'*32),'restore').status_code==409


def test_missing_suggestion_no_adoption(client):
    c,_,pid,_=client;s=c.get(f'/api/projects/{pid}/key-review').json()
    assert act(c,pid,s,'adopt').status_code==400


def test_identity_mismatch_not_reused(client):
    c,store,pid,result=client
    (store.run_dir(pid,result['run_id'])/'raw_madmom.lab').write_text('0 4 G:maj\n')
    s=key_review.evidence(store,pid,generate=True)
    assert not s['available'] and '身份' in s['reason']


def test_actual_frozen_xuni_inputs():
    store=Store();pid='087d8b0c86414fe6b675cc622f505128'
    try: project=store.project(pid)
    except (FileNotFoundError,HTTPException): pytest.skip('Optional private song fixture is not installed')
    if not (store.project_dir(pid)/project.get('audio','missing')).is_file():
        pytest.skip('Optional private song audio is not installed')
    s=key_review.state(store,pid)
    assert s['automatic']['key']=='G' and s['suggestion']['prediction']['key']=='C'
    assert not s['automatic_override_enabled']
    assert s['suggestion']['prediction']['mode_candidates_by_tonic']['C'][0]['components']['cadence_count']==0


def test_source_corrections_guard_global_revision(client):
    c,_,pid,result=client;s=c.get(f'/api/projects/{pid}/key-review').json();act(c,pid,s,'custom',key='C')
    body=dict(run_id=result['run_id'],source_id='raw_madmom',source_sha256=result['sources'][0]['sha256'],revision=0,key_review_revision=0,edits=[])
    assert c.post(f'/api/projects/{pid}/corrections',headers={'X-Harmony-Client':'studio'},json=body).status_code==409
