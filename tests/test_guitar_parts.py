"""Guitar results retain source/mode identity and cannot corrupt analysis assets."""
from uuid import uuid4
import threading
import pytest
from fastapi.testclient import TestClient
from app.web.server import create_app
from app.web.storage import atomic_json,signature,read_json
from app.web.stems_worker import stems_dir
from app.web.guitar_parts_worker import RECIPES,parts_manifest,valid_manifest

class FakeJobs:
    def __init__(self,store):self.store=store
    def submit(self,project_id,profile,**kwargs):return dict(id=uuid4().hex,project_id=project_id,profile=profile,**kwargs)
    def shutdown(self):pass

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setenv('CHORD_DESKTOP_DATA',str(tmp_path/'assets'))
    monkeypatch.setattr('app.web.guitar_parts_worker.fingerprint',lambda mode:{'weights':'v1'})
    with TestClient(create_app(tmp_path/'store',manager_factory=FakeJobs)) as client:
        headers={'X-Harmony-Client':'studio'}
        p=client.post('/api/upload',headers=headers,files={'file':('song.wav',b'RIFFx','audio/wav')}).json()
        store=client.app.state.store;sha=signature(store.project_dir(p['id'])/p['audio'])['sha256']
        base=stems_dir(sha,'bs_roformer_sw');(base/'wav').mkdir(parents=True)
        (base/'wav/guitar.wav').write_bytes(b'guitar-parent')
        atomic_json(base/'manifest.json',dict(tracks=[dict(id='guitar',file='guitar.mp3',wav='wav/guitar.wav')]))
        yield client,p,sha,base,headers

def manifest(sha,base,mode):
    gen=uuid4().hex;out=base/'guitar_parts'/gen;out.mkdir(parents=True)
    tracks=[]
    for name,*_ in RECIPES[mode]['parts']:
        for ext in ['mp3','wav']:(out/(name+'.'+ext)).write_bytes((mode+name).encode())
        tracks.append(dict(id=name,file=f'guitar_parts/{gen}/{name}.mp3',wav=f'guitar_parts/{gen}/{name}.wav'))
    data=dict(model='bs_roformer_sw',mode=mode,generation=gen,tracks=tracks,
        parent_sha256=signature(base/'wav/guitar.wav')['sha256'],fingerprint={'weights':'v1'})
    atomic_json(parts_manifest(sha,'bs_roformer_sw',mode),data)
    return data

def test_modes_remain_independent_and_model_bound(fixture):
    client,p,sha,base,_=fixture;pid=p['id']
    a=manifest(sha,base,'timbre');b=manifest(sha,base,'duet')
    catalog=client.get(f'/api/projects/{pid}/stems/guitar-parts-catalog').json()
    assert {r['mode'] for r in catalog['results']}=={'timbre','duet'}
    for mode in ['timbre','duet']:
        r=client.get(f'/api/projects/{pid}/stems/guitar-parts?model=bs_roformer_sw&mode={mode}').json()
        for field in ['urls','wav_urls']:
            for url in r[field].values():
                assert '?model=bs_roformer_sw' in url
                assert client.get(url).content.startswith(mode.encode())
    assert client.get(f'/api/projects/{pid}/stems/guitar-parts?model=htdemucs&mode=timbre').status_code==404
    assert a['generation']!=b['generation']

def test_source_or_weights_change_invalidates_without_deleting(fixture,monkeypatch):
    client,p,sha,base,_=fixture
    old=manifest(sha,base,'timbre');asset=base/old['tracks'][0]['wav']
    assert valid_manifest(sha,'bs_roformer_sw','timbre')
    monkeypatch.setattr('app.web.guitar_parts_worker.fingerprint',lambda mode:{'weights':'v2'})
    assert valid_manifest(sha,'bs_roformer_sw','timbre') is None and asset.exists()
    monkeypatch.setattr('app.web.guitar_parts_worker.fingerprint',lambda mode:{'weights':'v1'})
    (base/'wav/guitar.wav').write_bytes(b'updated-parent')
    assert valid_manifest(sha,'bs_roformer_sw','timbre') is None and asset.exists()

def test_reject_unsafe_manifest_and_request(fixture):
    client,p,sha,base,headers=fixture;pid=p['id']
    data=manifest(sha,base,'timbre')
    outside=base.parent/'private.wav';outside.write_bytes(b'private')
    data['tracks'][0]['wav']='../private.wav';atomic_json(parts_manifest(sha,'bs_roformer_sw','timbre'),data)
    assert valid_manifest(sha,'bs_roformer_sw','timbre') is None
    assert client.post(f'/api/projects/{pid}/guitar-parts',headers=headers,json={'mode':'../bad'}).status_code==422
    assert client.post(f'/api/projects/{pid}/guitar-parts',headers=headers,json={'model':'../bad'}).status_code==400
    assert client.post(f'/api/projects/{pid}/guitar-parts',headers=headers,json={'model':'htdemucs'}).status_code==409
    assert client.get(f'/api/projects/{pid}/stems/guitar_parts/invalid/private.wav?model=bs_roformer_sw').status_code==404

def test_submission_keeps_recipe_and_completion_preserves_analysis(fixture,monkeypatch,fake_job_process):
    from app.web.jobs import Jobs
    client,p,sha,base,headers=fixture;pid=p['id'];store=client.app.state.store
    row=client.post(f'/api/projects/{pid}/guitar-parts',headers=headers,json={'mode':'duet','force':True}).json()
    assert row['profile']=='guitar_parts' and row['guitar_mode']=='duet' and row['force']
    p.update(current_run='locked-analysis',stems={'bs_roformer_sw':{'sha256':'kept'}});store.save_project(p)
    atomic_json(store.job_path(row['id']),dict(row,status='queued'))
    manager=object.__new__(Jobs);manager.store=store;manager.lock=threading.RLock();manager.python='python';manager.processes={};manager.cancel_requested=set()
    fake_job_process()
    manager._execute(row['id'])
    actual=store.project(pid)
    assert actual['current_run']=='locked-analysis' and actual['stems']==p['stems']
    assert read_json(store.job_path(row['id']))['stage_label']=='吉他细分完成'
