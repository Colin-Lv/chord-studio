from pathlib import Path
from uuid import uuid4
import shutil
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from app.web import strings_stems as worker
from app.web.server import create_app
from app.web.storage import atomic_json,read_json,signature
from app.web.stems_worker import stems_dir,stem_wav,stems_for,resolve_stem

class Jobs:
    def __init__(self,store):self.store=store
    def submit(self,pid,profile,**kw):return dict(id=uuid4().hex,project_id=pid,profile=profile,**kw)
    def shutdown(self):pass

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setenv('CHORD_DESKTOP_DATA',str(tmp_path/'data'))
    monkeypatch.setattr(worker,'fingerprint',lambda model:{'model':model,'recipe':'one'})
    monkeypatch.setattr(worker,'_runtime',lambda:('test-python','cpu'))
    monkeypatch.setattr(worker,'_decode_to_wav',shutil.copyfile)
    monkeypatch.setattr(worker,'beat_grid',lambda *a:dict(bpm=120,bpm_source='test',bars=[0],beats=[]))
    monkeypatch.setattr(worker,'_ffmpeg',lambda:'test-encoder')
    def encode(args):Path(args[-1]).write_bytes(b'preview');return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(worker,'_run',encode)
    calls=[]
    def run(args,**kw):
        calls.append(args);out=Path(args[args.index('--outdir')+1]);out.mkdir(parents=True)
        x,sr=sf.read(args[args.index('--input')+1],dtype='float32',always_2d=True)
        for t in stems_for('strings_v2'):sf.write(out/(t['id']+'.wav'),x/7,sr,subtype='FLOAT')
        atomic_json(out/'inference.json',dict(output_frames=len(x),sum_error_relative_db=-130))
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(worker.subprocess,'run',run)
    wav=tmp_path/'song.wav';sf.write(wav,np.ones((4410,2),dtype='float32')*.2,44100,subtype='FLOAT')
    with TestClient(create_app(tmp_path/'store',manager_factory=Jobs)) as client:
        headers={'X-Harmony-Client':'studio'}
        p=client.post('/api/upload',headers=headers,files={'file':('song.wav',wav.read_bytes(),'audio/wav')}).json()
        store=client.app.state.store;p.update(current_run='a'*32,piano_transcription={'generation':'saved'})
        store.save_project(p);job=uuid4().hex;atomic_json(store.job_path(job),dict(project_id=p['id'],status='running',model='strings_v2'))
        yield client,store,p,job,calls,headers

def test_cache_versions_and_immutable_assets(setup):
    c,store,p,job,calls,h=setup
    a=worker.execute(store,job,'strings_v2');b=worker.execute(store,job,'strings_v2')
    assert b['cached'] and a['generation']==b['generation'] and len(calls)==1
    v1=worker.execute(store,job,'strings_v1');assert v1['generation']!=a['generation'] and len(calls)==2
    sha=a['audio_sha256'];assert stem_wav(sha,'strings_v2','guitar').is_file()
    for model in ['strings_v1','strings_v2']:
        url=f'/api/projects/{p["id"]}/stems?model={model}'
        m=c.get(url).json();assert len(m['tracks'])==7 and len(set(t['id'] for t in m['tracks']))==7
        for asset in [m['audio_url'],*m['urls'].values(),*m['wav_urls'].values()]:assert c.get(asset).status_code==200
        assert c.post(f'/api/projects/{p["id"]}/vocal-parts',headers=h,json={'model':model}).status_code==202
    actual=store.project(p['id']);assert actual['current_run']=='a'*32 and actual['piano_transcription']==p['piano_transcription']
    assert set(actual['stems'])=={'strings_v1','strings_v2'}

def test_failed_rebuild_preserves_published_generation(setup,monkeypatch):
    c,store,p,job,calls,h=setup;m=worker.execute(store,job,'strings_v2')
    base=stems_dir(m['audio_sha256'],'strings_v2');before=(base/'manifest.json').read_bytes()
    def fail(*a,**k):raise RuntimeError('simulated GPU failure')
    monkeypatch.setattr(worker.subprocess,'run',fail)
    with pytest.raises(RuntimeError):worker.execute(store,job,'strings_v2',True)
    assert (base/'manifest.json').read_bytes()==before
    assert all((base/t['wav']).is_file() for t in m['tracks'])

def test_changed_recipe_invalidates_cache_without_removing_audio(setup,monkeypatch):
    c,store,p,job,calls,h=setup;m=worker.execute(store,job,'strings_v2');sha=m['audio_sha256']
    assert worker.cached_manifest(sha,'strings_v2',{'recipe':'changed'}) is None
    assert stem_wav(sha,'strings_v2','strings').is_file()
    missing=stems_dir(sha,'strings_v2')/m['tracks'][0]['file'];missing.unlink()
    assert worker.cached_manifest(sha,'strings_v2',m['fingerprint']) is None

def test_generation_paths_cannot_escape(setup):
    c,store,p,job,calls,h=setup;m=worker.execute(store,job,'strings_v2');base=stems_dir(m['audio_sha256'],'strings_v2')
    with pytest.raises(ValueError):resolve_stem(base,'strings',{'tracks':[{'id':'strings','wav':'../secret.wav'}]})
    assert c.get(f'/api/projects/{p["id"]}/stems/generations/bad/source.wav?model=strings_v2').status_code==404

def test_harmonic_view_includes_strings_from_generation(setup,tmp_path):
    from app.web.stem_harmony import build_views
    c,store,p,job,calls,h=setup;m=worker.execute(store,job,'strings_v2');base=stems_dir(m['audio_sha256'],'strings_v2')
    views,ids,groups=build_views(base,m,m['duration'],tmp_path/'view')
    assert 'strings' in groups['accompaniment'] and 'strings' in ids
    y,_=sf.read(views['accompaniment']);np.testing.assert_allclose(y,.2*4/7,atol=1e-7)
