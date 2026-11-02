"""Audio conservation, parent invalidation and model-specific asset routing."""
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.web.server import create_app
from app.web.storage import atomic_json, signature, ROOT
from app.web.stems_worker import stems_dir
from app.web.vocal_parts_runner import split_complement


class Jobs:
    def __init__(self, store): self.store=store
    def submit(self, project_id, profile, **kwargs):
        return dict(id=uuid4().hex,project_id=project_id,profile=profile,**kwargs)
    def shutdown(self): pass


def test_float_complement_preserves_gain_channels_and_samples():
    rng=np.random.default_rng(13)
    parent=rng.normal(0,.5,(1000,2)).astype('float32')
    lead=rng.normal(0,.3,parent.shape).astype('float32')
    residual=split_complement(parent,lead)
    np.testing.assert_allclose(lead+residual,parent,atol=2e-7)
    with pytest.raises(ValueError):split_complement(parent,lead[:-1])
    with pytest.raises(ValueError):split_complement(parent,np.full_like(parent,np.nan))


def test_vocal_cache_parent_and_urls(tmp_path,monkeypatch):
    monkeypatch.setenv('CHORD_DESKTOP_DATA',str(tmp_path/'assets'))
    monkeypatch.setattr('app.web.vocal_parts_worker.model_fingerprint',lambda:dict(weights_sha256='model-v1',config_sha256='config-v1'))
    with TestClient(create_app(tmp_path/'store',manager_factory=Jobs)) as client:
        project=client.post('/api/upload',headers={'X-Harmony-Client':'studio'},
                           files={'file':('song.wav',b'RIFFaudio','audio/wav')}).json()
        store=client.app.state.store;pid=project['id'];audio=store.project_dir(pid)/project['audio']
        sha=signature(audio)['sha256'];model='bs_roformer_sw';folder=stems_dir(sha,model)
        (folder/'wav').mkdir(parents=True);parent=folder/'wav/vocals.wav';parent.write_bytes(b'parent-v1')
        gen=uuid4().hex;target=folder/'vocal_parts'/gen;target.mkdir(parents=True)
        tracks=[]
        for stem in ['lead_vocals','backing_vocals']:
            for ext in ['wav','mp3']:(target/f'{stem}.{ext}').write_bytes(f'{model}-{stem}-{ext}'.encode())
            tracks.append(dict(id=stem,file=f'vocal_parts/{gen}/{stem}.mp3',wav=f'vocal_parts/{gen}/{stem}.wav'))
        atomic_json(folder/'vocal_parts/manifest.json',dict(parent_sha256=signature(parent)['sha256'],tracks=tracks,
                    inference=dict(runner_sha256=signature(ROOT/'app/web/vocal_parts_runner.py')['sha256'],weights_sha256='model-v1',config_sha256='config-v1')))
        url=f'/api/projects/{pid}/stems/vocal-parts?model={model}'
        data=client.get(url).json()
        for key in ['urls','wav_urls']:
            for asset in data[key].values():
                assert '?model=bs_roformer_sw' in asset
                assert client.get(asset).content.startswith(b'bs_roformer_sw-')
        monkeypatch.setattr('app.web.vocal_parts_worker.model_fingerprint',lambda:dict(weights_sha256='model-v2',config_sha256='config-v1'))
        assert client.get(url).status_code==404
        monkeypatch.setattr('app.web.vocal_parts_worker.model_fingerprint',lambda:dict(weights_sha256='model-v1',config_sha256='config-v1'))
        parent.write_bytes(b'parent-v2')
        assert client.get(url).status_code==404
        assert client.get(f'/api/projects/{pid}/stems/wav/vocals.wav?model=../outside').status_code==404
        response=client.post(f'/api/projects/{pid}/vocal-parts',json={'model':model},headers={'X-Harmony-Client':'studio'})
        assert response.status_code==202 and response.json()['profile']=='vocal_parts'


def test_drums_urls_keep_selected_model(tmp_path,monkeypatch):
    monkeypatch.setenv('CHORD_DESKTOP_DATA',str(tmp_path/'assets'))
    with TestClient(create_app(tmp_path/'store',manager_factory=Jobs)) as client:
        project=client.post('/api/upload',headers={'X-Harmony-Client':'studio'},files={'file':('x.wav',b'RIFFx','audio/wav')}).json()
        store=client.app.state.store;pid=project['id']
        sha=signature(store.project_dir(pid)/project['audio'])['sha256']
        folder=stems_dir(sha,'bs_roformer_sw')/'drums_parts';folder.mkdir(parents=True)
        atomic_json(folder/'drums_parts.json',dict(tracks=[dict(id='kick',file='kick.mp3',wav='wav/kick.wav')]))
        r=client.get(f'/api/projects/{pid}/stems/drums-parts?model=bs_roformer_sw').json()
        assert r['urls']['kick'].endswith('?model=bs_roformer_sw')
        assert r['wav_urls']['kick'].endswith('drums_parts/wav/kick.wav?model=bs_roformer_sw')


@pytest.mark.parametrize('profile',['stems','vocal_parts'])
def test_jobs_preserve_analysis_and_model_cache(tmp_path,monkeypatch,profile,fake_job_process):
    import threading
    from app.web.jobs import Jobs as RealJobs
    from app.web.storage import Store,read_json
    with TestClient(create_app(tmp_path/'store',manager_factory=Jobs)) as client:
        project=client.post('/api/upload',headers={'X-Harmony-Client':'studio'},files={'file':('x.wav',b'RIFFx','audio/wav')}).json()
        store=client.app.state.store;pid=project['id'];jid=uuid4().hex
        project.update(current_run='frozen-analysis',stems={'bs_roformer_sw':{'sha256':'unchanged'}})
        store.save_project(project)
        atomic_json(store.job_path(jid),dict(id=jid,project_id=pid,profile=profile,status='queued'))
        manager=object.__new__(RealJobs);manager.store=store;manager.lock=threading.RLock();manager.python='python';manager.processes={};manager.cancel_requested=set()
        fake_job_process()
        manager._execute(jid)
        actual=store.project(pid)
        assert actual['current_run']=='frozen-analysis'
        assert actual['stems']=={'bs_roformer_sw':{'sha256':'unchanged'}}
        assert read_json(store.job_path(jid))['status']=='completed'
