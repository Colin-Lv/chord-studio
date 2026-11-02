from uuid import uuid4
from fastapi.testclient import TestClient
from app.web.server import create_app
from app.web.storage import Store,atomic_json,read_json
from app.web import stems_worker,guitar_parts_worker

class Queue:
    def __init__(self,store):self.store=store
    def shutdown(self):pass
    def submit(self,pid,profile,**kw):return dict(id=uuid4().hex,project_id=pid,profile=profile,**kw)


def test_default_separation_and_pipeline_include_strings_and_guitars(tmp_path):
    with TestClient(create_app(tmp_path,manager_factory=Queue)) as c:
        p=c.app.state.store.create('song',audio='song.wav');prefix=f'/api/projects/{p["id"]}'
        headers={'X-Harmony-Client':'studio'}
        row=c.post(prefix+'/separate',json={},headers=headers).json()
        assert row['model']=='strings_v2' and row['split_guitar'] and row['guitar_mode']=='timbre'
        row=c.post(prefix+'/pipeline',json={},headers=headers).json()
        assert row['pipeline']['model']=='strings_v2' and row['pipeline']['guitar_parts']
        row=c.post(prefix+'/separate',json={'model':'htdemucs'},headers=headers).json()
        assert row['model']=='htdemucs' and not row['split_guitar']
        row=c.post(prefix+'/separate',json={'guitar_parts':False},headers=headers).json()
        assert not row['split_guitar']


def test_reuse_parent_then_run_guitar_cache_without_forcing_it(tmp_path,monkeypatch):
    store=Store(tmp_path);p=store.create('song');jid=uuid4().hex;calls=[]
    atomic_json(store.job_path(jid),dict(id=jid,project_id=p['id'],model='strings_v2',
                split_guitar=True,force=True,status='running',guitar_mode='timbre'))
    parent={'generation':'approved','cached':True}
    def separate(*a,**kw):calls.append('parent');return parent
    def guitar(store,jid):
        row=read_json(store.job_path(jid));assert row['model']=='strings_v2' and not row['force']
        calls.append('guitar')
    monkeypatch.setattr(stems_worker,'execute',separate)
    monkeypatch.setattr(guitar_parts_worker,'execute',guitar)
    assert stems_worker.execute_with_parts(store,jid)==parent
    assert calls==['parent','guitar'] and read_json(store.job_path(jid))['force']
