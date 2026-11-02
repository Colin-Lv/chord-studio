import threading
from uuid import uuid4
from app.web import jobs
from app.web.storage import Store,atomic_json,read_json


def test_initial_status_write_failure_finishes_job_and_keeps_results(tmp_path,monkeypatch):
    store=Store(tmp_path);p=store.create('saved song',stems={'strings_v2':{'generation':'saved'}})
    jid=uuid4().hex;path=store.job_path(jid)
    atomic_json(path,dict(id=jid,project_id=p['id'],profile='stems',status='queued'))
    p.update(active_job=jid);store.save_project(p)
    manager=object.__new__(jobs.Jobs);manager.store=store;manager.python='unused';manager.lock=threading.RLock();manager.cancel_requested=set();manager.processes={}
    calls=[]
    def write(target,payload):
        calls.append(payload['status'])
        if len(calls)==1:raise PermissionError('simulated startup write collision')
        atomic_json(target,payload)
    monkeypatch.setattr(jobs,'atomic_json',write)
    manager._execute(jid)
    result=read_json(path);assert result['status']=='failed' and 'PermissionError' in result['error']
    saved=store.project(p['id']);assert saved['active_job'] is None and saved['status']=='completed'
    assert saved['stems']==p['stems']
