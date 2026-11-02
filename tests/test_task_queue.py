"""Real process cancellation in a temporary store; no music/GPU or live service."""
import json
import os
from pathlib import Path
import sys
import time

import psutil
import pytest
from fastapi.testclient import TestClient
from app.web import jobs
from app.web.job_process import JobProcess
from app.web.server import create_app
from app.web.storage import Store,atomic_json,read_json


def until(fn,timeout=12):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=fn()
        if value:return value
        time.sleep(.03)
    raise AssertionError('Condition did not complete')


@pytest.fixture
def setup(tmp_path,monkeypatch):
    events=tmp_path/'events';events.mkdir()
    def launch(command,**kw):
        jid=command[-1];store=Store(command[command.index('--store')+1]);row=read_json(store.job_path(jid))
        slow=store.project(row['project_id'])['title']=='slow'
        script="from app.web.job_process import wait_for_release; wait_for_release(); import subprocess,sys,time,json; from pathlib import Path; "
        if slow:
            child="import subprocess,sys,time,json; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); Path(sys.argv[1]).write_text(json.dumps([p.pid])); time.sleep(60)"
            script+=f"p=subprocess.Popen([sys.executable,'-c',{child!r},{str(events/(jid+'.child'))!r}]); "
        script+=f"Path({str(events/(jid+'.json'))!r}).write_text(json.dumps([__import__('os').getpid()"+(',p.pid' if slow else '')+"])); "
        if slow:script+='time.sleep(60)'
        return JobProcess([sys.executable,'-c',script],**kw)
    monkeypatch.setattr(jobs,'JobProcess',launch)
    yield tmp_path,events


def test_running_tree_and_queued_cancel_then_next_job_runs(setup):
    folder,events=setup;store=Store(folder/'store');manager=jobs.Jobs(store,sys.executable)
    first=store.create('slow',stems={'old':{'saved':True}})
    second=store.create('skip');third=store.create('next')
    try:
        a=manager.submit(first['id'],'stems');ap=events/(a['id']+'.json');cp=events/(a['id']+'.child')
        until(lambda:ap.exists() and cp.exists())
        pids=json.loads(ap.read_text())+json.loads(cp.read_text())
        b=manager.submit(second['id'],'stems');c=manager.submit(third['id'],'stems')
        listed=manager.list()['active'];assert [r['id'] for r in listed]==[a['id'],b['id'],c['id']]
        assert [r['queue_position'] for r in listed]==[None,1,2]
        assert manager.cancel(b['id'])['status']=='cancelled'
        assert manager.cancel(b['id'])['status']=='cancelled'
        assert store.project(second['id'])['active_job'] is None
        assert manager.cancel(a['id'])['status']=='cancelling'
        until(lambda:manager.get(c['id'])['status']=='completed')
        assert manager.get(a['id'])['status']=='cancelled'
        assert not (events/(b['id']+'.json')).exists()
        until(lambda:all(not psutil.pid_exists(pid) for pid in pids))
        saved=store.project(first['id']);assert saved['stems']==first['stems']
        assert saved['status']=='completed' and saved['active_job'] is None
        assert manager.cancel(c['id'])['status']=='completed'
        assert manager.list()['active']==[]
    finally:
        for row in manager.list()['active']:manager.cancel(row['id'])
        manager.shutdown()


def test_queue_http_auth_and_unknown_ids(setup):
    folder,events=setup
    with TestClient(create_app(folder/'store')) as client:
        store=client.app.state.store;p=store.create('slow')
        job=client.app.state.jobs.submit(p['id'],'stems')
        until(lambda:(events/(job['id']+'.json')).exists())
        try:
            listed=client.get('/api/jobs').json()['active'];assert listed[0]['project_title']=='slow'
            path='/api/jobs/'+job['id']+'/cancel'
            assert client.post(path).status_code==403
            response=client.post(path,headers={'X-Harmony-Client':'studio'})
            assert response.status_code==200 and response.json()['status']=='cancelling'
            until(lambda:client.get('/api/jobs/'+job['id']).json()['status']=='cancelled')
            assert client.get('/api/jobs').json()['active']==[]
            assert client.post('/api/jobs/'+'a'*32+'/cancel',headers={'X-Harmony-Client':'studio'}).status_code==404
            assert client.post('/api/jobs/invalid/cancel',headers={'X-Harmony-Client':'studio'}).status_code==400
        finally:client.app.state.jobs.cancel(job['id'])


def test_old_job_without_dates_does_not_break_queue(tmp_path):
    store=Store(tmp_path);p=store.create('legacy');jid='c'*32
    atomic_json(store.job_path(jid),dict(id=jid,project_id=p['id'],profile='stems',status='completed'))
    manager=jobs.Jobs(store)
    try:
        listed=manager.list();assert listed['active']==[]
        assert listed['recent'][0]['id']==jid and listed['recent'][0]['project_title']=='legacy'
    finally:manager.shutdown()
