from pathlib import Path
import copy
from uuid import uuid4
import numpy as np
import pretty_midi as pm
import pytest
from fastapi.testclient import TestClient
from app.web.reconstruction_midi import read_track,write_midi,verify_roundtrip
from app.web.storage import atomic_json,signature
from app.web.server import create_app


def test_instrument_absence_is_user_review_scoped_to_audio_not_model_output():
    from app.web.reconstruction_build import excluded_families
    c=dict(audio_sha256='original',instrument_presence_review=dict(audio_sha256='original',instruments={
        'bass':dict(present=False,origin='user'),'guitar':dict(present=False,origin='model')}))
    assert excluded_families(c)=={'bass'}
    c['audio_sha256']='replacement'
    assert excluded_families(c)==set()


def track(ident='piano',drums=False):
    return dict(id=ident,label=ident,program=0,is_drum=drums,notes=[dict(pitch=36 if drums else 60,velocity=80,start=.17,end=.4),dict(pitch=38 if drums else 64,velocity=49,start=2.67,end=3.8)],controls=[] if drums else [dict(number=64,value=127,time=.2),dict(number=64,value=0,time=4.)],bends=[] if drums else [dict(pitch=1000,time=2.7),dict(pitch=0,time=3.9)])


def test_multitrack_tempo_pedal_bends_drum_channel_and_swing_roundtrip(tmp_path):
    beats=[dict(time=t,beat=i%4+1) for i,t in enumerate([.8,1.55,2.3,3.05,3.8,4.3,4.8,5.3,5.8])]
    tracks=[track(),track('drums',True)];out=tmp_path/'all.mid'
    write_midi(tracks,out,beats,6.3);r=verify_roundtrip(tracks,out)
    assert r['notes']==4 and r['maximum_time_error_ms']<.2
    after=pm.PrettyMIDI(str(out));assert after.instruments[1].is_drum
    assert [x.pitch for x in after.instruments[0].pitch_bends]==[1000,0]
    assert after.instruments[0].control_changes[1].time==pytest.approx(4.,abs=.0002)
    assert after.instruments[0].notes[1].start==pytest.approx(2.67,abs=.0002)
    import mido
    assert mido.MidiFile(out).length==pytest.approx(6.3,abs=.0002)


def test_audio_bounds_are_explicit_and_do_not_remove_grace_notes(tmp_path):
    m=pm.PrettyMIDI();i=pm.Instrument(25);m.instruments.append(i)
    i.notes=[pm.Note(60,60,.2,.21),pm.Note(80,62,1.,2.5),pm.Note(70,64,3.,4.)]
    source=tmp_path/'raw.mid';m.write(str(source))
    tracks=read_track(source,'guitar','guitar',25,2.)
    assert len(tracks[0]['notes'])==2
    assert tracks[0]['bounds_check']==dict(discarded_outside_audio=1,clipped_to_audio=1)
    assert tracks[0]['notes'][0]['end']-tracks[0]['notes'][0]['start']<.02


class Jobs:
    def __init__(self,store):self.store=store
    def submit(self,project_id,profile,**kwargs):return dict(id='a'*32,project_id=project_id,profile=profile,**kwargs)
    def shutdown(self):pass


def test_api_exports_only_manifest_files_and_preserves_harmony(tmp_path):
    app=create_app(tmp_path,manager_factory=Jobs);store=app.state.store
    pid=uuid4().hex;gen=uuid4().hex;folder=store.project_dir(pid);folder.mkdir()
    audio=folder/'song.wav';audio.write_bytes(b'identity')
    project=dict(id=pid,title='song',audio='song.wav',current_run='b'*32,active_job=None,song_reconstruction=dict(generation=gen))
    atomic_json(folder/'project.json',project);out=folder/'reconstruction'/gen;out.mkdir(parents=True)
    (out/'performance.mid').write_bytes(b'test-midi');(out/'config.json').write_text('private config')
    atomic_json(out/'manifest.json',dict(generation=gen,audio_sha256=signature(audio)['sha256'],files={'performance.mid':signature(out/'performance.mid')}))
    with TestClient(app) as client:
        prefix=f'/api/projects/{pid}/reconstruction'
        assert client.get(prefix).status_code==200
        assert client.get(prefix+'/'+gen+'/performance.mid').content==b'test-midi'
        assert client.get(prefix+'/'+gen+'/config.json').status_code==404
        assert client.post(prefix,json={}).status_code==403
        r=client.post(prefix,json={},headers={'X-Harmony-Client':'studio'})
        assert r.status_code==202 and r.json()['profile']=='reconstruction'
        assert store.project(pid)['current_run']=='b'*32
        audio.write_bytes(b'new audio')
        assert client.get(prefix).status_code==404


@pytest.mark.parametrize('fail',[False,True])
def test_reconstruction_job_never_replaces_current_harmony(tmp_path,monkeypatch,fail,fake_job_process):
    from app.web.jobs import Jobs as RealJobs
    from app.web.storage import Store,read_json
    from app.web import jobs as module
    store=Store(tmp_path);pid=uuid4().hex;jid=uuid4().hex
    folder=store.project_dir(pid);folder.mkdir()
    atomic_json(folder/'project.json',dict(id=pid,current_run='c'*32,status='queued',active_job=jid))
    atomic_json(store.job_path(jid),dict(id=jid,project_id=pid,profile='reconstruction',status='queued'))
    def run(*args,**kwargs):
        if fail:raise RuntimeError('controlled failure')
    fake_job_process(run)
    manager=RealJobs(store)
    # Constructor marks vanished jobs interrupted; set a newly submitted job afterward.
    p=store.project(pid);p['active_job']=jid;store.save_project(p)
    atomic_json(store.job_path(jid),dict(id=jid,project_id=pid,profile='reconstruction',status='queued'))
    try:manager._execute(jid)
    finally:manager.shutdown()
    p=store.project(pid);assert p['current_run']=='c'*32 and p['active_job'] is None
    assert p['status']=='completed'
    assert read_json(store.job_path(jid))['status']==('failed' if fail else 'completed')
