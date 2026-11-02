from pathlib import Path
from uuid import uuid4
import json
import numpy as np
import pretty_midi as pm
import pytest
from fastapi.testclient import TestClient
from app.web.piano_midi import BeatMap, export, swing_regions, explicit_grid
from app.web.server import create_app
from app.web.storage import atomic_json, read_json, signature


def source(path, notes, pedal=True):
    m=pm.PrettyMIDI(resolution=6720);i=pm.Instrument(0);m.instruments.append(i)
    for pitch,start,end,velocity in notes:i.notes.append(pm.Note(velocity,pitch,start,end))
    if pedal:i.control_changes=[pm.ControlChange(64,127,.3),pm.ControlChange(64,0,3.7)]
    m.write(str(path))
    return m


def read(path):
    return sorted(pm.PrettyMIDI(str(path)).instruments[0].notes,key=lambda n:(n.start,n.pitch))


def test_tempo_map_roundtrip_preserves_all_notes_releases_and_pedals(tmp_path):
    beats=[dict(time=t,beat=i%4+1) for i,t in enumerate([.8,1.55,2.3,3.05,3.8,4.3,4.8,5.3,5.8])]
    raw=tmp_path/'raw.mid'
    source(raw,[(36,.1,.7,42),(60,.81,1.3,90),(64,.82,2.7,65),(60,1.56,3.1,70),(72,4.802,5.31,77)])
    report=export(raw,tmp_path,beats,6.5)
    before=read(raw);after=read(tmp_path/'performance.mid');grid=read(tmp_path/'grid.mid')
    assert len(before)==len(after)==len(grid)==5
    for a,b in zip(before,after):
        assert (a.pitch,a.velocity)==(b.pitch,b.velocity)
        assert abs(a.start-b.start)<.0003 and abs(a.end-b.end)<.0003
    cc=pm.PrettyMIDI(str(tmp_path/'performance.mid')).instruments[0].control_changes
    assert [(c.number,c.value) for c in cc]==[(64,127),(64,0)]
    assert cc[1].time==pytest.approx(3.7,abs=.0003)
    assert report['notes'][0]['bar']==0
    assert report['maximum_onset_shift_ms']<=40.001
    # A notated release is not inferred by rounding key release to a beat.
    assert grid[1].end-grid[1].start==pytest.approx(before[1].end-before[1].start,abs=.0003)


@pytest.mark.parametrize('phase',[.6,.62,2/3,.71])
def test_swing_preserves_variable_ratio_and_triplet_offsets(tmp_path,phase):
    raw=tmp_path/'raw.mid';notes=[]
    for i in range(8):
        for offset in (0,phase):notes.append((60,i+offset,i+offset+.18,80))
    source(raw,notes,False)
    beats=[dict(time=float(i),beat=i%4+1) for i in range(9)]
    r=export(raw,tmp_path,beats,9.)
    assert len(r['swing'])==2
    assert all(x['offbeat_phase']==pytest.approx(phase,abs=.001) for x in r['swing'])
    inferred=read(tmp_path/'grid.mid')
    assert inferred[1].start==pytest.approx(phase,abs=.001)
    assert abs(inferred[1].start-.5)>.09


@pytest.mark.parametrize('division',[2,3,4,5,7])
def test_regular_tuplets_are_not_misclassified_as_swing(division):
    grid=BeatMap([dict(time=float(i),beat=i%4+1) for i in range(9)],9.)
    notes=[dict(start=i+j/division) for i in range(8) for j in range(division)]
    assert swing_regions(notes,grid)==[]


def test_grace_notes_repeats_and_unmetered_intro_are_not_deleted(tmp_path):
    raw=tmp_path/'raw.mid'
    source(raw,[(60,.1,.2,50),(60,1.98,1.99,80),(60,2.01,2.08,85),(61,2.131,2.157,30)],False)
    beats=[dict(time=float(i),beat=i%4+1) for i in range(1,10)]
    r=export(raw,tmp_path,beats,10.)
    notes=read(tmp_path/'grid.mid')
    assert len(notes)==4 and all(n.end>n.start for n in notes)
    assert notes[1].start<notes[2].start
    assert r['notes'][0]['bar']==0


def test_beat_failure_still_exports_real_timing_without_inventing_bars(tmp_path):
    raw=tmp_path/'raw.mid';source(raw,[(60,.4,1.2,70)],False)
    report=export(raw,tmp_path,[],2.)
    assert report['bars']==[] and report['snapped_onsets']==0
    assert read(tmp_path/'grid.mid')[0].start==pytest.approx(.4,abs=.0002)


def test_user_grid_preserves_septuplet_times_and_known_bar_structure(tmp_path):
    raw=tmp_path/'raw.mid';events=[]
    for i in range(32):
        for j in range(7):
            start=.6+(i+j/7)*.75
            events.append((60+j,start,start+.06,70))
    source(raw,events,False)
    r=export(raw,tmp_path,explicit_grid(80,.6,4,25.),25.)
    assert r['count']==224
    assert max(n['bar'] for n in r['notes'])==8
    assert r['quarter_note_bpm']==80
    assert r['swing']==[]
    for before,after in zip(read(raw),read(tmp_path/'grid.mid')):
        assert abs(before.start-after.start)<.0003


class FakeJobs:
    def __init__(self,store):self.store=store
    def shutdown(self):pass
    def submit(self,project_id,profile,**kwargs):return dict(id=uuid4().hex,project_id=project_id,profile=profile,**kwargs)


def test_api_sources_busy_job_download_and_saved_generation(tmp_path,monkeypatch):
    monkeypatch.setattr('app.web.piano_worker.PIANO_PYTHON',Path(__file__))
    with TestClient(create_app(tmp_path/'store',manager_factory=FakeJobs)) as client:
        store=client.app.state.store;headers={'X-Harmony-Client':'studio'}
        p=client.post('/api/upload',headers=headers,files={'file':('piano.wav',b'RIFFtest','audio/wav')}).json()
        base='/api/projects/'+p['id']+'/piano'
        assert client.get(base).status_code==404
        for kind in ['mix','piano']:
            r=client.post(base,headers=headers,json={'input':kind});assert r.status_code==202
            assert r.json()['pipeline']=={'input':kind} and r.json()['profile']=='piano'
        assert client.post(base,headers=headers,json={'input':'arbitrary-file'}).status_code==422
        assert client.post(base,headers=headers,json={'input':'piano','bpm':80}).status_code==422
        manual=dict(input='piano',bpm=80,first_downbeat=.6,beats_per_bar=4)
        assert client.post(base,headers=headers,json=manual).json()['pipeline']==manual
        gen=uuid4().hex;folder=store.project_dir(p['id'])/'piano'/gen;folder.mkdir(parents=True)
        source(folder/'performance.mid',[(60,.1,.5,80)],False)
        manifest=dict(generation=gen,audio_sha256=signature(store.project_dir(p['id'])/p['audio'])['sha256'],files={'performance.mid':{}})
        atomic_json(folder/'manifest.json',manifest)
        p['piano_transcription']={'generation':gen};store.save_project(p)
        result=client.get(base).json();url=result['urls']['performance.mid']
        assert client.get(url).content.startswith(b'MThd')
        assert client.get(base+'/'+gen+'/inference.log').status_code==404
        assert client.get(base+'/bad/performance.mid').status_code==400
        jid=uuid4().hex;p['active_job']=jid;store.save_project(p)
        atomic_json(store.job_path(jid),dict(profile='stems'))
        assert client.post(base,headers=headers,json={'input':'mix'}).status_code==409
        assert client.get(url).content.startswith(b'MThd')
        (store.project_dir(p['id'])/p['audio']).write_bytes(b'replaced')
        assert client.get(base).status_code==404
        assert client.get(url).status_code==200 # immutable prior generation


def test_piano_job_completion_does_not_replace_harmony_run(tmp_path,monkeypatch,fake_job_process):
    from app.web.jobs import Jobs
    from app.web.storage import Store
    store=Store(tmp_path);run=uuid4().hex;p=store.create('song');p['current_run']=run;store.save_project(p)
    jid=uuid4().hex
    atomic_json(store.job_path(jid),dict(id=jid,project_id=p['id'],profile='piano',status='queued'))
    manager=Jobs(store)
    atomic_json(store.job_path(jid),dict(id=jid,project_id=p['id'],profile='piano',status='queued'))
    def fake_run(*args,**kwargs):
        project=store.project(p['id']);project['piano_transcription']={'generation':uuid4().hex};store.save_project(project)
    fake_job_process(fake_run)
    try:manager._execute(jid)
    finally:manager.shutdown()
    assert store.project(p['id'])['current_run']==run
    assert read_json(store.job_path(jid))['status']=='completed'


def test_failed_piano_job_preserves_existing_analysis_and_result(tmp_path,monkeypatch,fake_job_process):
    from app.web.jobs import Jobs
    from app.web.storage import Store
    store=Store(tmp_path);p=store.create('song');run=uuid4().hex
    p.update(current_run=run,piano_transcription={'generation':'previous'});store.save_project(p)
    jid=uuid4().hex
    atomic_json(store.job_path(jid),dict(id=jid,project_id=p['id'],profile='piano',status='ready'))
    manager=Jobs(store)
    atomic_json(store.job_path(jid),dict(id=jid,project_id=p['id'],profile='piano',status='queued'))
    def fail(*args,**kwargs):raise RuntimeError('inference error')
    fake_job_process(fail)
    try:manager._execute(jid)
    finally:manager.shutdown()
    actual=store.project(p['id'])
    assert actual['current_run']==run and actual['piano_transcription']=={'generation':'previous'}
    assert actual['status']=='completed' and actual['active_job'] is None
    assert read_json(store.job_path(jid))['status']=='failed'
