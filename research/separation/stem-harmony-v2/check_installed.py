"""Run the installed job queue, verify immutable baseline/key and usable exports."""
import hashlib,json,time
from pathlib import Path
import requests
here=Path(__file__).resolve().parent
s=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text())
base='http://127.0.0.1:'+str(s['port'])
headers={'x-chord-session':s['token'],'X-Harmony-Client':'studio'}
def get(path):
    r=requests.get(base+path,headers=headers,timeout=40);r.raise_for_status();return r.json()
checks=[]
for pid in ['aa5a41797bb343e590abcf4d8b27ae54','330d75b7138349e9b9132e53b8a9763e']:
    prefix='/api/projects/'+pid
    p=get(prefix);assert not p.get('active_job'),'Do not interfere with a running job'
    before=get(prefix+'/result')
    # Snapshot before publication, including source rows and key metadata.
    (here/(pid+'_before.json')).write_text(json.dumps(before,ensure_ascii=False,indent=2),encoding='utf8')
    r=requests.post(base+prefix+'/fusion',headers=headers,json={'model':'bs_roformer_sw'},timeout=40)
    r.raise_for_status();job=r.json();assert job['profile']=='fusion'
    print(pid,'job submitted',flush=True)
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        job=get('/api/jobs/'+job['id'])
        if job['status'] in ['completed','failed','interrupted']: break
        time.sleep(2)
    assert job['status']=='completed',job.get('error') or job['status']
    after=get(prefix+'/result')
    assert after['run_id']==before['run_id']
    assert after['key_analysis']==before['key_analysis']
    assert after['default_source']==before['default_source']
    for old in before['sources']:
        assert next(x for x in after['sources'] if x['id']==old['id'])==old
    meta=after['stem_harmony_v2'];source=next(x for x in after['sources'] if x['id']==meta['source_id'])
    assert source['rows'][0]['start']==0
    assert abs(source['rows'][-1]['end']-after['duration'])<.002
    assert get(prefix+'/corrections/'+source['id'])['revision']==0
    export=requests.get(base+prefix+'/export/'+source['id']+'/original.lab',headers=headers,timeout=30)
    export.raise_for_status();assert hashlib.sha256(export.content).hexdigest()==source['sha256']
    check=dict(project=pid,run_id=after['run_id'],job_id=job['id'],source_id=source['id'],
               segments=len(source['rows']),seconds=meta['seconds'],key_unchanged=True,
               original_sources_unchanged=True,default_unchanged=True,export_checksum=True)
    checks.append(check);print(check,flush=True)
(here/'installed_checks.json').write_text(json.dumps(checks,indent=2))
