"""Verify real installed worker queue and exports without opening the UI."""
import json,hashlib,time
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent
session=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text())
base='http://127.0.0.1:'+str(session['port'])
headers={'x-chord-session':session['token'],'X-Harmony-Client':'studio'}
def get(path):
    response=requests.get(base+path,headers=headers,timeout=40);response.raise_for_status();return response.json()
pid='c1ebfe4bb5924849b320be11b0aa1f9e';prefix='/api/projects/'+pid
project=get(prefix);assert not project.get('active_job'),'A user job is active'
before=get(prefix+'/result');(HERE/'installed_before_result.json').write_text(json.dumps(before,ensure_ascii=False,indent=2),encoding='utf8')
response=requests.post(base+prefix+'/fusion',headers=headers,json={'model':'bs_roformer_sw'},timeout=40)
response.raise_for_status();job=response.json();assert job['profile']=='fusion'
print('Submitted installed fusion job',job['id'],flush=True)
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    job=get('/api/jobs/'+job['id'])
    if job['status'] in ('completed','failed','interrupted'):break
    time.sleep(2)
assert job['status']=='completed',job.get('error') or job['status']
after=get(prefix+'/result');assert before['run_id']==after['run_id']
assert before['key_analysis']==after['key_analysis']
assert before['default_source']==after['default_source']
for old in before['sources']:assert next(s for s in after['sources'] if s['id']==old['id'])==old
source=next(s for s in after['sources'] if s['id']==after['stem_bass_harmony']['source_id'])
assert source['rows'][0]['start']==0 and abs(source['rows'][-1]['end']-after['duration'])<.002
for left,right in zip(source['rows'],source['rows'][1:]):assert abs(left['end']-right['start'])<1e-7
for t,label in [(23.9,'B:maj/D#'),(26.9,'E:maj'),(35.9,'B:maj/D#')]:
    assert next(r['label'] for r in source['rows'] if r['start']<=t<r['end'])==label
export=requests.get(base+prefix+'/export/'+source['id']+'/original.lab',headers=headers,timeout=30);export.raise_for_status()
assert hashlib.sha256(export.content).hexdigest()==source['sha256']
assert get(prefix+'/corrections/'+source['id'])['revision']==0
report=dict(job_id=job['id'],source_id=source['id'],status=job['status'],segments=len(source['rows']),
            original_sources_unchanged=True,key_unchanged=True,default_unchanged=True,export_verified=True,
            passage=[r for r in source['rows'] if r['start']<42 and r['end']>17])
(HERE/'installed_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('Verified installed job, original sources, key, bass labels and LAB export.',flush=True)
