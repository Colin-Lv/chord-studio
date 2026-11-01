import json,hashlib,sys
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from app.web.editor import validate_document
session=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text())
base='http://127.0.0.1:'+str(session['port']);headers={'x-chord-session':session['token'],'X-Harmony-Client':'studio'}
request=json.loads((HERE/'integrated_live_job.json').read_text());prefix='/api/projects/'+request['project_id']
def get(path):
    r=requests.get(base+path,headers=headers,timeout=40);r.raise_for_status();return r.json()
job=get('/api/jobs/'+request['job_id']);assert job['status']=='completed',job.get('error') or job['stage_label']
assert any(s['id']=='joint_harmony' and s['status']=='completed' for s in job['stages'])
result=get(prefix+'/result');assert result['run_id']==request['job_id']
assert result['default_source'].startswith('retained10_bass_')
assert result['joint_harmony']['automatic'] and result['model_identity']['bass_voice']['version']=='bass-voice-v2'
source=next(s for s in result['sources'] if s['id']==result['default_source'])
doc=get(prefix+'/editor/'+source['id']);validate_document(doc,source,result['duration'])
checks=[]
for time,label in [(20.82,'B:maj/C#'),(23.84,'B:maj/D#'),(26.83,'E:maj'),(35.83,'B:maj/D#')]:
    actual=next(e['label'] for e in doc['events'] if e['start']<=time<e['end']);assert actual==label
    checks.append(dict(time=time,label=actual))
export=requests.get(base+prefix+'/export/'+source['id']+'/original.lab',headers=headers,timeout=30);export.raise_for_status()
assert hashlib.sha256(export.content).hexdigest()==source['sha256']
assert 'B:maj/C#' in export.text
before=json.loads((HERE/'integrated_before_analysis.json').read_text(encoding='utf8'))
old_path=Path('D:/ChordStudio/data/projects')/request['project_id']/'runs'/request['previous_run']/'analysis.json'
assert json.loads(old_path.read_text(encoding='utf8'))==before
report=dict(job=job['id'],run=result['run_id'],default_source=source['id'],source_name=source['name'],
    request=request['request'],automatic=result['joint_harmony']['automatic'],passage=checks,
    source_rows=len(source['rows']),editor_document_valid=True,export_hash_verified=True,
    previous_analysis_unchanged=True,stage_seconds={s['id']:s['seconds'] for s in job['stages']})
(HERE/'integrated_live_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
(HERE/'integrated_current_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False))
