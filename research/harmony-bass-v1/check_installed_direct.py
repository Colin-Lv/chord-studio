"""Use installed implementation on the current completed run, without touching the queued job."""
import sys,json,hashlib
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
from app.web.stem_bass_harmony import recognize,publish_result,VERSION
from app.web.storage import read_json,atomic_json
session=read_json('D:/ChordStudio/data/active-session.json');base='http://127.0.0.1:'+str(session['port'])
headers={'x-chord-session':session['token'],'X-Harmony-Client':'studio'}
pid='c1ebfe4bb5924849b320be11b0aa1f9e';prefix='/api/projects/'+pid
def get(path):
    response=requests.get(base+path,headers=headers,timeout=40);response.raise_for_status();return response.json()
project=get(prefix);before=get(prefix+'/result')
atomic_json(HERE/'installed_before_result.json',before)
folder=Path('D:/ChordStudio/data/projects')/pid;run=folder/'runs'/project['current_run']
stem=Path(project['stems']['bs_roformer_sw']['manifest']).parent
rows,meta=recognize(run,stem,run/VERSION/'bs_roformer_sw',lambda s:print(s,flush=True))
assert get(prefix)['current_run']==project['current_run'],'Current run changed during analysis; result retained for review only'
source_id=publish_result(run,rows,meta,False)
after=get(prefix+'/result')
assert before['key_analysis']==after['key_analysis'] and before['default_source']==after['default_source']
for old in before['sources']:assert next(s for s in after['sources'] if s['id']==old['id'])==old
source=next(s for s in after['sources'] if s['id']==source_id)
for t,label in [(20.9,'B:maj/C#'),(23.9,'B:maj/D#'),(26.9,'E:maj'),(35.9,'B:maj/D#')]:
    assert next(r['label'] for r in source['rows'] if r['start']<=t<r['end'])==label
export=requests.get(base+prefix+'/export/'+source_id+'/original.lab',headers=headers,timeout=30);export.raise_for_status()
assert hashlib.sha256(export.content).hexdigest()==source['sha256']
assert get(prefix+'/corrections/'+source_id)['revision']==0
report=dict(source_id=source_id,run_id=project['current_run'],installed_implementation=True,api_readback=True,
            original_sources_unchanged=True,key_unchanged=True,default_unchanged=True,export_verified=True,
            queue_not_modified=True,pending_job=project.get('active_job'),
            passage=[r for r in rows if r['start']<42 and r['end']>17])
atomic_json(HERE/'installed_check.json',report)
print('Installed implementation and live API export verified:',source_id,flush=True)
