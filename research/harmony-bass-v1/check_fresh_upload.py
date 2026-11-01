"""Cold ordinary-worker run on self-rendered MIDI in an isolated project store."""
import sys,json,uuid,shutil,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from app.web.storage import Store,atomic_json,signature,read_json
from app.web.editor import read_document,validate_document
DEST=Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
store=Store(HERE/'fresh-upload-store');project=store.create('Synthetic slash-bass cold-start check')
audio=store.project_dir(project['id'])/'audio.wav'
shutil.copy2(HERE/'integrated-challenge/0-fixed_upper.wav',audio)
project.update(audio=audio.name,audio_identity=signature(audio));store.save_project(project)
job_id=uuid.uuid4().hex
atomic_json(store.job_path(job_id),dict(id=job_id,project_id=project['id'],profile='retained10',
    status='running',stages=[],download=False,key_mode='continuous'))
python=read_json(DEST/'desktop_runtime.json')['model_python'];env=os.environ.copy()
env.update(PYTHONUTF8='1',CHORD_DESKTOP_DATA='D:/ChordStudio/data',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
request=dict(job_id=job_id,project_id=project['id'],store=str(store.root),had_prior_analysis=False,had_prior_stem_preference=False)
atomic_json(HERE/'fresh_upload_request.json',request)
with (HERE/'fresh_upload_worker.log').open('w',encoding='utf8') as log:
    subprocess.run([python,'-B','-m','app.web.worker','--store',str(store.root),'--job',job_id],cwd=DEST,env=env,
        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
result=read_json(store.run_dir(project['id'],job_id)/'analysis.json')
assert result['default_source'].startswith('retained10_bass_') and result['joint_harmony']['automatic']
project=store.project(project['id']);project['current_run']=job_id;store.save_project(project)
source=next(s for s in result['sources'] if s['id']==result['default_source'])
doc=read_document(store,project['id'],source['id']);validate_document(doc,source,result['duration'])
report=dict(**request,default_source=source['id'],rows=source['rows'],editor_valid=True,
    stages=read_json(store.job_path(job_id))['stages'],model_identity=result['model_identity'])
atomic_json(HERE/'fresh_upload_check.json',report)
print(json.dumps(dict(default_source=source['id'],editor_valid=True,slash_labels=[r['label'] for r in source['rows'] if '/' in r['label']]),ensure_ascii=False),flush=True)
