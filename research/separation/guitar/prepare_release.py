"""Local installation only; preserve exact pre-release files and evidence."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
MODELS=Path('D:/ChordStudio/data/models/guitar')
installed=Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
files=['app/web/guitar_engine.py','app/web/guitar_parts_runner.py','app/web/guitar_parts_worker.py',
       'app/web/server.py','app/web/jobs.py','app/web/worker.py','frontend/stem-audio.js','frontend/guitar-parts.js','frontend/stems.js','frontend/stems.html','frontend/stems.css']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
training=json.loads((MODELS/'role_adapt_v3/training.json').read_text())
assert training['best_step']==100 and training['history'][-1]['step']==250
chosen=MODELS/'role_adapt_v3/best.ckpt';destination=MODELS/'guitar_role_v1.ckpt'
if destination.exists():assert sha(destination)==sha(chosen),'Do not overwrite a different released checkpoint'
else:shutil.copy2(chosen,destination)
recipe=dict(version='guitar-role-v1-local',architecture='MelBandRoformer listra92 config, strict weights',
    checkpoint_sha256=sha(chosen),base_sha256=sha(MODELS/'mbr_lead_rhythm_guitar_listra92.ckpt'),
    selected_step=100,finished_validation_step=250,selection='player03 validation; early stopping after 3 worse checks',
    training=training,semantics=['estimated melody guitar','estimated accompaniment guitar'],
    limitation='Not performer identity recovery. Fails on several similar classical guitar duets.')
(MODELS/'guitar_role_v1.json').write_text(json.dumps(recipe,indent=2))
rows=[]
for relative in files:
    source=ROOT/relative;target=installed/relative;backup=HERE/'installed_before'/relative
    if target.exists() and not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert sha(source)==sha(target)
    rows.append(dict(file=relative,sha256=sha(source),backup=str(backup) if backup.exists() else None))
(HERE/'installed_files.json').write_text(json.dumps(rows,indent=2))
print('Installed',len(rows),'files; model checksum',sha(destination))
