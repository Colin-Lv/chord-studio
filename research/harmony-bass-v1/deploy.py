"""Back up and install this revision only; no desktop interaction or restart."""
import json,shutil,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
INSTALLED=Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
FILES=['app/web/bass_voice.py','app/web/stem_bass_harmony.py','app/web/full_decode.py',
       'app/web/fusion_worker.py','app/web/models/bass_voice_v2.joblib','app/web/models/bass_voice_v2.json',
       'frontend/index.html','frontend/pipeline.js']
records=[]
for name in FILES:
    source=ROOT/name;target=INSTALLED/name;backup=HERE/'installed_before'/name
    if target.exists() and not backup.exists():
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert source.read_bytes()==target.read_bytes()
    records.append(dict(file=name,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
(HERE/'installed_files.json').write_text(json.dumps(records,indent=2))
print('Installed',len(records),'files; next worker job uses this revision.')
