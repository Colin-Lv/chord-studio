"""Install only this change; keep an exact reversible asset backup."""
from pathlib import Path
import hashlib,json,shutil
root=Path(__file__).resolve().parents[3]
here=Path(__file__).resolve().parent
installed=Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
files=['app/web/stem_harmony.py','app/web/fusion_worker.py','app/web/fusion_engine.py',
       'app/web/pipeline_worker.py','app/web/server.py','frontend/pipeline.js','frontend/index.html']
rows=[]
for file in files:
    source=root/file;target=installed/file;backup=here/'installed_before'/file
    if target.exists() and not backup.exists():
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
    shutil.copy2(source,target)
    assert source.read_bytes()==target.read_bytes()
    rows.append(dict(file=file,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
(here/'installed_files.json').write_text(json.dumps(rows,indent=2))
print('Installed',len(rows),'files; worker changes load on the next job; no desktop automation.')
