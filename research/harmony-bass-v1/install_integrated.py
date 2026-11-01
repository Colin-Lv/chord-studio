import json,shutil,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
DEST=Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
FILES=['app/web/joint_harmony.py','app/web/retained_worker.py','app/web/pipeline_worker.py',
       'app/web/stems_worker.py','app/web/stem_bass_harmony.py',
       'frontend/index.html','frontend/app.js','frontend/pipeline.js']
backup=Path('D:/ChordStudio/data/backups')/('joint-harmony_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True,exist_ok=False);report={'backup':str(backup),'files':[]}
for name in FILES:
    target=DEST/name;copy=backup/name
    if target.exists():copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,copy)
    shutil.copy2(ROOT/name,target)
    assert target.read_bytes()==(ROOT/name).read_bytes()
    report['files'].append(dict(path=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
report['bass_weights']=dict(path='app/web/models/bass_voice_v2.joblib',sha256=hashlib.sha256((DEST/'app/web/models/bass_voice_v2.joblib').read_bytes()).hexdigest(),candidate_v3_promoted=False)
(HERE/'integrated_installed.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('Installed',len(FILES),'files. Backup:',backup)
