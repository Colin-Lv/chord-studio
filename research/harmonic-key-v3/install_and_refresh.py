"""Install the reviewed implementation and add timelines to selected current runs.

Backup files first. Refuse an analysis update if its bytes or current run changed.
Never touch project status, queued jobs, audio, labels or correction records.
"""
import sys,json,hashlib,shutil,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from app.web.harmonic_key import refresh
from app.web.storage import atomic_json

DEST=Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
HERE=Path(__file__).resolve().parent
FILES=['app/harmonic_key_v3.py','app/web/harmonic_key.py','app/web/models/harmonic_key_v3.json','app/chord_semantics_v3.py',
       'app/web/retained_worker.py','app/web/worker.py','app/web/stem_harmony.py','app/web/stem_bass_harmony.py',
       'frontend/harmonic-key.js','frontend/app.js','frontend/index.html','frontend/style.css']
PROJECTS=['c1ebfe4bb5924849b320be11b0aa1f9e','aa5a41797bb343e590abcf4d8b27ae54']


def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=Path('D:/ChordStudio/data/backups')/('harmonic-key-v3_'+stamp)
    backup.mkdir(parents=True,exist_ok=False);report={'backup':str(backup),'files':[],'projects':[]}
    for relative in FILES:
        target=DEST/relative;copy=backup/'installed'/relative
        if target.exists():copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,copy)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
        assert target.read_bytes()==(ROOT/relative).read_bytes()
        report['files'].append(dict(path=relative,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    for pid in PROJECTS:
        folder=Path('D:/ChordStudio/data/projects')/pid;project=json.loads((folder/'project.json').read_text(encoding='utf8'))
        run=folder/'runs'/project['current_run'];path=run/'analysis.json';before=path.read_bytes()
        result=json.loads(before);source_rows=json.dumps(result['sources'],sort_keys=True)
        human={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*.json')
               if p!=path}
        refresh(result)
        assert json.dumps(result['sources'],sort_keys=True)==source_rows
        if path.read_bytes()!=before or json.loads((folder/'project.json').read_text())['current_run']!=project['current_run']:
            raise RuntimeError('分析在计算期间发生变化，未覆盖：'+pid)
        copy=backup/'projects'/pid/'analysis.json';copy.parent.mkdir(parents=True);copy.write_bytes(before)
        atomic_json(path,result)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in human.items())
        assert json.loads((folder/'project.json').read_text())==project
        (HERE/(pid+'_analysis.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        timelines={s:t for s,t in result['harmonic_keys'].items()}
        summary=dict(project=pid,run=project['current_run'],default_source=result['default_source'],
            main_key=result['key_analysis'].get('key'),other_json_records_preserved=len(human),
            sources={s:dict(status=t['status'],segments=[{k:v for k,v in x.items() if k in ('start','end','key','margin','status')} for x in t.get('segments',[])]) for s,t in timelines.items()})
        report['projects'].append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    (HERE/'installed.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('Backup:',backup)

if __name__=='__main__':main()
