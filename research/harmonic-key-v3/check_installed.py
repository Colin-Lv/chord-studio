import json,requests,hashlib,sys
from pathlib import Path
DEST=Path('C:/Users/33573/AppData/Local/Programs/Chord Studio/backend')
sys.path.insert(0,str(DEST))
import app.harmonic_key_v3 as v3
HERE=Path(__file__).resolve().parent
session=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text())
base='http://127.0.0.1:'+str(session['port'])
headers={'x-chord-session':session['token'],'X-Harmony-Client':'studio'}
checks={'loaded_module':str(Path(v3.__file__).resolve()),'assets':{},'projects':[]}
assert Path(v3.__file__).resolve().is_relative_to(DEST)
for url in ['/','/assets/harmonic-key.js','/assets/app.js']:
    response=requests.get(base+url,headers=headers,timeout=30);response.raise_for_status()
    checks['assets'][url]=dict(status=response.status_code,sha256=hashlib.sha256(response.content).hexdigest())
    if url=='/':assert 'harmonic-key.js' in response.text and 'harmonicKeyTimeline' in response.text
for pid in ['c1ebfe4bb5924849b320be11b0aa1f9e','aa5a41797bb343e590abcf4d8b27ae54']:
    response=requests.get(base+'/api/projects/'+pid+'/result',headers=headers,timeout=30);response.raise_for_status()
    result=response.json();timeline=result['harmonic_keys'][result['default_source']]
    source=next(s for s in result['sources'] if s['id']==result['default_source'])
    computed=v3.analyze(source['rows'],result['duration'],result['bars'],result['structure']['sections'])
    path=lambda t:[(s['start'],s['end'],s['key']) for s in t['segments']]
    assert path(computed)==path(timeline)
    checks['projects'].append(dict(project=pid,main=timeline['main_key'],path=path(timeline),changes=timeline['changes']))
    print(pid,timeline['main_key'],path(timeline))
(HERE/'live_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
print('Installed Python runtime, live API, assets and actual song inference verified. No desktop automation.')
