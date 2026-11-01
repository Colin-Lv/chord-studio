import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.web.stem_harmony import recognize
from app.web.stems_worker import stems_dir
from app.web.storage import Store, signature, atomic_json
from evaluate import score

here = Path(__file__).resolve().parent
store = Store(Path('D:/ChordStudio/data/projects'))
for name, pid in [('synth6','21fedd34166b40c6b7cb11573cc8b493'), ('dashisuixiongkou','330d75b7138349e9b9132e53b8a9763e')]:
    project = store.project(pid)
    run = store.run_dir(pid,project['current_run'])
    sha = signature(store.project_dir(pid)/project['audio'])['sha256']
    base = stems_dir(sha,'bs_roformer_sw')
    if name == 'synth6':
        base = here/'synth6_source';(base/'wav').mkdir(parents=True,exist_ok=True)
        original = Path('D:/ChordStudio/research/separation/synth6/model6')
        tracks = ['vocals','drums','bass','other','guitar','piano']
        for track in tracks:
            shutil.copy2(original/f'mix_({track})_BS-Roformer-SW.wav',base/'wav'/f'{track}.wav')
        atomic_json(base/'manifest.json', dict(audio_sha256=sha,model={'name':'bs_roformer_sw'},tracks=[{'id':t} for t in tracks]))
    rows, meta = recognize(run,base,here/name,lambda m:print(name,m,flush=True))
    print(name,meta['segments'],meta['seconds'],flush=True)
    if name == 'synth6':
        truth=[dict(start=i*4*60/105,end=(i+1)*4*60/105,label=['C:maj','A:min','F:maj','G:maj'][i%4]) for i in range(16)]
        result=json.loads((here/name/'result.json').read_text(encoding='utf8'))
        scores={key:score(value,truth) for key,value in [('released_full',result['baseline']),('stem_v2',rows)]}
        atomic_json(here/'synth6_scores.json',scores)
        print(scores,flush=True)
