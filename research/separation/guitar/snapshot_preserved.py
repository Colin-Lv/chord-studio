import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DATA=Path('D:/ChordStudio/data')
pid='aa5a41797bb343e590abcf4d8b27ae54';project=json.loads((DATA/'projects'/pid/'project.json').read_text())
base=DATA/'stems/d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f'
paths=list(base.glob('*/vocal_parts/manifest.json'))+list(base.glob('*/wav/guitar.wav'))
paths += list((DATA/'projects'/pid/'runs'/project['current_run']).glob('*.json'))
snapshot=dict(current_run=project['current_run'],files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
if '--check' in sys.argv:
    assert snapshot==json.loads((ROOT/'preserved_before.json').read_text());print('Original analysis, parent guitar and vocal manifests unchanged:',len(paths),'files')
else:
    (ROOT/'preserved_before.json').write_text(json.dumps(snapshot,indent=2));print('Snapshotted',len(paths),'files')
