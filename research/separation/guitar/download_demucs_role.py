import json,requests
from pathlib import Path
from download_specialists import walk
from download_assets import download,ROOT
item=next(v for k,v in walk(json.loads((ROOT/'community_models.json').read_text())) if k=='demucs4_lead_rhythm_guitar_drypaint')
for field in ['config_url','checkpoint_url']:
    url=item[field];name=url.split('/')[-1].split('?')[0]
    r=requests.head(url,allow_redirects=False,timeout=30);r.raise_for_status()
    size=int(r.headers.get('x-linked-size') or r.headers['content-length'])
    print(name,size,flush=True)
    download((url,Path('D:/ChordStudio/data/models/guitar')/name,size,None))
