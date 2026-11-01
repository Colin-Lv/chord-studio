from pathlib import Path
import json, requests, concurrent.futures
from download_assets import download, ROOT, DATA

def walk(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,dict) and 'checkpoint_url' in v:yield k,v
            else:yield from walk(v)
    elif isinstance(o,list):
        for v in o:yield from walk(v)

if __name__=='__main__':
    jobs=[]
    wanted=['mbr_lead_rhythm_guitar_listra92','bs_mega_53stem_acoustic-guitar_mvsep','bs_mega_53stem_electric-guitar_mvsep']
    for key,item in walk(json.loads((ROOT/'community_models.json').read_text(encoding='utf-8'))):
        if key not in wanted:continue
        for field in ['checkpoint_url','config_url']:
            url=item[field];name=url.split('/')[-1].split('?')[0]
            response=requests.head(url,allow_redirects=False,timeout=30);response.raise_for_status()
            size=int(response.headers.get('x-linked-size') or response.headers['content-length'])
            jobs.append((url,Path('D:/ChordStudio/data/models/guitar')/name,size,None))
    for key in ['egset12','guitarset']:
        for item in json.loads((ROOT/(key+'_metadata.json')).read_text())['files']:
            name=item['key']
            if key=='egset12' and not name.endswith(('.wav','.jams','.gp')):continue
            if key=='guitarset' and name not in ['annotation.zip','audio_mono-mic.zip']:continue
            jobs.append((item['links']['self'],DATA/key/name,item['size'],item['checksum']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(download,jobs))
    (ROOT/'specialists_download_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
