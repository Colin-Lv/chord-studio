"""Retrieve individual ZIP entries by public byte ranges, verify CRC and hashes."""
import concurrent.futures, hashlib, json, struct, time, zlib
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
DATA=Path('D:/ChordStudio/data/research/guitar')
URL='https://zenodo.org/records/12802440/files/GuitarDuets.zip?download=1'
index=json.loads((ROOT/'guitarduets_zip_index.json').read_text())
selected=[x for x in index if x['size'] and ('/test/' in x['name'] or '/Real/Valses Poeticos ' in x['name'])]
def retrieve(item):
    target=DATA/item['name'];target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        start=item['offset'];end=start+item['compressed_size']+4096
        for attempt in range(4):
            try:
                r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=90);r.raise_for_status();assert r.status_code==206
                header=struct.unpack('<4s5H3I2H',r.content[:30]);assert header[0]==b'PK\x03\x04'
                pos=30+header[-2]+header[-1];packed=r.content[pos:pos+item['compressed_size']]
                audio=zlib.decompress(packed,-15) if header[3]==8 else packed
                assert len(audio)==item['size'] and zlib.crc32(audio)==item['crc']
                target.write_bytes(audio);break
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    raw=target.read_bytes();assert len(raw)==item['size'] and zlib.crc32(raw)==item['crc']
    print('verified',item['name'],flush=True)
    return dict(item,path=str(target),sha256=hashlib.sha256(raw).hexdigest(),url=URL)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(retrieve,selected))
    (ROOT/'duets_subset_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
