"""Download public research assets, retaining URLs, hashes and upstream metadata."""
from pathlib import Path
import concurrent.futures, hashlib, json, requests, time

ROOT=Path(__file__).resolve().parent
DATA=Path('D:/ChordStudio/data/research/guitar')
MODEL=Path('D:/ChordStudio/data/models/mus3d')
metadata=json.loads((ROOT/'guitarduets_metadata.json').read_text())
file=metadata['files'][0]
jobs=[(file['links']['self'], DATA/file['key'],file['size'],file['checksum'])]
hf=json.loads((ROOT/'mus3d_metadata.json').read_text())
for item in hf:
    name=item['path']
    if item['type']=='file' and (name.startswith(('BS-Locoformer_query/','SEG_mixemb/','classifiers/')) or name=='README.md'):
        jobs.append(('https://huggingface.co/remus-aalto/MuS3D/resolve/main/'+name,MODEL/name,item['size'],None))

def download(job):
    url,dest,size,checksum=job;dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_name(dest.name+'.partial')
    if not dest.exists() or dest.stat().st_size!=size:
        for attempt in range(6):
            try:
                offset=partial.stat().st_size if partial.exists() else 0
                with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code!=206:offset=0
                    with partial.open('ab' if offset else 'wb') as target:
                        for chunk in response.iter_content(1024*1024):target.write(chunk)
                assert partial.stat().st_size==size,(partial,partial.stat().st_size,size)
                partial.replace(dest);break
            except Exception as exc:
                print('retry',dest.name,attempt,str(exc)[:200],flush=True);time.sleep(2)
        else:raise RuntimeError(str(dest))
    sha=hashlib.sha256();md5=hashlib.md5()
    with dest.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk);md5.update(chunk)
    if checksum:assert checksum=='md5:'+md5.hexdigest(),(dest,checksum)
    result=dict(url=url,path=str(dest),size=size,sha256=sha.hexdigest(),md5=md5.hexdigest())
    dest.with_name(dest.name+'.download.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ready',str(dest),size,flush=True)
    return result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(download,jobs))
    (ROOT/'download_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
