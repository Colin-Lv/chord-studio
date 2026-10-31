"""Download verified official key corpora; keep rights and content identities.

GiantSteps+: https://doi.org/10.5281/zenodo.1095691, CC BY-SA 4.0.
Billboard annotations: DDMAL, CC0; original commercial audio is not provided.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import requests

BILLBOARD='https://ddmal.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/'


def fetch(url,path,md5=None):
    if not path.is_file():
        temp=path.with_suffix(path.suffix+'.part')
        with requests.get(url,stream=True,timeout=(30,90)) as response:
            response.raise_for_status()
            with temp.open('wb') as stream:
                for block in response.iter_content(1024*1024):stream.write(block)
        temp.replace(path)
    a,b=hashlib.md5(),hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):a.update(block);b.update(block)
    if md5 and a.hexdigest()!=md5:raise ValueError('Published checksum mismatch: '+str(path))
    return dict(url=url,path=str(path.resolve()),bytes=path.stat().st_size,md5=a.hexdigest(),sha256=b.hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    record=requests.get('https://zenodo.org/api/records/1095691',timeout=30);record.raise_for_status();record=record.json()
    (root/'giantsteps_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    receipts=[]
    for f in record['files']:
        receipt=fetch(f['links']['self'],root/f['key'],f['checksum'].split(':',1)[1]);receipt.update(doi=record['doi'],license=record['metadata']['license']);receipts.append(receipt)
        print('Verified',f['key'],flush=True)
    index=[];(root/'audio').mkdir(exist_ok=True)
    with zipfile.ZipFile(root/'audio.zip') as archive:
        for name in archive.namelist():
            if '__MACOSX' in name or not name.lower().endswith('.mp3'):continue
            match=re.match(r'(\d+)',Path(name).name)
            if not match:raise ValueError('Unknown track identity '+name)
            identifier=match[1];data=archive.read(name);sha=hashlib.sha256(data).hexdigest();target=root/'audio'/(identifier+'.mp3')
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=sha:
                raise ValueError('Existing audio differs from the annotation-matched archive: '+str(target))
            if not target.exists():target.write_bytes(data)
            index.append(dict(id=identifier,archive_member=name,sha256=sha,bytes=len(data)))
    (root/'audio_index.json').write_text(json.dumps(dict(source_record=record['doi'],files=index),ensure_ascii=False,indent=2),encoding='utf-8')
    response=requests.get(BILLBOARD,timeout=30);response.raise_for_status();html=response.text
    (root/'billboard_source.html').write_text(html,encoding='utf-8')
    links={text:url for url,text in re.findall(r'href="([^"]+)"[^>]*>(billboard[^<]+)',html)}
    for name,target in [('billboard-2.0-index.csv','billboard-index.csv'),('billboard-2.0-salami_chords.tar.gz','billboard-salami.tar.gz'),('billboard-2.0.1-lab.tar.gz','billboard-lab.tar.gz')]:
        receipt=fetch(links[name],root/target);receipt.update(license='CC0',source_page=BILLBOARD,audio=False);receipts.append(receipt)
    (root/'corpus_downloads.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
