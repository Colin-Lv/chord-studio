"""Predeclare extra held-out cases before viewing their output; never train on player05."""
import json,concurrent.futures
import download_duets_subset as source
source.URL='https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1'
source.DATA=source.DATA/'guitarset/audio'
index=json.loads((source.ROOT/'guitarset_zip_index.json').read_text())
songs=['Funk1-97-C','BN1-129-Eb','Rock3-117-Bb','Jazz3-150-C']
names=[f'05_{song}_{role}_mic.wav' for song in songs for role in ['comp','solo']]
selected=[x for x in index if x['name'] in names]
assert len(selected)==8
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(source.retrieve,selected))
(source.ROOT/'guitarset_final_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
