import json,concurrent.futures
import download_duets_subset as source
source.URL='https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1'
source.DATA=source.DATA/'guitarset/audio'
index=json.loads((source.ROOT/'guitarset_zip_index.json').read_text())
names=[f'{player}_{song}_{role}_mic.wav' for player in ['00','01','02','04']
       for song in ['Rock2-142-D','Jazz1-200-B','Funk1-97-C','BN1-129-Eb'] for role in ['comp','solo']]
selected=[x for x in index if x['name'] in names]
assert len(selected)==32
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(source.retrieve,selected))
(source.ROOT/'guitarset_training_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
