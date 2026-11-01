"""Generate both complete song results through the installed app's real job queue."""
import json,time,urllib.request,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parent;DATA=Path('D:/ChordStudio/data');PID='aa5a41797bb343e590abcf4d8b27ae54'
session=json.loads((DATA/'active-session.json').read_text());BASE='http://127.0.0.1:'+str(session['port'])
headers={'x-chord-session':session['token'],'X-Harmony-Client':'studio'}
def request(path,body=None,extra=None):
    h={**headers,**(extra or {})}
    if body is not None:h['Content-Type']='application/json'
    req=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers=h)
    return urllib.request.urlopen(req,timeout=40)
def api(path,body=None):
    with request(path,body) as r:return json.load(r)
catalog=api(f'/api/projects/{PID}/stems/guitar-parts-catalog');assert all(r['ready'] for r in catalog['modes'])
base=DATA/'stems/d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f/bs_roformer_sw'
results=[]
for mode in ['timbre','duet']:
    job=api(f'/api/projects/{PID}/guitar-parts',dict(model='bs_roformer_sw',mode=mode,force=False))
    assert job['profile']=='guitar_parts' and job['guitar_mode']==mode
    print('submitted',mode,job['id'],flush=True);deadline=time.monotonic()+1800;last=None
    while time.monotonic()<deadline:
        row=api('/api/jobs/'+job['id'])
        state=(row['status'],row.get('stage_label'))
        if state!=last:print(mode,state,flush=True);last=state
        if row['status']=='completed':break
        if row['status'] in ['failed','interrupted']:raise RuntimeError(row.get('error'))
        time.sleep(3)
    else:raise TimeoutError(mode)
    manifest=api(f'/api/projects/{PID}/stems/guitar-parts?model=bs_roformer_sw&mode={mode}')
    parent,sr=sf.read(base/'wav/guitar.wav',dtype='float32',always_2d=True);total=np.zeros_like(parent);tracks=[]
    for track in manifest['tracks']:
        p=base/track['wav'];x,rate=sf.read(p,dtype='float32',always_2d=True)
        assert rate==sr and x.shape==parent.shape and np.isfinite(x).all();total+=x
        for key in ['urls','wav_urls']:
            with request(manifest[key][track['id']],extra={'Range':'bytes=0-4095'}) as response:
                assert response.status==206 and len(response.read())==4096
        tracks.append(dict(id=track['id'],file=str(p),rms_db=track['rms_db'],peak=float(np.abs(x).max())))
    error=float(np.abs(parent-total).max());assert error<2e-6
    results.append(dict(mode=mode,job_id=job['id'],generation=manifest['generation'],seconds=manifest['seconds'],
        duration=manifest['duration'],samples=len(parent),reconstruction_max_error=error,tracks=tracks,inference=manifest['inference']))
    (ROOT/'installed_results.json').write_text(json.dumps(results,indent=2))
    print('verified',mode,len(tracks),'tracks',manifest['duration'],'seconds',flush=True)
print('Both complete song results are saved and served by the installed app.',flush=True)
