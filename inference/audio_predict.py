"""Bounded local audio-to-retained10 research inference, with reusable audio-only caches."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, sys, time, uuid
from pathlib import Path
from datetime import datetime, timezone
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
HERE=Path(__file__).resolve().parent
MIB=1024**2
def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(MIB),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def log(phase,**value): print(json.dumps(dict(utc=now(),phase=phase,**value),ensure_ascii=False),flush=True)
def verify_bundle():
    manifest=read(HERE/'audio_manifest.json')
    for name,h in manifest['files_sha256'].items():
        if sha(HERE/name)!=h:raise ValueError('Audio bundle file differs: '+name)
    return manifest
def runtime_identity(config):
    expected=read(config);actual={}
    for name in ('ffmpeg','sonic_annotator','nnls_plugin','madmom_python'):
        p=Path(expected[name]['path']).resolve()
        if not p.is_file() or sha(p)!=expected[name]['sha256']:raise ValueError('Installed runtime differs: '+name)
        actual[name]=dict(path=str(p),sha256=expected[name]['sha256'])
    for name,item in expected['madmom_models'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Installed madmom model differs: '+name)
    actual['madmom_models']=expected['madmom_models'];actual['madmom_version']=expected['madmom_version']
    return actual
def run(command,folder,label,env=None,timeout=600):
    proc=subprocess.run(command,capture_output=True,env=env,timeout=timeout)
    (folder/(label+'.stdout')).write_bytes(proc.stdout);(folder/(label+'.stderr')).write_bytes(proc.stderr)
    if proc.returncode:raise RuntimeError(label+' failed: '+proc.stderr.decode('utf-8',errors='replace')[-1800:])
    return proc.stdout
def worker(request_path):
    request=read(request_path);output=Path(request['output_dir']);audio=Path(request['audio']);cache=Path(request['cache_dir'])
    manifest=verify_bundle();runtime=runtime_identity(request['runtime']);stat=audio.stat();source=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=sha(audio))
    if source['bytes']>512*MIB:raise ValueError('Audio exceeds the supported512MiB source limit')
    identity=dict(schema='retained10_audio_frontend_v82',audio_sha256=source['sha256'],frontend_sha256={k:v for k,v in manifest['files_sha256'].items() if k in ('audio_predict.py','frontend_math.py','madmom_helper.py','chord_extension_model.py')},runtime=runtime)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();final=cache/key;reused=final.exists()
    sys.path.insert(0,str(HERE/'model'));sys.path.insert(0,str(HERE))
    import numpy as np
    from scipy.io import wavfile
    from predict import Retained10
    import frontend_math as front
    if reused:
        receipt=read(final/'features_complete.json')
        if receipt['identity']!=identity:raise ValueError('Cache identity mismatch')
        for name,h in receipt['files_sha256'].items():
            if sha(final/name)!=h:raise ValueError('Corrupt cache: '+name)
        if receipt['audio_seconds']>request['max_seconds']:raise ValueError('Cached audio exceeds this request duration limit')
        log('cache_reused',cache_key=key)
    else:
        final_pending=cache/(key+'.pending-'+uuid.uuid4().hex);final_pending.mkdir();final=final_pending
        try:
            duration_limit=request['max_seconds'];ffmpeg=runtime['ffmpeg']['path'];common=[ffmpeg,'-hide_banner','-loglevel','error','-threads','1','-i',str(audio),'-t',f'{duration_limit+.1:.6f}','-vn']
            pcm=run(common+['-ac','1','-ar','44100','-f','f32le','pipe:1'],final,'decode_mono',timeout=180)
            y=np.frombuffer(pcm,dtype=np.float32)
            if not len(y) or not np.isfinite(y).all():raise ValueError('Empty or nonfinite decoded audio')
            if len(y)>round(duration_limit*44100):raise ValueError('Audio exceeds duration limit; no truncated prediction produced')
            duration=len(y)/44100;np.save(final/'mono.npy',y);log('decoded',seconds=duration,samples=len(y))
            # Native-rate/channel RMS preserves the original v51 energy contract.
            native=final/'native.wav';native_limit=256*MIB
            run(common+['-c:a','pcm_f32le','-f','wav','-fs',str(native_limit),'-n',str(native)],final,'decode_native',timeout=180)
            if native.stat().st_size>=native_limit-4096:raise ValueError('Native PCM exceeds bounded256MiB temporary limit')
            rate,original=wavfile.read(native,mmap=True)
            if original.ndim==1:original=original[:,None]
            if rate>192000 or original.ndim!=2 or original.shape[1]>8 or original.dtype!=np.float32:raise ValueError('Native audio rate/channels outside the supported contract')
            if abs(len(original)/rate-duration)>1/rate+1/44100:raise ValueError('Native and mono decoding clocks disagree')
            native_info=dict(sample_rate=int(rate),samples=len(original),channels=original.shape[1])
            env=os.environ.copy();env['PATH']=str(Path(ffmpeg).parent)+os.pathsep+env.get('PATH','');env['VAMP_PATH']=str(Path(runtime['nnls_plugin']['path']).parent)
            helper_request=dict(audio=str(audio),mono=str(final/'mono.npy'),folder=str(final),runtime=runtime)
            write(final/'helper_request.json',helper_request)
            for action in ('anchors','deep'):
                log('extracting_'+action)
                run([runtime['madmom_python']['path'],'-I','-B',str(HERE/'madmom_helper.py'),str(final/'helper_request.json'),action],final,action,env=env,timeout=600)
            log('extracting_nnls')
            padded=final/'nnls_input.wav';wavfile.write(padded,44100,np.pad(y,(0,front.BLOCK_SIZE)))
            transform=final/'nnls.ttl';transform.write_text(front.TRANSFORM,encoding='utf-8')
            raw=run([runtime['sonic_annotator']['path'],'-q','-t',str(transform),'-w','csv','--csv-stdout','--csv-omit-filename','--csv-sample-timing','--csv-digits','9',str(padded)],final,'nnls',env=env,timeout=600)
            positions,activations=front._read_csv(raw.decode('utf-8'));keep=(positions>=0)&(positions<len(y));positions=positions[keep];activations=activations[:,keep]
            metadata=dict(duration_seconds=duration,decoded_samples=len(y),sample_rate=44100,reference_inputs=False)
            notes=front.NoteEvidence(positions.astype('float64')/44100,activations,metadata);front._validate_evidence(notes)
            anchors=[]
            for line in (final/'madmom.lab').read_text(encoding='utf-8').splitlines():
                a,b,c=line.split();anchors.append(front.TimedChord(float(a),float(b),c))
            features,cells=front.build_absolute_features(anchors,notes)
            windows=[front.TimedChord(max(0.,(c.start+c.end)/2-.25),min(duration,(c.start+c.end)/2+.25),'C:maj') for c in cells]
            features[:,:96]=front.features_for_intervals(notes,windows)[:,:96]
            serial=[dict(start=c.start,end=c.end,chord=c.chord) for c in cells]
            if not cells or cells[0].start!=0 or abs(cells[-1].end-duration)>1e-8 or any(abs(a.end-b.start)>1e-8 for a,b in zip(cells,cells[1:])):raise ValueError('Audio anchors do not cover the complete decoded timeline')
            energy=front.energy_features(original,rate,serial);del original
            with np.load(final/'deep.npz',allow_pickle=False) as z:
                if int(z['samples'])!=len(y) or int(z['sample_rate'])!=44100:raise ValueError('DeepChroma and NNLS audio clocks differ')
            np.savez_compressed(final/'features.npz',features=features);np.savez_compressed(final/'energy.npz',energy=energy);np.savez_compressed(final/'notes.npz',times=notes.times,activations=notes.activations,metadata=np.array(json.dumps(metadata)));write(final/'cells.json',serial)
            reclaimed=0
            for intermediate in ('mono.npy','native.wav','nnls_input.wav','nnls.stdout','decode_mono.stdout'):
                p=final/intermediate;reclaimed+=p.stat().st_size;p.unlink()
            after=audio.stat()
            if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):raise ValueError('Audio changed while extracting')
            files=('features.npz','cells.json','deep.npz','energy.npz','madmom.lab','notes.npz')
            receipt=dict(completed_utc=now(),identity=identity,audio_seconds=duration,samples=len(y),sample_rate=44100,native_audio=native_info,cells=len(cells),files_sha256={n:sha(final/n) for n in files},scratch_bytes_removed=reclaimed,new_recordings=0,new_fits=0,new_updates=0,reference_inputs=False)
            write(final/'features_complete.json',receipt);target=cache/key;final.rename(target);final=target
            log('features_complete',cells=len(cells),cache_key=key)
        except BaseException as exc:
            write(final/'failure.json',dict(failed_utc=now(),error_type=type(exc).__name__,error=str(exc),prediction_written=False));raise
    model=Retained10(HERE/'model');result=model.predict_folder(final)
    if result['segments'][0]['start']!=0 or abs(result['segments'][-1]['end']-receipt['audio_seconds'])>1e-8:raise ValueError('Prediction lacks full audio coverage')
    payload=dict(created_utc=now(),research_only=True,default_changed=False,reference_inputs=False,input_audio=str(audio),input_sha256=source['sha256'],sample_rate=44100,decoded_samples=receipt['samples'],audio_seconds=receipt['audio_seconds'],native_audio=receipt['native_audio'],cache_directory=str(final),cache_reused=reused,model_bundle_sha256=sha(HERE/'model/manifest.json'),weights_sha256=model.manifest['files_sha256']['retained10.npz'],supported_qualities=model.manifest['qualities'],real_expert_weights=result['real_expert_weights'],segments=result['segments'],limitations=model.manifest['limitations']+['Audio entry tested on a few existing development recordings only; no new accuracy or long-recording guarantee.'])
    write(output/'prediction.json',payload)
    with (output/'prediction.lab').open('x',encoding='utf-8',newline='\n') as f:f.write(''.join(f'{p["start"]:.17g} {p["end"]:.17g} {p["chord"]}\n' for p in result['segments']))
    np.savez_compressed(output/'posterior.npz',chord=result['chord_probabilities'],N=result['N_probabilities'],router=np.asarray(result['real_expert_weights']))
    log('prediction_complete',output=str(output/'prediction.json'),cache_reused=reused,segments=len(result['segments']))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,default=HERE/'runtime.local.json');parser.add_argument('--max-seconds',type=float,default=300.);parser.add_argument('--memory-budget-mib',type=int,default=1500)
    a=parser.parse_args();audio=a.audio.resolve();out=a.output_dir.resolve();cache=a.cache_dir.resolve()
    if not audio.is_file():raise ValueError('Audio must be an existing local file')
    if out.exists():raise ValueError('Output directory already exists; use a new name')
    if not math.isfinite(a.max_seconds) or not 0<a.max_seconds<=600:raise ValueError('Choose0<max-seconds<=600')
    if not 400<=a.memory_budget_mib<=2400:raise ValueError('Choose400..2400MiB memory budget')
    import msvcrt,psutil
    cache.mkdir(parents=True,exist_ok=True)
    with (cache/'pipeline.lock').open('a+b') as lock:
        lock.write(b'1');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Another audio job owns this cache lock')
        try:
            # The budget is a ceiling for the worker tree, not an upfront reservation.
            # Measured peaks are 185-510 MiB even on four-minute songs, and Windows
            # reclaims standby memory on demand, so gating on "budget + reserve" made
            # normal 16 GiB machines with a browser open fail every analysis.
            available_mib=psutil.virtual_memory().available//MIB
            memory_floor_mib=256
            if available_mib<memory_floor_mib:
                raise RuntimeError(f'可用内存过低：当前仅 {available_mib} MiB，推理至少需要约 {memory_floor_mib} MiB 可用内存。请关闭部分应用后重试。')
            budget_mib=a.memory_budget_mib
            if available_mib<budget_mib+256:
                budget_mib=max(400,available_mib-192)
                log('memory_budget_adapted',requested_mib=a.memory_budget_mib,effective_mib=budget_mib,available_mib=available_mib)
            if psutil.disk_usage(str(cache)).free<4*1024**3:
                raise RuntimeError(f'缓存磁盘空间不足：至少需要 4 GiB，当前仅剩 {psutil.disk_usage(str(cache)).free/1024**3:.1f} GiB。')
            out.mkdir(parents=True)
            req=dict(audio=str(audio),output_dir=str(out),cache_dir=str(cache),runtime=str(a.runtime.resolve()),max_seconds=a.max_seconds);write(out/'request.json',req)
            start=time.monotonic();peak=0;minimum=psutil.virtual_memory().available;abort=None
            with (out/'worker.stdout').open('wb') as stdout,(out/'worker.stderr').open('wb') as stderr:
                child=subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--_worker',str(out/'request.json')],stdout=stdout,stderr=stderr,cwd=out)
                proc=psutil.Process(child.pid)
                while child.poll() is None:
                    available=psutil.virtual_memory().available;minimum=min(minimum,available)
                    try:owned=[proc]+proc.children(recursive=True)
                    except psutil.Error:owned=[]
                    rss=0
                    for item in owned:
                        try:rss+=item.memory_info().rss
                        except psutil.Error:pass
                    peak=max(peak,rss)
                    # Windows keeps "available" near zero when it leans on standby
                    # memory, so a low system number alone is not a reason to kill a
                    # 200-500 MiB job. Only abort when this tree itself is over its
                    # ceiling, or when the machine is starving and we are a real
                    # contributor to that pressure.
                    if rss>(budget_mib+256)*MIB:abort='worker_tree_budget'
                    elif available<32*MIB and rss>(budget_mib//2)*MIB:abort='system_memory_exhausted'
                    elif time.monotonic()-start>1200:abort='20minute_run_timeout'
                    elif psutil.disk_usage(str(cache)).free<2*1024**3:abort='2GiB_disk_emergency_floor'
                    if abort:
                        for item in reversed(owned):
                            try:item.terminate()
                            except psutil.Error:pass
                    time.sleep(.2)
                code=child.wait()
            status=dict(completed_utc=now(),exit_code=code,abort=abort,seconds=time.monotonic()-start,peak_process_tree_bytes=peak,minimum_system_available_bytes=minimum,budget_MiB=budget_mib,successful=code==0 and abort is None,only_owned_children_terminated=bool(abort))
            write(out/'run_status.json',status)
            if not status['successful']:raise RuntimeError('Audio inference failed; inspect '+str(out/'worker.stderr'))
            log('complete',output=str(out/'prediction.json'),LAB=str(out/'prediction.lab'),seconds=status['seconds'],peak_MiB=round(peak/MIB,1))
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
if __name__=='__main__':
    try:
        if len(sys.argv)==3 and sys.argv[1]=='--_worker':worker(Path(sys.argv[2]))
        else:main()
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);raise SystemExit(1)
