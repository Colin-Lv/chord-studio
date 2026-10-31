"""Audio-only external evidence, bounded workers, reusable hashed artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_global_key_fusion import signature
from key_fusion import analyze_key_evidence, read_raw_lab


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--phases',nargs='+',default=['development','calibration','sealed_test'])
    parser.add_argument('--evidence-only',action='store_true',help='Rebuild evidence from verified acoustic outputs; never run an acoustic subprocess')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.split.read_text(encoding='utf-8'))
    rows=[r for r in manifest['rows'] if r['selected'] and r['split'] in args.phases]
    app=Path(__file__).resolve().parent
    env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    def one(row):
        began=time.perf_counter();audio=Path(row['audio']);directory=args.output/row['id'];directory.mkdir(exist_ok=True)
        if not audio.is_file():raise FileNotFoundError('Extract the verified GiantSteps+ audio.zip first: '+str(audio))
        audio_hash=signature(audio);commands=[]
        index=json.loads((audio.parent.parent/'audio_index.json').read_text(encoding='utf-8'))
        expected=next(v['sha256'] for v in index['files'] if v['id']==row['id'])
        if audio_hash['sha256']!=expected:raise ValueError('Audio differs from annotation-matched Zenodo archive')
        paths=dict(cnn=directory/'cnn.json',neutral=directory/'neutral_analysis.json',raw=directory/'raw.lab')
        cache=directory/'acoustics.json'
        cached=json.loads(cache.read_text()) if cache.exists() else None
        valid_cache=bool(cached and cached.get('audio',{}).get('sha256')==audio_hash['sha256'] and all(
            p.is_file() and signature(p)['sha256']==cached.get('outputs',{}).get(k,{}).get('sha256') for k,p in paths.items()))
        if not valid_cache:
            if args.evidence_only:raise ValueError('Missing or changed acoustic cache in --evidence-only mode')
            commands=[
                [sys.executable,str(app/'madmom_bridge.py'),str(audio),'-o',str(paths['raw'])],
                [sys.executable,str(app/'key_bridge.py'),str(audio),'-o',str(paths['cnn'])],
                [sys.executable,str(app/'chord_studio.py'),str(audio),'-o',str(directory/'neutral'),'--beat-source','madmom',
                 '--feature-mode','fast','--analysis-model','template','--mode','seventh','--all-key-candidates','--no-html','--no-structure']]
            with (directory/'acoustics.log').open('w',encoding='utf-8') as log:
                for command in commands:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            cache.write_text(json.dumps(dict(audio=audio_hash,commands=commands,elapsed_seconds=time.perf_counter()-began,
                outputs={k:signature(p) for k,p in paths.items()},reference_inputs=False),indent=2),encoding='utf-8')
        cnn=json.loads(paths['cnn'].read_text(encoding='utf-8'));neutral=json.loads(paths['neutral'].read_text(encoding='utf-8'))
        if signature(audio)['sha256']!=audio_hash['sha256']:raise ValueError('Audio changed during inference')
        evidence=analyze_key_evidence(cnn,neutral,read_raw_lab(paths['raw']))['evidence']
        # References join only after acoustic + harmonic evidence is complete.
        record=dict(id=row['id'],dataset=row['dataset'],genre=row['genre'],group=row['group'],split=row['split'],reference=row['reference'],
                    evidence=evidence,inputs={k:signature(p) for k,p in paths.items()},audio=audio_hash,
                    evidence_implementation={name:signature(app/name)['sha256'] for name in ['key_fusion.py','detect_key_timeline.py','chord_studio.py']},
                    cnn_models=cnn.get('model_files'),new_acoustic_inference=not valid_cache,elapsed_seconds=time.perf_counter()-began)
        (directory/'case.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        return dict(id=row['id'],split=row['split'],seconds=time.perf_counter()-began)
    progress=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks={executor.submit(one,row):row for row in rows}
        for future in as_completed(tasks):
            try:result=future.result()
            except Exception as exc:result=dict(id=tasks[future]['id'],split=tasks[future]['split'],error=f'{type(exc).__name__}: {exc}')
            progress.append(result)
            (args.output/'progress.json').write_text(json.dumps(dict(split=signature(args.split),total=len(rows),completed=progress),indent=2),encoding='utf-8')
            print(f"{len(progress)}/{len(rows)} {result}",flush=True)


if __name__=='__main__':main()
