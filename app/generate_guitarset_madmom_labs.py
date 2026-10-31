"""Generate frozen GuitarSet LABs with one audio-only madmom worker.

Test10 requires an explicit --splits test10 request. No labels are opened.
"""
import argparse,csv,json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
from chord_recognizer import decode_audio
from madmom_bridge import DEFAULT_MADMOM_PYTHON,file_identity
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('experiment_dir',type=Path,help='Directory containing frozen manifests/train40.csv and val10.csv')
parser.add_argument('--splits',nargs='+',choices=['train40','val10','test10'],default=['train40','val10'])
parser.add_argument('--run-name',default='',help='Optional distinct report subdirectory, e.g. test10_inference')
args=parser.parse_args()
if args.run_name and not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name):parser.error('run-name must be a plain directory name')
root=args.experiment_dir.resolve();run_dir=root/args.run_name if args.run_name else root;run_dir.mkdir(parents=True,exist_ok=True);temporary=run_dir/'.audio_cache';temporary.mkdir(exist_ok=True)
jobs=[]
try:
 for split in args.splits:
  for row in csv.DictReader((root/'manifests'/(split+'.csv')).open(encoding='utf-8-sig')):
   audio=Path(row['audio']);npy=temporary/(row['id']+'.npy');identity=file_identity(audio);signal=decode_audio(audio,sample_rate=44100).astype(np.float32);np.save(npy,signal)
   jobs.append({'id':row['id'],'split':split,'npy':str(npy),'lab':row['madmom_lab'],'audio_identity':identity})
 print('Prepared audio-only arrays: '+str(len(jobs)),flush=True)
 jobpath=run_dir/'audio_only_jobs.json';jobpath.write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf-8')
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
 status=subprocess.run([str(DEFAULT_MADMOM_PYTHON),str(Path(__file__).with_name('guitarset_madmom_worker.py').resolve()),str(jobpath)],env=env).returncode
 for job in jobs:
  if file_identity(Path(job['audio_identity']['path'])) != job['audio_identity']:
   raise RuntimeError('Audio changed during inference')
 raise SystemExit(status)
finally:
 for job in jobs:Path(job['npy']).unlink(missing_ok=True)
 if not any(temporary.iterdir()):temporary.rmdir()
