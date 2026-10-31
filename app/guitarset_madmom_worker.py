import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore',message='pkg_resources is deprecated')
from madmom.features.chords import CNNChordFeatureProcessor,CRFChordRecognitionProcessor
from madmom.processors import SequentialProcessor
from madmom.models import CHORDS_CNN_FEAT,CHORDS_CFCRF
parser=argparse.ArgumentParser(description='Resident audio-only madmom CNN+CRF worker')
parser.add_argument('jobs',type=Path)
parser.add_argument('--skip-reuse-check',action='store_true',help='Generate each planned recording once without a repeat inference')
args=parser.parse_args()
job_path=args.jobs; jobs=json.loads(job_path.read_text(encoding='utf-8'));directory=job_path.parent
if not jobs:raise ValueError('At least one audio-only job is required')
weights=[{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in [CHORDS_CNN_FEAT[0],CHORDS_CFCRF[0]]]
processor=SequentialProcessor([CNNChordFeatureProcessor(),CRFChordRecognitionProcessor()])
records=[]
for job in jobs:
 began=time.perf_counter(); y=np.load(job['npy']);segments=processor(y);target=Path(job['lab']);target.parent.mkdir(parents=True,exist_ok=True)
 text=''.join(f'{float(start):.3f} {float(end):.3f} {label}\n' for start,end,label in segments)
 pending=target.with_suffix('.pending.lab');pending.write_text(text,encoding='utf-8');pending.replace(target)
 provenance={'schema':'madmom_chord_cache_v1','audio':job['audio_identity'],'lab':{'path':str(target),'bytes':target.stat().st_size,'mtime_ns':target.stat().st_mtime_ns,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},'sample_rate':44100,'offset':0,'requested_duration':None,'duration':len(y)/44100,'timestamp_origin':'decoded_window_start','frontend':'madmom CNNChordFeatureProcessor + CRFChordRecognitionProcessor','runtime_python':sys.executable,'model_weights':weights,'threads':2,'resident_processor_reused':True,'reference_hints_used':False}
 target.with_name(target.name+'.provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
 record={'id':job['id'],'split':job['split'],'audio':job['audio_identity'],'lab':provenance['lab'],'duration':len(y)/44100,'segments':len(segments),'elapsed_seconds':round(time.perf_counter()-began,3),'status':'complete'};records.append(record)
 (directory/'raw_generation_report.json').write_text(json.dumps({'protocol':'pure full audio at44100; one resident CNN+CRF; no labels/key/grid used','reference_hints_used':False,'threads':2,'model_weights':weights,'completed_count':len(records),'planned_count':len(jobs),'remaining_count':len(jobs)-len(records),'tracks':records},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:record[k] for k in ['id','split','elapsed_seconds','status']})+' completed='+str(len(records)),flush=True)
 # Historical callers can request the old repeat check; new baselines infer once.
 if args.skip_reuse_check or job is not jobs[0]:Path(job['npy']).unlink()
if args.skip_reuse_check:
 (directory/'resident_reuse_check.json').write_text(json.dumps({'first_recording_reprocessed_after_all_tracks':False,'status':'not_requested'}),encoding='utf-8')
else:
 check=processor(np.load(jobs[0]['npy']));text=''.join(f'{float(start):.3f} {float(end):.3f} {label}\n' for start,end,label in check)
 identical=text==Path(jobs[0]['lab']).read_text(encoding='utf-8');(directory/'resident_reuse_check.json').write_text(json.dumps({'first_recording_reprocessed_after_all_tracks':True,'exact_lab_text_equal':identical}),encoding='utf-8')
 if not identical:raise RuntimeError('Resident processor changed repeated audio output')
 Path(jobs[0]['npy']).unlink()
