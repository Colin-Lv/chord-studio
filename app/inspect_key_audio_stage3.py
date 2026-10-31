"""Re-extract the original chord model's potentials and exact CRF marginals.

Run with the existing madmom39 interpreter. No new model weights or thresholds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from scipy.special import softmax
from app.chord_crf_diagnostics import marginals, maximum_path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out, pcm_dir):
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor, majmin_targets_to_chord_labels
    from madmom.models import CHORDS_CNN_FEAT, CHORDS_CFCRF
    import madmom
    out.mkdir(parents=True,exist_ok=False)
    frontend=CNNChordFeatureProcessor();decoder=CRFChordRecognitionProcessor();crf=decoder.processors[0]
    labels=['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
    labels=[x+':maj' for x in labels]+[x+':min' for x in labels]+['N']
    records=[]
    pcm_manifest=json.loads((pcm_dir/'manifest.json').read_text(encoding='utf-8'))
    for sid in ['191347','853834']:
        audio=Path('D:/ChordStudio/_archive/2026-10-30/benchmarks/external_key_data/audio')/(sid+'.mp3')
        case=Path('benchmarks/reports/harmonic_key_v2_20261006/acoustics')/sid
        provenance=json.loads((case/'raw.lab.provenance.json').read_text(encoding='utf-8'))
        assert sha(audio)==provenance['audio']['sha256']
        print('Extracting frozen chord frontend:',sid,flush=True)
        pcm=next(r for r in pcm_manifest['rows'] if r['id']==sid)
        assert pcm['audio_sha256']==sha(audio)
        assert pcm['pcm_file_sha256']==sha(pcm_dir/(sid+'.npy'))
        features=frontend(np.load(pcm_dir/(sid+'.npy')))
        emission=np.asarray(features,dtype=np.float64).dot(crf.W)+crf.c
        posterior,log_z=marginals(emission,crf.pi,crf.A,crf.tau)
        ours=maximum_path(emission,crf.pi,crf.A,crf.tau)
        original=crf(features)
        assert np.array_equal(ours,original),'Viterbi implementation mismatch'
        decoded=majmin_targets_to_chord_labels(original,fps=10)
        rows=[dict(start=float(r['start']),end=float(r['end']),label=str(r['label'])) for r in decoded]
        (out/(sid+'.lab')).write_text('\n'.join('{start:.3f} {end:.3f} {label}'.format(**r) for r in rows)+'\n',encoding='utf-8')
        cached=[line.split() for line in (case/'raw.lab').read_text().splitlines() if line.strip()]
        cached_ids=np.full(len(features),-1,dtype=int)
        times=np.arange(len(features))/10
        for start,end,label in cached:
            cached_ids[(times>=float(start)-1e-7)&(times<float(end)-1e-7)]=labels.index(label)
        local=softmax(emission,axis=1)
        top=lambda p:[dict(label=labels[i],mean=float(p.mean(axis=0)[i])) for i in np.argsort(-p.mean(axis=0))[:6]]
        intervals=[]
        for r in rows:
            mask=(times>=r['start']-1e-7)&(times<r['end']-1e-7)
            intervals.append(dict(**r,crf_top=top(posterior[mask]),local_top=top(local[mask])))
        np.savez_compressed(out/(sid+'.npz'),features=features,emissions=emission,posterior=posterior,
                            local_softmax=local,path=original,labels=np.array(labels),times=times)
        record=dict(id=sid,audio_sha256=sha(audio),frames=len(features),fps=10,
                    cached_frame_agreement=float((cached_ids==original).mean()),viterbi_implementation_matches=True,
                    crf_top=top(posterior),local_top=top(local),log_partition=log_z,intervals=intervals,
                    max_marginal_row_sum_error=float(np.max(np.abs(posterior.sum(axis=1)-1))),
                    interpretation='model-internal marginals, not calibrated chord truth; local softmax excludes transitions')
        records.append(record)
        print(json.dumps({k:record[k] for k in ['id','frames','cached_frame_agreement','crf_top','local_top']},ensure_ascii=False),flush=True)
    result=dict(records=records,pcm_manifest=pcm_manifest,model_hashes={str(p):sha(p) for p in CHORDS_CNN_FEAT+CHORDS_CFCRF},
                madmom_version=madmom.__version__,implementation_sha256=sha(__file__),
                marginal_implementation_sha256=sha(Path('app/chord_crf_diagnostics.py')))
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--pcm-dir',required=True,type=Path)
    args=ap.parse_args();main(args.out,args.pcm_dir)
