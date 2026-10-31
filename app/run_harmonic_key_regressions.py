"""Evaluate known audio regressions only after the new harmonic contract freeze.

Reuse saved CNN/LAB/grid. Only the fixed chroma diagnostic decodes audio anew.
Reference labels are joined after predictions, never used by either engine.
"""
import argparse
import csv
import json
from pathlib import Path
import time

from evaluate_global_key_fusion import signature
from evaluate_harmonic_key_v2 import load_contract,save_report
from harmonic_key_v2 import Policy,analyze
from key_fusion import analyze_key_evidence,key_identity,read_raw_lab
from key_profile_baseline import analyze_audio


def evaluate(items,contract_path,output,dataset):
    contract=load_contract(contract_path);policy=Policy(**contract['policy'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];failures=[];began=time.perf_counter()
    for index,item in enumerate(items):
        try:
            paths={name:Path(item[name]) for name in ['audio','cnn','raw','neutral']}
            cnn=json.loads(paths['cnn'].read_text(encoding='utf-8'));neutral=json.loads(paths['neutral'].read_text(encoding='utf-8'))
            events=read_raw_lab(paths['raw'])
            profile=analyze_audio(paths['audio'],output/'profiles'/(item['id']+'.json'))
            if abs(neutral['duration']-profile['duration'])>1/22050+1e-6:raise ValueError('Frontend duration differs')
            current=analyze(events,neutral['duration'],neutral.get('bars',[]),policy)
            if item.get('old_case'):
                old=json.loads(Path(item['old_case']).read_text(encoding='utf-8'))['evidence']
            else:old=analyze_key_evidence(cnn,neutral,events)['evidence']
            prediction_path=output/'predictions'/(item['id']+'.json');prediction_path.parent.mkdir(exist_ok=True)
            prediction_path.write_text(json.dumps(current,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            rows.append(dict(id=item['id'],reference=key_identity(item['reference']),genre=item.get('genre',dataset),
                cnn=key_identity(cnn['key']),harmonic_v1=old['raw_chord_functional']['key'],chroma=profile['key'],
                harmonic_v2=current['key'],tonic=current['tonic'],mode=current['mode'],status=current['status'],
                prediction_file=signature(prediction_path),inputs={name:signature(path) for name,path in paths.items()},
                cnn_models=cnn.get('model_files'),
                profile=signature(output/'profiles'/(item['id']+'.json')),prediction=current))
            print(f'{dataset} {index+1}/{len(items)}',flush=True)
        except Exception as exc:failures.append(dict(id=item['id'],error=repr(exc)))
    result=save_report(rows,output/'report.json',dict(dataset_kind='known_regression',dataset=dataset,
        requested=len(items),failures=failures,contract=signature(contract_path),elapsed_seconds=time.perf_counter()-began,
        new_neural_inference=False,new_profile_audio_analysis=True,new_training=False,test_used_for_selection=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--old-root',type=Path,default=Path('benchmarks/reports/key_resolver_20261006'))
    parser.add_argument('--fusion-root',type=Path,default=Path('benchmarks/reports/key_fusion_20261006'))
    args=parser.parse_args();load_contract(args.contract)
    old=json.loads((args.old_root/'splits/giantsteps_split.json').read_text(encoding='utf-8'))
    items=[]
    for row in old['rows']:
        if not row['selected']:continue
        directory=args.old_root/'giantsteps'/row['id']
        items.append(dict(row,cnn=str(directory/'cnn.json'),raw=str(directory/'raw.lab'),
                          neutral=str(directory/'neutral_analysis.json'),old_case=str(directory/'case.json')))
    results=[evaluate(items,args.contract,args.output/'giantsteps180','giantsteps180')]
    for name in ['beatles17','xuni']:
        manifest=args.fusion_root/name/'manifest.csv'
        items=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
        for item in items:
            for field in ['audio','cnn','raw','neutral']:item[field]=str((manifest.parent/item[field]).resolve())
        results.append(evaluate(items,args.contract,args.output/name,name))
    if any(result['failures'] for result in results):raise RuntimeError('Regression input failures; see reports')


if __name__=='__main__':main()
