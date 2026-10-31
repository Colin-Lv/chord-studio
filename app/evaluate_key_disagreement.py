"""Evaluate frozen resolver and rule baseline; no threshold search in this file."""
import argparse
import csv
import json
from pathlib import Path

from key_disagreement import load_model
from train_key_disagreement import read_cases,evaluate_cases
from evaluate_global_key_fusion import signature


def export_tables(report,output):
    """Readable per-song decisions and full 24 x 24 confusion matrices."""
    from key_fusion import key_identity
    keys=[key_identity(root+mode) for mode in ['', 'm'] for root in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']]
    output=Path(output)
    fields=['id','dataset','genre','reference','cnn','harmonic','final','decision_path','correct','cnn_correct','resolved','reason']
    with output.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in report['resolver']['rows']:writer.writerow({**row,'reason':json.dumps(row['reason'],ensure_ascii=False)})
    for method in ['cnn','harmonic','final']:
        matrix=report['resolver']['confusion_matrix'][method]
        with output.with_name(output.stem+'_'+method+'_confusion.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['reference / prediction']+keys)
            for ref in keys:writer.writerow([ref]+[matrix.get(f'{ref} -> {prediction}',0) for prediction in keys])


def regression(manifest,artifact,output):
    """Apply already frozen gates to known regressions, with no fitting."""
    from key_disagreement import analyze
    from key_fusion import key_identity,read_raw_lab
    model=load_model(artifact)
    if model is None:raise ValueError('Freeze the resolver before reading regression results')
    manifest=Path(manifest).resolve();output=Path(output)
    evidence_dir=output.parent/(output.stem+'_evidence');evidence_dir.mkdir(parents=True,exist_ok=True)
    cases=[];inputs=[]
    for i,row in enumerate(csv.DictReader(manifest.open(encoding='utf-8-sig'))):
        paths={name:(manifest.parent/row[name]).resolve() for name in ['cnn','neutral','raw']}
        cnn=json.loads(paths['cnn'].read_text(encoding='utf-8'));neutral=json.loads(paths['neutral'].read_text(encoding='utf-8'))
        result=analyze(cnn,neutral,read_raw_lab(paths['raw']),model=model)
        (evidence_dir/f'{i:03d}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        cases.append(dict(id=row['id'],dataset='known_regression',genre='unknown',reference=key_identity(row['reference']),evidence=result['evidence']))
        inputs.append(dict(id=row['id'],files={name:signature(path) for name,path in paths.items()},audio=signature(manifest.parent/row['audio']) if row.get('audio') else None))
    report=dict(schema='key_disagreement_eval_v2',phase='known_regression',artifact=signature(artifact),manifest=signature(manifest),
        resolver=evaluate_cases(cases,model),rule=evaluate_cases(cases,rule=model['pure_rule_baseline']),agreement_only=evaluate_cases(cases),
        inputs=inputs,acoustic_predictions=True,new_acoustic_inference=False,new_training=False,test_used_for_selection=False)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');export_tables(report,output)
    print({k:{n:v for n,v in report[k].items() if n not in {'rows','confusion_matrix'}} for k in ['resolver','rule','agreement_only']})
    return report


def evaluate(split,cases,artifact,output,phase='sealed_test'):
    model=load_model(artifact)
    if model is None:raise ValueError('Missing calibrated artifact')
    records,missing=read_cases(split,cases,{phase})
    if phase=='sealed_test':
        forbidden=set(model['training_ids'])|set(model['calibration_ids'])
        if forbidden&{r['id'] for r in records}:raise ValueError('Sealed IDs seen during calibration')
        if signature(split)['sha256']!=model['split']['sha256']:raise ValueError('Split changed after freezing')
    report=dict(schema='key_disagreement_eval_v2',phase=phase,artifact=signature(artifact),split=signature(split),
        resolver=evaluate_cases(records,model),rule=evaluate_cases(records,rule=model['pure_rule_baseline']),
        agreement_only=evaluate_cases(records),missing_inputs=missing,
        by_genre={g:evaluate_cases([r for r in records if r.get('genre','unknown')==g],model) for g in sorted({r.get('genre','unknown') for r in records})},
        acoustic_predictions=True,new_training=False,test_used_for_selection=False,
        uncertainty='Small pilot; cutoffs calibrated but no guarantee of probability calibration or generalization')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    export_tables(report,output)
    print({k:{n:v for n,v in report[k].items() if n not in {'rows','confusion_matrix'}} for k in ['resolver','rule','agreement_only']})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',type=Path);p.add_argument('--cases',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['development','calibration','sealed_test'],default='sealed_test')
    a=p.parse_args()
    if a.manifest:
        if a.split or a.cases:p.error('Use either a regression manifest or an external split/cases pair')
        regression(a.manifest,a.model,a.output)
    else:
        if not a.split or not a.cases:p.error('--split and --cases are required for external evaluation')
        evaluate(a.split,a.cases,a.model,a.output,a.phase)
