"""Prepare/freeze and run one retrospective nested-group top3 LR experiment."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import warnings
import numpy as np
from app.web.storage import ROOT,atomic_json,read_json,signature
from app.key_top3_reranker_v1 import (SCHEMA,FEATURES,CODE_FILES,NOTES,PC,canonical,candidates,feature_rows,
                                    target_labels,fit_model,rank,load_model,decision_scores)

FROZEN=ROOT/'benchmarks/reports/harmonic_key_v2_20261006'
REVIEW=ROOT/'benchmarks/reports/key_review_data_20261006'
DEFAULT_OUTPUT=ROOT/'benchmarks/reports/key_top3_rerank_20261006_v1'
SEED=2026100603
GRID=[.1,1.,10.]


def stamp():return datetime.now(timezone.utc).isoformat()
def digest_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def identity(path):return dict(path=str(Path(path).resolve()),**signature(path))
def verify(value):
    if signature(value['path'])['sha256']!=value['sha256']:raise ValueError('Input hash changed: '+value['path'])
def safe_features(rows):
    return [{**r,'M2':[float(v) if math.isfinite(v) else None for v in r['M2']]} for r in rows]


def connect_groups(metadata,roles):
    parents={r['id']:r['id'] for r in metadata};seen={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for r in metadata:
        rid=r['id'];tokens=[str(t).strip().casefold() for t in r['group_tokens']]+['historical:'+r['group']]
        if rid in roles:tokens+=['audio:'+roles[rid]['audio_sha256'],'role_component:'+roles[rid]['group_id']]
        for token in tokens:
            if token in seen:parents[find(rid)]=find(seen[token])
            else:seen[token]=rid
    components=defaultdict(list)
    for r in metadata:components[find(r['id'])].append(r['id'])
    group={rid:hashlib.sha256('|'.join(sorted(ids)).encode()).hexdigest() for ids in components.values() for rid in ids}
    return group,list(components.values())


def freeze_folds(records,desired,fixed_groups=()):
    fixed=set(fixed_groups);pool=[r for r in records if r['group'] not in fixed];by_group=defaultdict(list)
    for r in pool:by_group[r['group']].append(r['id'])
    for count in range(min(desired,len(by_group)),1,-1):
        bins=[[] for _ in range(count)];sizes=[0]*count
        ordered=sorted(by_group,key=lambda g:(-len(by_group[g]),hashlib.sha256((str(SEED)+g).encode()).hexdigest()))
        for group in ordered:
            index=min(range(count),key=lambda i:(sizes[i],i));bins[index].append(group);sizes[index]+=len(by_group[group])
        folds=[];valid=True
        for i,groups in enumerate(bins):
            test=[r for r in records if r['group'] in set(groups)];train=[r for r in records if r['group'] not in set(groups)]
            ys=[v for r in train for v in target_labels([f['key'] for f in r['features']],r['reference'])]
            if not test or set(ys)!={0,1}:valid=False;break
            folds.append(dict(fold=i,train_ids=[r['id'] for r in train],evaluation_ids=[r['id'] for r in test],
                              train_groups=sorted({r['group'] for r in train}),evaluation_groups=sorted(set(groups))))
        if valid:return folds
    raise ValueError('Fewer than two usable non-overlapping grouped folds with binary supervision')


def prepare(output):
    if output.exists():raise ValueError('Refusing to overwrite experiment; use a new output directory')
    contract=read_json(FROZEN/'engine_contract.json')
    for name,digest in contract['implementation'].items():
        if signature(ROOT/'app'/name)['sha256']!=digest:raise ValueError('Frozen engine changed: '+name)
    # Read actual complete prior deliverables rather than trusting summary numbers.
    previous=read_json(REVIEW/'delivery.json');audit=read_json(REVIEW/'candidate_audit.json');inventory=read_json(REVIEW/'data_inventory.json')
    if audit['counts']!={'both_correct':15,'harmonic_only':2,'cnn_only':11,'both_wrong':5}:raise ValueError('Prior audit inconsistent')
    roles_all=[json.loads(line) for line in (REVIEW/'data_roles.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    roles={r['item_id'].removeprefix('giantsteps_'):r for r in roles_all if r['item_id'].startswith('giantsteps_') and r['data_role']=='retired_known_development'}
    split=read_json(FROZEN/'split.json');metadata={r['id']:r for r in split['rows']};groups,components=connect_groups(split['rows'],roles)
    source_reports=[FROZEN/'regressions/giantsteps180/report.json',*[FROZEN/(phase+'.json') for phase in ['harmonic_dev','harmonic_calibration','harmonic_new_sealed']]]
    cached={}
    for path in source_reports:
        for r in read_json(path)['rows']:
            if r['id'] in cached:raise ValueError('Duplicate frozen prediction ID')
            cached[r['id']]=(r,path)
    records=[];excluded=[]
    for rid,role in sorted(roles.items()):
        try:
            if role['annotation_status']!='historical_official_family_label' or role.get('tonic') is None or role.get('mode') not in ('major','minor'):raise ValueError('No credible single complete reference')
            ref=canonical(role['tonic']+('m' if role['mode']=='minor' else ''))
            r,report=cached[rid];meta=metadata[rid]
            if ref!=canonical(meta['reference']) or ref!=canonical(r['reference']):raise ValueError('Conflicting reference identities')
            audio=r['inputs'].get('audio') or r['audio'];verify(audio)
            if audio['sha256']!=role['audio_sha256']:raise ValueError('Role audio identity differs')
            required={name:r['inputs'][name] for name in ['raw','cnn','neutral']}
            for v in required.values():verify(v)
            raw_meta_path=Path(required['raw']['path']+'.provenance.json');raw_meta=read_json(raw_meta_path)
            if raw_meta['audio']['sha256']!=audio['sha256'] or raw_meta['lab']['sha256']!=required['raw']['sha256']:raise ValueError('Raw/audio binding differs')
            neutral=read_json(required['neutral']['path'])
            if raw_meta.get('offset',0)!=0 or abs(raw_meta['duration']-neutral['duration'])>1/22050+1e-6:raise ValueError('Raw/neutral time support differs')
            cnn=read_json(required['cnn']['path']);vector=candidates(cnn);p=r['prediction']
            if p.get('key') and p['policy']!=contract['policy']:raise ValueError('Frozen harmonic policy differs')
            if r.get('prediction_file'):
                verify(r['prediction_file'])
                if read_json(r['prediction_file']['path'])!=p:raise ValueError('Embedded harmonic evidence differs from artifact')
            if canonical(r['cnn'])!=vector[0]['key']:raise ValueError('Historical CNN result differs')
            features=feature_rows(cnn,p)
            records.append(dict(id=rid,group=groups[rid],reference=ref,original_mode_label=meta['original_key'],annotation_confidence=meta['annotation_confidence'],
                annotation_source=role['source'],domain=role['domain'],historical_split=role['historical_split'],
                role_history=role['role_history']+[dict(at=stamp(),role='retrospective_development_top3_v1',training_in_this_round=True)],
                audio_sha256=audio['sha256'],audio=audio,inputs=required,raw_provenance=identity(raw_meta_path),
                evidence_report=identity(report),cnn_vector=vector,harmonic_key=canonical(p['key']) if p.get('key') else None,
                features=safe_features(features),candidate_miss=ref not in [f['key'] for f in features],
                harmonic_available=bool(p.get('key')),
                upstream_rule_selection_member=rid in contract['development_ids']))
        except (OSError,ValueError,KeyError,TypeError) as exc:excluded.append(dict(id=rid,reason=str(exc)))
    # Duplicate recordings cannot masquerade as separate songs or cross folds.
    unique={};conflicts=set()
    for r in records:
        sha=r['audio_sha256']
        if sha in unique:
            if unique[sha]['reference']!=r['reference']:conflicts.add(sha)
            excluded.append(dict(id=r['id'],reason='duplicate_audio_sha',retained=unique[sha]['id']))
        else:unique[sha]=r
    records=[r for sha,r in unique.items() if sha not in conflicts]
    excluded.extend(dict(id=r['id'],reason='conflicting_duplicate_audio_labels') for r in unique.values() if r['audio_sha256'] in conflicts)
    if not records:raise ValueError('No actual usable materials: '+repr(excluded))
    upstream_groups={groups[rid] for rid in contract['development_ids']}
    scopes={}
    for name,fixed in [('all_retired_grouped',set()),('upstream_selection_groups_training_only',upstream_groups)]:
        outer=freeze_folds(records,5,fixed)
        lookup={r['id']:r for r in records}
        for fold in outer:
            train=[lookup[i] for i in fold['train_ids']];fold['inner']=freeze_folds(train,3,fixed)
            if set(fold['train_groups']) & set(fold['evaluation_groups']):raise ValueError('Outer group overlap')
            if fixed & set(fold['evaluation_groups']):raise ValueError('Upstream selection group in stricter evaluation')
        scopes[name]=dict(fixed_training_groups=sorted(fixed),outer=outer,evaluation_songs=sum(len(f['evaluation_ids']) for f in outer))
    output.mkdir(parents=True)
    protected=read_json(REVIEW/'protected_before.json')
    for path in previous['changed_files']:
        if path not in ('PROJECT_REPORT.md','README_chord_recognizer.md'):protected[path]=signature(ROOT/path)['sha256']
    protected.update({name:signature(ROOT/name)['sha256'] for name in ['app/web/worker.py','app/analyze_with_madmom.py','app/key_bridge.py']})
    cnn_weight=Path('C:/Users/33573/.codex/acr_envs/madmom39/Lib/site-packages/madmom/models/key/2018/key_cnn.pkl')
    if signature(cnn_weight)['sha256']!=contract.get('cnn_weight_sha256',previous['protected_verification']['cnn_weight']['sha256']):raise ValueError('CNN weight changed')
    protected[str(cnn_weight)]=signature(cnn_weight)['sha256']
    for path in (ROOT/'projects/web').rglob('*.json'):
        if '/jobs/' not in path.as_posix():protected[path.relative_to(ROOT).as_posix()]=signature(path)['sha256']
    atomic_json(output/'protected_before.json',protected)
    shutil.copyfile(ROOT/'PROJECT_REPORT.md',output/'previous_PROJECT_REPORT.md')
    manifest=dict(schema='retrospective_key_top3_manifest_v1',records=records,excluded=excluded,
                  registered=len(roles),included=len(records),groups=len({r['group'] for r in records}),
                  grouping='Transitive all-368 eligible metadata tokens/components + role components + audio SHA; no label-driven partition',
                  all_component_members=components,upstream_selection_ids=contract['development_ids'],upstream_selection_groups=sorted(upstream_groups),
                  sources=[identity(p) for p in source_reports]+[identity(REVIEW/'data_roles.jsonl'),identity(FROZEN/'split.json')],
                  prior_review_inventory=inventory['items'],new_independent_calibration=0,new_independent_test=0)
    atomic_json(output/'manifest.json',manifest);atomic_json(output/'folds.json',scopes)
    protocol=dict(schema='key_top3_nested_protocol_v1',created_at=stamp(),fitting_started=False,seed=SEED,K=3,models=['M0','M1','M2'],
        implementation={name:signature(ROOT/name)['sha256'] for name in CODE_FILES},harmonic_contract=identity(FROZEN/'engine_contract.json'),
        dependencies=dict(python=sys.version,numpy=np.__version__,sklearn=__import__('sklearn').__version__),
        features=FEATURES,feature_schema=SCHEMA,grid=GRID,solver='L2 logistic regression, lbfgs, max_iter=2000, tol=1e-8, class_weight=None',
        per_candidate_weight=1/3,per_song_total_weight=1,preprocessing='Fold-local constant-0 imputer, StandardScaler; validity masks for all 13 harmonic fields',
        feature_view='Frozen global centre/mode scores minus existing loop contribution; tonic maximum excludes local share. No local/Viterbi output, loop term, chroma or joint input. Derived competition ranks share ties.',
        interactions=[],selection='Inner full-key hits, then fewer harmful changes versus CNN, then smaller C; all validation songs including candidate misses',
        known_design_exposure='K=3 and this research question were proposed from the known 33-song candidate audit; not a blind prespecified choice',
        final_model_C='Most frequent inner-selected C across the five all-retired outer folds; ties choose smaller C; no OOF-score choice',
        fold_policy='Label-blind group-size balancing, hash order seed; outer downgrade 5/4/3/2, inner 3/2 only if groups or binary training labels insufficient',
        candidate_ties='Exact CNN score ties retain original vector order; exact LR score ties prefer CNN higher rank',
        scopes=list(scopes),confidence_intervals=dict(method='paired component bootstrap, pooled song accuracy difference',draws=4000,seed=SEED),
        threshold_selection=False,automatic_override_enabled=False,upstream_exposure='Historical harmonic parameters selected using 86 development references; all groups evaluation is only a new-layer OOF conditional on historical engine. Stricter scope excludes those connected groups from evaluation; all remain known retrospective materials.',
        manifest=identity(output/'manifest.json'),folds=identity(output/'folds.json'),protected=identity(output/'protected_before.json'))
    atomic_json(output/'protocol.json',protocol)
    print(dict(prepared=len(records),groups=manifest['groups'],excluded=len(excluded),candidate_misses=sum(r['candidate_miss'] for r in records),
               scopes={k:v['evaluation_songs'] for k,v in scopes.items()},fit_started=False),flush=True)


def relation(ref,pred):
    if ref==pred:return 'exact'
    a,b=PC[ref.removesuffix('m')],PC[pred.removesuffix('m')];am=ref.endswith('m');bm=pred.endswith('m')
    if a==b:return 'parallel_major_minor'
    if am!=bm and ((am and (b-a)%12==3) or (not am and (a-b)%12==3)):return 'relative_major_minor'
    if am==bm and (b-a)%12 in (5,7):return 'perfect_fifth'
    return 'other'


def metrics(rows,method):
    n=len(rows);correct=sum(r[method]==r['reference'] for r in rows)
    tonic=sum(r[method].removesuffix('m')==r['reference'].removesuffix('m') for r in rows)
    mode=sum(r[method].endswith('m')==r['reference'].endswith('m') for r in rows)
    return dict(n=n,full_correct=correct,full_accuracy=correct/n if n else None,tonic_correct=tonic,tonic_accuracy=tonic/n if n else None,
                mode_correct=mode,mode_accuracy=mode/n if n else None,mode_given_correct_tonic=correct/tonic if tonic else None,
                error_relations=dict(Counter(relation(r['reference'],r[method]) for r in rows)))


def compare(rows,baseline,method):
    corrected=[r['id'] for r in rows if r[baseline]!=r['reference'] and r[method]==r['reference']]
    harmful=[r['id'] for r in rows if r[baseline]==r['reference'] and r[method]!=r['reference']]
    return dict(corrected=corrected,harmful=harmful,net_correct=len(corrected)-len(harmful),
        both_correct=sum(r[baseline]==r['reference'] and r[method]==r['reference'] for r in rows),
        both_wrong=sum(r[baseline]!=r['reference'] and r[method]!=r['reference'] for r in rows),
        changed_wrong_to_wrong=[r['id'] for r in rows if r[baseline]!=r['reference'] and r[method]!=r['reference'] and r[baseline]!=r[method]])


def summary(rows):
    return dict(methods={m:dict(overall=metrics(rows,m),major=metrics([r for r in rows if not r['reference'].endswith('m')],m),
              minor=metrics([r for r in rows if r['reference'].endswith('m')],m)) for m in ['M0','M1','M2']},
              comparisons={base+'->'+method:compare(rows,base,method) for base,method in [('M0','M1'),('M0','M2'),('M1','M2')]},
              candidate_misses=[r['id'] for r in rows if r['candidate_miss']],candidate_coverage=sum(not r['candidate_miss'] for r in rows)/len(rows) if rows else None)


def interval(rows,baseline,method):
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(r)
    values=list(groups.values());n=np.asarray([len(g) for g in values]);delta=np.asarray([sum(int(r[method]==r['reference'])-int(r[baseline]==r['reference']) for r in g) for g in values])
    rng=np.random.default_rng(SEED);draws=rng.integers(0,len(values),(4000,len(values)))
    samples=delta[draws].sum(axis=1)/n[draws].sum(axis=1)
    return dict(difference=float(delta.sum()/n.sum()),percentile_95=np.quantile(samples,[.025,.975]).tolist(),groups=len(values),draws=4000,
                scope='Paired retrospective group bootstrap; overlapping CV training ignored, not deployment risk guarantee')


def predict(model,records):
    return {r['id']:rank(model,r['features']) for r in records}


def choose_C(train,folds,method,lookup):
    search=[]
    for C in GRID:
        hits=harmful=0;audits=[]
        for fold in folds:
            fitted=fit_model([lookup[i] for i in fold['train_ids']],method,C)
            validation=[lookup[i] for i in fold['evaluation_ids']];predictions=predict(fitted,validation)
            if set(fitted['training_groups']) & set(fold['evaluation_groups']):raise ValueError('Inner group leakage')
            correct=sum(predictions[r['id']][0]['key']==r['reference'] for r in validation)
            bad=sum(r['features'][0]['key']==r['reference'] and predictions[r['id']][0]['key']!=r['reference'] for r in validation)
            hits+=correct;harmful+=bad
            audits.append(dict(fold=fold['fold'],train_ids=fitted['training_ids'],preprocessing_fit_ids=fitted['preprocessing_fit_ids'],
                validation_ids=fold['evaluation_ids'],correct=correct,harmful=bad,iterations=fitted['iterations']))
        search.append(dict(C=C,full_hits=hits,harmful_changes=harmful,inner=audits))
    best=max(search,key=lambda r:(r['full_hits'],-r['harmful_changes'],-r['C']))
    return best['C'],search


def run(output):
    if (output/'results.json').exists():raise ValueError('Refusing to overwrite completed experiment')
    protocol=read_json(output/'protocol.json');began=stamp();start=time.perf_counter();cpu=time.process_time()
    for v in (protocol['manifest'],protocol['folds'],protocol['protected'],protocol['harmonic_contract']):verify(v)
    for name,digest in protocol['implementation'].items():
        if signature(ROOT/name)['sha256']!=digest:raise ValueError('New implementation changed after protocol freeze')
    manifest=read_json(output/'manifest.json');records=manifest['records'];lookup={r['id']:r for r in records};folds=read_json(output/'folds.json')
    # All referenced input identities are checked before the first real fit.
    for r in records:
        verify(r['audio']);[verify(v) for v in r['inputs'].values()];verify(r['raw_provenance']);verify(r['evidence_report'])
    captured=[];results={};all_rows={};selected=defaultdict(list);fit_count=0
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        for name,scope in folds.items():
            oof=[];selection=[]
            for fold in scope['outer']:
                train=[lookup[i] for i in fold['train_ids']];validation=[lookup[i] for i in fold['evaluation_ids']];predictions={};model_paths={}
                for method in ['M1','M2']:
                    C,search=choose_C(train,fold['inner'],method,lookup);fit_count+=len(GRID)*len(fold['inner'])
                    if name=='all_retired_grouped':selected[method].append(C)
                    model=fit_model(train,method,C);fit_count+=1
                    model.update(implementation=protocol['implementation'],harmonic_contract=protocol['harmonic_contract'],
                        experiment_protocol=identity(output/'protocol.json'),manifest=protocol['manifest'],folds=protocol['folds'],dependencies=protocol['dependencies'],
                        purpose='new-layer grouped retrospective development only',scope=name,outer_fold=fold['fold'])
                    model_path=output/'models'/name/(method+'_fold'+str(fold['fold'])+'.json');atomic_json(model_path,model)
                    reloaded=load_model(model_path);predictions[method]=predict(reloaded,validation);model_paths[method]=identity(model_path)
                    if predictions[method]!=predict(model,validation):raise ValueError('Save/load changed rankings')
                    selection.append(dict(fold=fold['fold'],method=method,selected_C=C,search=search))
                for r in validation:
                    oof.append(dict(id=r['id'],group=r['group'],reference=r['reference'],original_mode_label=r['original_mode_label'],annotation_confidence=r['annotation_confidence'],
                        candidate_miss=r['candidate_miss'],candidates=[f['key'] for f in r['features']],M0=r['features'][0]['key'],
                        M1=predictions['M1'][r['id']][0]['key'],M2=predictions['M2'][r['id']][0]['key'],
                        rankings={m:predictions[m][r['id']] for m in ['M1','M2']},fold=fold['fold'],prediction_kind='outer_group_held_out',scope=name,
                        historical_split=r['historical_split'],models=model_paths,training_groups=fold['train_groups'],
                        upstream_rule_selection_group=r['group'] in set(manifest['upstream_selection_groups']),
                        original_sources_agree=r['harmonic_key']==r['features'][0]['key'],
                        original_agreement_both_wrong=r['harmonic_key']==r['features'][0]['key'] and r['features'][0]['key']!=r['reference']))
                print(name+' fold '+str(fold['fold'])+': '+str({m:metrics(oof[-len(validation):],m)['full_correct'] for m in ['M0','M1','M2']}),flush=True)
            if len({r['id'] for r in oof})!=len(oof):raise ValueError('Repeated evaluation song')
            all_rows[name]=oof
            report=summary(oof)
            report.update(by_fold={str(f['fold']):summary([r for r in oof if r['fold']==f['fold']]) for f in scope['outer']},
                by_group={g:summary([r for r in oof if r['group']==g]) for g in sorted({r['group'] for r in oof})},
                by_original_agreement={s:summary([r for r in oof if ('agree' if r['original_sources_agree'] else 'disagree')==s]) for s in ['agree','disagree']},
                agreement_both_wrong=summary([r for r in oof if r['original_agreement_both_wrong']]),
                by_original_error_relation={kind:summary([r for r in oof if relation(r['reference'],r['M0'])==kind]) for kind in ['exact','relative_major_minor','parallel_major_minor','perfect_fifth','other']},
                bootstrap={a+'->'+b:interval(oof,a,b) for a,b in [('M0','M1'),('M0','M2'),('M1','M2')]},
                upstream_reference_exposure_evaluated=sum(r['upstream_rule_selection_group'] for r in oof),
                inner_search=selection)
            results[name]=report
            atomic_json(output/(name+'_oof.json'),dict(rows=oof,purpose='fixed historical upstream engine conditional evaluation'))
            with (output/(name+'_oof.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['id','group','reference','original_mode_label','annotation_confidence','fold','M0','M1','M2','candidate_miss','original_sources_agree','upstream_rule_selection_group','prediction_kind'],extrasaction='ignore');writer.writeheader();writer.writerows(oof)
        final_models={}
        for method in ['M1','M2']:
            counts=Counter(selected[method]);C=min(counts,key=lambda c:(-counts[c],c));model=fit_model(records,method,C);fit_count+=1
            model.update(implementation=protocol['implementation'],harmonic_contract=protocol['harmonic_contract'],experiment_protocol=identity(output/'protocol.json'),
                manifest=protocol['manifest'],folds=protocol['folds'],dependencies=protocol['dependencies'],purpose='full retrospective development fit; known regression only',scope='full_fit',C_vote=dict(counts))
            path=output/'models'/(method+'_full.json');atomic_json(path,model);final_models[method]=identity(path)
        captured=[dict(category=n.category.__name__,message=str(n.message)) for n in notices]
    if any(n['category']=='ConvergenceWarning' for n in captured):raise ValueError('LR did not converge; no performance claim permitted')
    # Xuni only after ALL selection, never part of development folds.
    case=read_json(FROZEN/'regressions/xuni/report.json')['rows'][0]
    for v in case['inputs'].values():verify(v)
    features=feature_rows(read_json(case['inputs']['cnn']['path']),case['prediction']);xuni={}
    for method,v in final_models.items():
        model=load_model(v['path']);ranked=rank(model,features);X=np.asarray([r[method] for r in features]);X=np.where(np.isnan(X),0,X)
        parts=((X-np.asarray(model['mean']))/np.asarray(model['scale']))*np.asarray(model['coefficient'])
        xuni[method]=dict(key=ranked[0]['key'],ranking=ranked,contributions=[dict(key=r['key'],terms=dict(zip(model['features'],p.tolist())),intercept=model['intercept']) for r,p in zip(features,parts)])
    atomic_json(output/'xuni_regression.json',dict(M0=features[0]['key'],cnn_top3=[f['key'] for f in features],models=xuni,features=safe_features(features),
         original_harmonic_key=case['prediction']['key'],reference_joined_after_prediction='C (prior user-confirmed known case)',
         prediction_kind='known_regression_not_in_training_or_selection',inputs=case['inputs'],final_models=final_models,
         automatic_override_enabled=False,new_acoustic_inference=False))
    # The previously audited 33 are explicitly OOF in the new development role.
    known_ids={r['id'] for r in read_json(FROZEN/'harmonic_new_sealed.json')['rows']}
    atomic_json(output/'known33_new_layer_oof.json',{scope:dict(summary=summary([r for r in rows if r['id'] in known_ids]),rows=[r for r in rows if r['id'] in known_ids],
        purpose='known former-sealed retrospective rows; new reranker group-held-out, not new blind test') for scope,rows in all_rows.items()})
    protected=read_json(output/'protected_before.json');changes=[name for name,digest in protected.items() if signature(ROOT/name)['sha256']!=digest]
    if changes:raise ValueError('Protected default/model/human output changed: '+repr(changes))
    import psutil
    atomic_json(output/'results.json',dict(schema='key_top3_nested_results_v1',scopes=results,manifest=protocol['manifest'],protocol=identity(output/'protocol.json'),
        final_models=final_models,xuni={m:xuni[m]['key'] for m in xuni},fits=fit_count,warnings=captured,
        new_acoustic_inference=False,automatic_override_enabled=False,new_independent_test_labels=0,
        evaluation_kind='fixed historical engine conditional new-layer grouped retrospective evaluation',
        first_fit_at=began,completed_at=stamp(),elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        memory_bytes=psutil.Process().memory_info()._asdict(),protected_files_checked=len(protected),protected_changes=changes,
        actual_argv=[sys.executable,'-B','-m','app.train_key_top3_reranker_v1','run','--output',str(output)]))
    print(dict(fits=fit_count,xuni={m:xuni[m]['key'] for m in xuni},summary={name:{m:report['methods'][m]['overall']['full_correct'] for m in ['M0','M1','M2']} for name,report in results.items()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run']);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    a=p.parse_args();(prepare if a.phase=='prepare' else run)(a.output.resolve())


if __name__=='__main__':main()
