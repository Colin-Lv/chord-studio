"""Independent fold/preprocessing/identity audit and delivery for the one run."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import json
import sys
import numpy as np
from app.web.storage import ROOT,atomic_json,read_json,signature
from app.key_top3_reranker_v1 import CODE_FILES,load_model,rank,target_labels

OUT=ROOT/'benchmarks/reports/key_top3_rerank_20261006_v1'


def main():
    protocol=read_json(OUT/'protocol.json');results=read_json(OUT/'results.json');manifest=read_json(OUT/'manifest.json');folds=read_json(OUT/'folds.json')
    assert datetime.fromisoformat(protocol['created_at'])<datetime.fromisoformat(results['first_fit_at'])
    lookup={r['id']:r for r in manifest['records']};checked_models=0;inner_checked=0;selection_checked=0
    for scope,plan in folds.items():
        oof=read_json(OUT/(scope+'_oof.json'))['rows'];assert len(oof)==plan['evaluation_songs']
        assert len({r['id'] for r in oof})==len(oof)
        for fold in plan['outer']:
            assert not set(fold['train_groups'])&set(fold['evaluation_groups'])
            assert not set(plan['fixed_training_groups'])&set(fold['evaluation_groups'])
            for method in ('M1','M2'):
                model=load_model(OUT/'models'/scope/(method+'_fold'+str(fold['fold'])+'.json'))
                assert model['training_ids']==fold['train_ids']==model['preprocessing_fit_ids']
                assert not set(model['training_groups'])&set(fold['evaluation_groups'])
                X=np.asarray([f[method] for rid in fold['train_ids'] for f in lookup[rid]['features']],dtype=float);X=np.nan_to_num(X,nan=0.)
                np.testing.assert_allclose(model['mean'],X.mean(axis=0),atol=1e-12)
                scale=X.std(axis=0);scale[scale==0]=1.
                np.testing.assert_allclose(model['scale'],scale,atol=1e-12)
                for row in [r for r in oof if r['fold']==fold['fold']]:
                    assert row[method]==rank(model,lookup[row['id']]['features'])[0]['key']
                    assert not lookup[row['id']]['group'] in model['training_groups']
                checked_models+=1
        for search in results['scopes'][scope]['inner_search']:
            best=max(search['search'],key=lambda x:(x['full_hits'],-x['harmful_changes'],-x['C']))
            assert search['selected_C']==best['C'];selection_checked+=1
            outer=plan['outer'][search['fold']]
            for C in search['search']:
                for inner in C['inner']:
                    expected=outer['inner'][inner['fold']]
                    assert inner['train_ids']==expected['train_ids']==inner['preprocessing_fit_ids']
                    assert inner['validation_ids']==expected['evaluation_ids']
                    assert not {lookup[i]['group'] for i in inner['train_ids']}&{lookup[i]['group'] for i in inner['validation_ids']}
                    inner_checked+=1
        changes=[]
        for row in oof:
            for method in ('M1','M2'):
                old,new=row['M0'],row[method]
                if old!=new:changes.append(dict(id=row['id'],group=row['group'],fold=row['fold'],method=method,reference=row['reference'],old=old,new=new,
                    outcome='corrected' if new==row['reference'] else 'harmful' if old==row['reference'] else 'changed_wrong_to_wrong',
                    original_sources_agree=row['original_sources_agree'],candidate_miss=row['candidate_miss']))
        with (OUT/(scope+'_changes.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
    misses=[r for r in lookup.values() if r['candidate_miss']]
    assert len(misses)==8 and all(target_labels([f['key'] for f in r['features']],r['reference'])==[0,0,0] for r in misses)
    unavailable=[r['id'] for r in lookup.values() if not r['harmonic_available']]
    assert len(unavailable)==3
    assert all(all(v==0 for v in lookup[i]['features'][0]['M2'][-13:]) for i in unavailable)
    regression=read_json(OUT/'xuni_regression.json')
    for method in ('M1','M2'):
        cli=read_json(OUT/('xuni_'+method+'_cli.json'))
        assert cli['ranking']==regression['models'][method]['ranking']
    protected=read_json(OUT/'protected_before.json');changed=[name for name,digest in protected.items() if signature(ROOT/name)['sha256']!=digest]
    assert not changed,changed
    audit=dict(protocol_frozen_before_first_fit=True,outer_models_checked=checked_models,inner_fit_identities_checked=inner_checked,
        selections_checked=selection_checked,outer_group_overlap=0,inner_group_overlap=0,preprocessing_means_scales_recomputed=True,
        fold_predictions_recomputed_from_saved_models=True,candidate_misses_retained=len(misses),missing_harmonic_retained=unavailable,
        cli_regression_equal=True,protected_files=len(protected),protected_changes=changed)
    atomic_json(OUT/'verification.json',audit)
    core={name:signature(ROOT/name) for name in CODE_FILES}
    commands=[
        [sys.executable,'-B','-m','pytest','tests/test_key_top3_reranker_v1.py','-q'],
        [sys.executable,'-B','-m','app.train_key_top3_reranker_v1','prepare'],
        results['actual_argv'],
        *[[sys.executable,'-B','-m','app.key_top3_reranker_v1','--model',str(OUT/'models'/(m+'_full.json')),
          '--cnn',regression['inputs']['cnn']['path'],'--harmonic',read_json(ROOT/'benchmarks/reports/harmonic_key_v2_20261006/regressions/xuni/report.json')['rows'][0]['prediction_file']['path'],
          '--output',str(OUT/('xuni_'+m+'_cli.json'))] for m in ('M1','M2')],
        [sys.executable,'-B','-m','app.finish_key_top3_rerank_v1']]
    atomic_json(OUT/'commands.json',dict(actual_successful_argv=commands,numeric_threads=2,
        initial_preparation='An unused 324-song draft rejected three empty harmonic outputs; corrected before any real fit. Original draft preserved at ../key_top3_rerank_20261006; final protocol prepared separately with all 327.',
        rerun='Entrypoints refuse to overwrite protocol/results. For reproduction use a fresh --output directory with prepare then run. No extra model/grid executed after this run.'))
    summary={scope:dict(n=s['methods']['M0']['overall']['n'],models={m:s['methods'][m] for m in ('M0','M1','M2')},
        comparisons={k:dict(corrected=len(v['corrected']),harmful=len(v['harmful']),net=v['net_correct']) for k,v in s['comparisons'].items()},
        bootstrap=s['bootstrap'],upstream_rule_exposed_evaluation_songs=s['upstream_reference_exposure_evaluated']) for scope,s in results['scopes'].items()}
    for scope,plan in folds.items():
        summary[scope]['per_fold_M2_minus_M1']=[results['scopes'][scope]['by_fold'][str(f['fold'])]['comparisons']['M1->M2']['net_correct'] for f in plan['outer']]
    models={p.relative_to(ROOT).as_posix():signature(p) for p in (OUT/'models').rglob('*.json')}
    files={p.relative_to(ROOT).as_posix():signature(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}
    atomic_json(OUT/'delivery.json',dict(schema='key_top3_rerank_development_delivery_v1',
        objective='One fixed-top3 candidate LR ranking experiment; M0 vs M1 vs M2',
        changed_files={**core,'app/finish_key_top3_rerank_v1.py':signature(__file__),'tests/test_key_top3_reranker_v1.py':signature(ROOT/'tests/test_key_top3_reranker_v1.py'),
                       'PROJECT_REPORT.md':signature(ROOT/'PROJECT_REPORT.md'),'README_chord_recognizer.md':signature(ROOT/'README_chord_recognizer.md')},
        artifacts=files,models=models,input_manifest=protocol['manifest'],fold_manifest=protocol['folds'],feature_schema=protocol['feature_schema'],protocol=signature(OUT/'protocol.json'),
        available_songs=327,groups=275,excluded=manifest['excluded'],retrospective_role=True,
        annotation_coverage=dict(complete_accepted_family_reference=327,major=35,minor=292,harmonic_available=324,harmonic_unavailable=3,
          label_confidence=dict(Counter(r['annotation_confidence'] for r in lookup.values())),original_subtype_retained=True),
        source_and_boundary='Same full excerpt, unchanged original CNN 24 scores/raw LAB/measured grid; fixed K=3 for every song. No boundary or acoustic re-evaluation.',
        candidate_diagnostic=dict(covered=319,total=327,missing_ids=[r['id'] for r in misses],reference_injection=False),
        real_fit_count=results['fits'],model_grid=protocol['grid'],tests=dict(passed=21,failed=0,skipped=0,historical_256_rerun=False,historical_10_predictions_replayed=False),
        verification=audit,metrics=summary,xuni=dict(candidates=regression['cnn_top3'],M0=regression['M0'],M1=results['xuni']['M1'],M2=results['xuni']['M2'],
             known_not_in_training_or_selection=True,automatic_override=False),
        upstream_exposure='All-retired outer evaluation includes 86 prior harmonic weight-selection songs. Stricter 241 songs/199 groups exclude all 76 connected selection groups from both outer and inner evaluation; those 86 remain training-only. K=3 design came from known audit; upstream CNN pretraining overlap unknown.',
        results_kind='New-layer group-OOF retrospective development conditional on fixed historical engine; never a new blind end-to-end test',
        resources=dict(elapsed_seconds=results['elapsed_seconds'],cpu_seconds=results['cpu_seconds'],memory_bytes=results['memory_bytes'],
            memory_scope='OS-recorded training process working-set peak, not machine physical-memory peak; prepare/hash checks outside timed training phase',numeric_threads=2),
        failures_and_warnings=dict(initial_draft_replaced_before_fit=True,training_warnings=results['warnings'],convergence_failures=0),
        new_acoustic_inference=0,new_frozen_symbolic_analysis=0,new_downloads=0,new_cnn_hgb_training=0,new_lr_training=True,
        deployment=False,automatic_override_enabled=False,frontend_changed=False,old_thresholds_changed=False,
        conclusions=['Ranking shows exploratory net gains over CNN on both grouped scopes.',
           'M2 exceeds M1 by only 3/4 correct songs; paired 95% intervals include zero; additional harmonic value is not stable.',
           'M1 major-family hits drop 20->18 and 16->14; stricter M2 major hits drop 16->15. Minority class regressions cannot be hidden.',
           'On all-retired source-agreement songs M2 does not improve baseline and underperforms M1; no hard agreement shortcut.',
           'Xuni remains G under both final models; no single-song tuning.',
           'Retain this narrow offline result and stop; next evidence needs target-domain labels and independently isolated evaluation, not another rules/grid round.'],
        unverified=['New target-pop/folk generalization, representative calibration, deployment thresholds and risk budget',
                    'Complete upstream pretraining isolation and alias/unlisted-recording version identity',
                    'Chord/bass/boundary/song-section accuracy; no new acoustic inference'],
        method_sources=['https://scikit-learn.org/stable/modules/cross_validation.html','https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
                        'https://scikit-learn.org/stable/common_pitfalls.html','https://scikit-learn.org/stable/modules/calibration.html']))
    print(dict(verified=audit,summary={k:v['comparisons'] for k,v in summary.items()}))


if __name__=='__main__':main()
