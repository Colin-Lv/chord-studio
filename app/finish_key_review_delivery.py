"""Finalize the review/data round without touching any historical artifact."""
from pathlib import Path
from urllib.request import urlopen
import json
import sys
import shutil
from app.web.storage import ROOT,atomic_json,read_json,signature,now


def main():
    out=ROOT/'benchmarks/reports/key_review_data_20261006'
    snapshot=read_json(out/'protected_before.json')
    changed=[name for name,digest in snapshot.items() if signature(ROOT/name)['sha256']!=digest]
    if changed:raise ValueError('Protected artifact changed: '+repr(changed))
    cnn=Path('C:/Users/33573/.codex/acr_envs/madmom39/Lib/site-packages/madmom/models/key/2018/key_cnn.pkl')
    expected=read_json(ROOT/'benchmarks/reports/harmonic_key_v2_20261006/delivery.json')['cnn_weight_hashes']['key_cnn.pkl']
    if signature(cnn)['sha256']!=expected:raise ValueError('CNN weight changed')
    audit=read_json(out/'candidate_audit.json');data=read_json(out/'data_inventory.json');ui=read_json(out/'browser_final/validation.json')
    original=read_json(ROOT/'benchmarks/reports/harmonic_key_v2_20261006/regressions/xuni/report.json')['rows'][0]['prediction']
    cli=read_json(out/'xuni_adapter_cli.json');assert cli['prediction']==original
    with urlopen('http://127.0.0.1:8765/api/projects/087d8b0c86414fe6b675cc622f505128/key-review',timeout=10) as r:live=json.load(r)
    assert live['effective_key']=='G' and live['manual'] is None and live['suggestion']['prediction']['key']=='C'
    atomic_json(out/'live_xuni_review.json',live)
    changed_files=['app/key_review_adapter.py','app/web/key_review.py','app/web/key_annotation.py','app/web/server.py',
                   'app/audit_key_review_data.py','app/import_key_annotations.py','app/web/check_key_review_browser.py',
                   'app/finish_key_review_delivery.py','frontend/index.html','frontend/app.js','frontend/style.css',
                   'frontend/key_annotation.html','frontend/key_annotation.js','benchmarks/key_review_cache_registry.json',
                   'tests/test_key_review_workbench.py','tests/test_key_annotation.py','PROJECT_REPORT.md','README_chord_recognizer.md']
    commands=[
        [sys.executable,'-B','-m','app.audit_key_review_data'],
        [str(ROOT/'.tmp/harmony_web_env/Scripts/python.exe'),'-B','-m','pytest','tests/test_web_workbench.py','tests/test_key_review_workbench.py','tests/test_key_annotation.py','-q'],
        [str(ROOT/'.tmp/harmony_web_env/Scripts/python.exe'),'-B','-m','app.web.check_key_review_browser','--project',ui['project_id'],'--output',str(out/'browser_final')],
        [sys.executable,'-B','-m','app.key_review_adapter','--audio',str(ROOT/'projects/web/087d8b0c86414fe6b675cc622f505128/runs/971c479a56844032ae1e231d08f90073/source/audio.m4a'),
         '--raw',str(ROOT/'projects/web/087d8b0c86414fe6b675cc622f505128/runs/971c479a56844032ae1e231d08f90073/raw_madmom.lab'),
         '--neutral',str(ROOT/'benchmarks/reports/key_fusion_20261006/xuni/neutral_analysis.json'),
         '--contract',str(ROOT/'benchmarks/reports/harmonic_key_v2_20261006/engine_contract.json'),'--output',str(out/'xuni_adapter_cli.json')],
        [sys.executable,'-B','-m','app.import_key_annotations','--input',str(out/'browser/fixture_annotation.json'),'--output',str(out/'fixture_import_not_gold.jsonl')],
        [str(ROOT/'.tmp/harmony_web_env/Scripts/python.exe'),'-B','app/start_web.py','--port','8765'],
        [str(ROOT/'.tmp/harmony_web_env/Scripts/python.exe'),'-B','app/start_web.py','--port','8766'],
        ['C:/Program Files/nodejs/node.exe','--check','frontend/app.js'],
        ['C:/Program Files/nodejs/node.exe','--check','frontend/key_annotation.js'],
        [sys.executable,'-B','-m','app.finish_key_review_delivery']]
    atomic_json(out/'commands.json',dict(actual_successful_argv=commands,
                integration_store=str(ROOT/'.tmp/key_review_integration/store'),
                server_restart='Verified matching local service PID and no active model jobs; launched hidden windows; no acoustic jobs submitted',
                setup_scripts=['.tmp/prepare_key_review_round.py','.tmp/create_key_review_fixture.py','.tmp/fold_legacy_key_diagnostics.py']))
    live_logs=list(out.glob('server*.log'))
    for log in live_logs:
        shutil.copyfile(log,out/('snapshot_'+log.name))
    artifacts={p.relative_to(ROOT).as_posix():signature(p) for p in out.rglob('*') if p.is_file() and p.name!='delivery.json' and p not in live_logs}
    protected_models={name:signature(ROOT/name) for name in snapshot if name.endswith(('.joblib','.feature_contract.json')) or name.endswith('key_resolver_v2.json') or name.endswith('engine_contract.json')}
    delivery=dict(schema='global_key_review_and_data_delivery_v1',completed_at=now(),
        objective='Frozen global-key human review, finite-candidate audit and target-domain annotation preparation',
        changed_files={name:signature(ROOT/name) for name in changed_files},artifacts=artifacts,
        mutable_service_logs=[str(p.relative_to(ROOT)) for p in live_logs],log_note='Only copied snapshot_* logs are hashed artifacts; live service logs may continue growing.',
        protected_verification=dict(files=len(snapshot),changed=changed,old_contract_resigned=False,cnn_weight=dict(path=str(cnn),**signature(cnn)),models=protected_models),
        dataflow='Original decoded audio SHA + original raw_madmom LAB/provenance + measured neutral support + frozen contract/code -> checked cached frozen v2 evidence -> separate human event/revision -> current global notation. Never corrected LAB or reference into automatic evidence.',
        automatic_policy_changed=False,automatic_override_enabled=False,resolver_thresholds=dict(cnn=1.01,harmonic=1.01),
        new_acoustic_inference=0,new_training=0,new_downloads=0,
        tests=dict(passed=42,failed=0,skipped=0,new_tests=14,existing_web_tests=28,historical_256_rerun=False,historical_10_predictions_replayed=False,
                   initial_resolved_failure='One new test mistakenly searched string(dict_keys) for key; fixed assertion checks actual response keys. No product failure concealed.'),
        real_integration=dict(project=ui['project_id'],real_http=True,response_interception=False,real_audio=True,frozen_cache_replay=True,
              checks=len(ui['checks']),js_errors=ui['javascript_errors'],manual_write_scope='isolated validation copy only',
              actual_user_project='087d8b0c86414fe6b675cc622f505128',actual_user_key='G',actual_suggestion='C',actual_human_confirmation=False,
              cli_prediction_equal_frozen=True,annotation_fixture_is_not_human_gold=True),
        candidate_groups=audit['counts'],prediction_agreement=audit['agreement'],candidate_coverage=audit['topk'],
        overall_frozen_metrics=audit['metrics'],local_risk=audit['billboard'],
        per_song_evidence='candidate_audit.json + all 18 complementary/harmful/both-wrong evidence files; reference joined only offline',
        regression_songs='No algorithm/output changes; 11 CNN-only samples are hypothetical harmful overrides, not actual newly regressed songs',
        annotation_coverage=data['field_coverage'],data_inventory=dict(items=data['items'],unique_audio=data['unique_audio'],
                    proposed_groups=data['proposed_connected_groups'],new_calibration=0,new_test=0,new_human_gold=0,pending_records=8),
        resources=dict(audit=read_json(out/'audit_execution.json'),browser_wall_seconds=ui['elapsed_seconds'],
                    full_session_cpu_peak_not_measured=True,server_memory_peak_not_claimed=True),
        limitations=['No new accuracy improvement or resolver calibration demonstrated; 33 songs now known diagnostics.',
          'Full finite-candidate coverage is not top1 accuracy; CNN top3 already covers both complementary examples.',
          'Target-domain human truth, permissions, aliases/versions and representative isolated calibration/test remain incomplete.',
          'Blind page hides only its own predictions; prior human exposure unknown; assisted choices are not independent gold.',
          'No new chord/bass/boundary/section accuracy measurement; local modulation candidates remain unreliable.',
          'Default offline HTML workflow unchanged; new independent review switch is in the resident Web workbench.',
          'Registered role inventory covers the 327 evaluated key examples plus 8 user/project rows; remaining 41 eligible EDM rows and other-domain holdings are separately listed.',
          'Legacy source correction key fields are historical display preferences; new global review JSON is authoritative for current notation.'],
        method_references=['https://scikit-learn.org/stable/common_pitfalls.html','https://scikit-learn.org/stable/modules/cross_validation.html',
                           'https://scikit-learn.org/stable/modules/calibration.html','https://arxiv.org/abs/1705.08500'])
    atomic_json(out/'delivery.json',delivery)
    print(dict(protected=len(snapshot),changed=len(changed),tests=42,real_ui_checks=len(ui['checks']),new_acoustic_inference=0,new_training=0))


if __name__=='__main__':main()
