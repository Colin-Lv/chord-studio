"""Finalize A's evidence and archive; no model or original reference writes."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from collections import Counter

from run_chord_error_repair_A import ROOT, load, write, table, sha, protect_check


def finish(out):
    delivery=load(out/'delivery.json')
    if delivery.get('listening_audio_cache_hits'):
        previous=ROOT/'benchmarks/reports/chord_error_repair_A_20261006/delivery.json'
        write(out/'audio_generation_execution.json',load(previous))
        delivery['listening_excerpts_generation_record']='audio_generation_execution.json'
        delivery['total_task_listening_decodes']=26
        delivery['resource_limitations']='2.67s final preparation and 6.17s initial preparation are separately measured; no RSS profiling; no acoustic inference/training'
    corpus=ROOT/'exports/manual_model_comparison_20261006'
    with (corpus/'all_primary_interval_comparisons.csv').open(encoding='utf-8-sig') as f:
        primary=list(csv.DictReader(f))
    kept,artifacts=[],[]
    for row in primary:
        a,b=float(row['start_seconds']),float(row['end_seconds'])
        left,right=round(a*22050),round(b*22050)
        if right<=left:
            artifacts.append({'reason':'sub_analysis_sample_intersection','original_row':row})
        else:
            kept.append({**row,'canonical_start_sample':left,'canonical_end_sample':right,
                         'canonical_duration_seconds':(right-left)/22050})
    table(out/'normalized_comparison_intersections.csv',kept)
    write(out/'numeric_fragment_evidence.json',{'original_rows':len(primary),'retained_rows':len(kept),
        'artifact_count':len(artifacts),'artifact_raw_seconds':sum(float(x['original_row']['duration_seconds']) for x in artifacts),
        'policy':'sample precision only, not minimum chord duration; original complete CSV unchanged', 'artifacts':artifacts})
    ranking_path=ROOT/'benchmarks/reports/key_top3_rerank_20261006_v1/xuni_regression.json'
    ranking=load(ranking_path)
    deltas=[]
    for name,model in ranking['models'].items():
        terms={x['key']:x for x in model['contributions']}
        scores={x['key']:x['score'] for x in model['ranking']}
        for field,c in terms['C']['terms'].items():
            deltas.append({'model':name,'feature':field,'C_contribution':c,'G_contribution':terms['G']['terms'][field],
                           'C_minus_G':c-terms['G']['terms'][field]})
        assert abs(sum(r['C_minus_G'] for r in deltas if r['model']==name)-(scores['C']-scores['G']))<1e-9
    table(out/'xuni_existing_ranker_contributions.csv',deltas)
    write(out/'key_task_audit.json',{'legacy_key':'C','existing_user_confirmation':'C, as documented; retain prior reference, do not redefine old metrics',
        'new_schema_policy':'display coordinate and actual tonic/mode are separate; explicit task confirmation required before new tonic-mode training',
        'xuni_top3':ranking['cnn_top3'],'M1':ranking['models']['M1']['ranking'],'M2':ranking['models']['M2']['ranking'],
        'source_sha256':sha(ranking_path),'new_ranking_or_training':False,'new_model_accuracy_claim':False})
    table(out/'listened_review_response_template.csv',[{'review_id':r['review_id'],'review_status':'unresolved','heard_pitches':'',
        'explicit_bass':'','chord_label':'','notation_key':'','actual_tonic_pc':'','mode':'',
        'confirmed_fields':'','boundary_notes':'','recording_sha256':r['audio_sha256'],'reviewer':''} for r in load(out/'listening_queue.json')])
    delivery.update(numeric_fragment_count=len(artifacts),numeric_fragment_raw_seconds=sum(float(x['original_row']['duration_seconds']) for x in artifacts),
                    normalized_comparison_rows=len(kept),targeted_test_log='tests.txt',**protect_check(out))
    with (out/'semantic_changes.csv').open(encoding='utf-8-sig') as f:changes=list(csv.DictReader(f))
    actual=Counter((r['case_id'],r['label']) for r in changes if r['identity_changed']=='True')
    write(out/'semantic_contract.json',{'schema':'reference_semantics_contract_v3','semantics_schema':'chord_semantics_v3',
        'feature_model_contract':None,'model_retraining_performed':False,'old_weights_with_new_semantics_allowed':False,
        'code_sha256':{p:sha(ROOT/p) for p in ['app/chord_semantics_v3.py','app/reference_events_v1.py','app/run_chord_error_repair_A.py','app/finish_chord_error_repair_A.py','app/chord_semantics_v2.py']},
        'analysis_sample_rate':22050,'legacy_source_policy':'legacy_studio_bars_spans','review_required_for_training':True})
    lines=['# A：语义与参考确定性修复','',
        '已读五份审查资料与附加说明，仅实施第一轮 A。没有修改默认页面、旧解析器、声学权重、冻结契约或原人工参考。', '',
        '## 实际修复','',
        '- 新建 chord_semantics_v3：7sus4/9sus4（及 sus2）映射到标准结构；保留原文、slash、扩展与删除度数。旧 v2 不变，旧权重不能直接使用新语义。',
        '- 将11份对照另存 canonical_reference_events_v1。canonical_events 是唯一可编辑数据源；events、bars.spans、beats 在同一 revision 派生。beats 内变化用 segments 表示，mixed 不冒充单一和弦。',
        '- 显式导入 policy 固定为 bars.spans，保留旧视图冲突诊断。新事务支持版本冲突、字段级确认及带锁原子保存；未接入旧前端。',
        '- 根音、性质、七音、9/11/13、低音、主音/调式分别掩码。无 slash 不算根音低音；未写七音/扩展不当缺失负例；confidence=1/edited=true 不升级听辨真值。',
        '- 新数据将显示母调与实际 tonic/mode 分开，旧 G/C/A/E 字段保留作显示坐标，不能直接生成大小调训练目标。原历史主调指标未重写。', '',
        '## 真实结果与未决事项','',
        f'- 2282个保存事件中，37个语义身份改变、2245个保持；全部改变都是 G9sus4 别名恢复，不是模型听辨提高。各版计数：{dict(actual)}。',
        '- 最新老男孩为16个规范原事件、23.370918秒；审查的18个是与模型边界交切后的对照片段，二者计数层次不同。',
        '- 老男孩07旧文件确为213 spans / 262 events；ヨルシカ97首 beat=E、首 span=F#m。新规范保留97小节16格，最短约0.11746秒，未用最短和弦时长吞并。',
        '- 紅蓮華七处、12.410019秒音集/半减七标签冲突仍待原声复核；只把相应七音目标掩码，不自动改成 dim7，更没有全库替换 m7b5。具体区间在 gurenge_conflicts.csv。',
        f'- 原2675个交切片段中，{len(artifacts)}个低于分析采样精度的浮点碎片另存原证据，{len(kept)}个有效片段保留。量化只定义新参考时钟；没有为了提高分数移动预测，没有覆盖旧时间。',
        '- 26个完整短句复核队列及原声 WAV 已生成。只是解码/截取本机已有音频，不是声学模型推理、音高分析或人工听辨；全部仍 unresolved。最初26次解码约6.17秒准备时间，最终版按音源SHA/边界/片段SHA复用全部26条，最终准备约2.67秒，不再次解码；见 audio_generation_execution.json。没有采样RSS。',
        '- 各字段“结构上已写明的时长”与“明确听辨后可用于可靠训练的时长”分列。目前后者全为0：原文件没有字段级听辨确认记录，不能由文件名/修订标志反推。',
        '- 借用、次属、N、快速变化与复制来源全部保留。来源可叠加，来源时长不可相加；多个版本不累计成独立作品。',
        '- 虚拟原 M1/M2 均仍G；已提取 C-G 的逐特征贡献，线性和与原分差一致，没有重新训练或重复实现Top3排序。',
        '- 审查证据JSON未在这五个文件中提供；本轮直接从本机原JSON/CSV复现，没有虚构缺失审计制品。', '',
        '## 验证与使用','',
        '- 见 tests.txt：只运行本轮新错误案例测试，不重跑历史256项或10首字节回放。708个既有实现/模型/契约/人工数据/网页结果哈希全部保持。',
        '- 双击 listening_queue.html，先听，再展开候选问题；表单回应可填写 listened_review_response_template.csv。播放不会自动提交答案或改变参考。',
        '- references/*.reference_events_v1.json 保留旧原文与原始时间；所有源路径、参考SHA和音频SHA见 reference_identities.json 与 listening_queue.json。',
        '- 监督覆盖逐版见 supervision_coverage.csv；delivery.json 的 selected_group_coverage 按7个作品各选一个版本，未将重复版本相加。虚拟没有逐和弦真值。',
        '- B/C消融、候选流失的声学定位、净纠错/误改与新准确率：本轮未做，按任务书留待第二轮。A的代码和规范完成，但26句实际复核尚未完成，不开展大规模训练。', '',
        '实际命令：','', '```powershell',
        'C:/Users/33573/miniconda3/python.exe -m pytest tests/test_chord_error_repair_A.py -q',
        'C:/Users/33573/miniconda3/python.exe '+' '.join(delivery['actual_argv']),
        'C:/Users/33573/miniconda3/python.exe app/finish_chord_error_repair_A.py --output '+out.relative_to(ROOT).as_posix(),'```','',
        '运行入口拒绝覆盖既有新参考版本。开始新轮次需另建目录，并先冻结输入/受保护文件身份。', '']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'delivery.json',delivery)
    for p in ['app/chord_semantics_v3.py','app/reference_events_v1.py','app/run_chord_error_repair_A.py','app/finish_chord_error_repair_A.py','tests/test_chord_error_repair_A.py']:
        target=out/'implementation'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    manifest=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'package_manifest.json',manifest)
    archive=ROOT/'exports'/(out.name+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for m in manifest:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(archive),'zip_bytes':archive.stat().st_size,'CRC_SHA_verified':True,**protect_check(out)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'benchmarks/reports/chord_error_repair_A_20261006')
    finish(p.parse_args().output.resolve())
