"""Consolidate frozen results and provenance; never select or change policies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

from evaluate_global_key_fusion import signature


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    if not (root/'completed.json').is_file():raise RuntimeError('Experiment incomplete; no delivery summary')
    split=read('split.json');contract=read('engine_contract.json');sealed=read('harmonic_new_sealed.json')
    dev=read('harmonic_dev.json');cal=read('harmonic_calibration.json');bill=read('billboard.json')
    preselection=read('pre_selection_code.json')
    if preselection['implementation']!=contract['implementation']:raise RuntimeError('Rules changed after pre-selection snapshot')
    regressions={name:read('regressions/'+name+'/report.json') for name in ['giantsteps180','beatles17','xuni']}
    protected=read('protected_start.json');checks={path:dict(before=value['sha256'],after=signature(path)['sha256']) for path,value in protected['files'].items()}
    for item in checks.values():item['unchanged']=item['before']==item['after']
    if not all(item['unchanged'] for item in checks.values()):raise RuntimeError('A protected implementation/model changed')
    test_suites=ET.parse(root/'tests.xml').getroot().iter('testsuite')
    tests=dict(tests=0,failures=0,errors=0,skipped=0)
    for suite in test_suites:
        for key in tests:tests[key]+=int(suite.attrib.get(key,0))
    oldphase=read('acoustics/progress.json')
    acoustics=[json.loads(path.read_text(encoding='utf-8')) for path in (root/'acoustics').glob('*/acoustics.json')]
    duration=sum(json.loads(path.read_text(encoding='utf-8'))['duration'] for path in (root/'acoustics').glob('*/neutral_analysis.json'))
    artifacts=['split.json','audio_identity_audit.json','compatibility.json','pre_selection_code.json','engine_contract.json','sealed_opening.json','harmonic_dev.json',
               'harmonic_calibration.json','harmonic_new_sealed.json','billboard.json','tests.xml','resources.json','commands.json',
               'regressions/giantsteps180/report.json','regressions/beatles17/report.json','regressions/xuni/report.json']
    changed=['prepare_harmonic_key_data.py','harmonic_key_v2.py','key_profile_baseline.py','run_harmonic_key_profiles.py',
             'audit_harmonic_compatibility.py','evaluate_harmonic_key_v2.py','evaluate_harmonic_billboard_v2.py',
             'run_harmonic_key_regressions.py','finish_harmonic_key_experiment.py','summarize_harmonic_key_experiment.py']
    comparisons={}
    for name,report in [('dev',dev),('calibration',cal),('new_sealed',sealed)]+list(regressions.items()):
        rows=report['rows']
        disagree=[row for row in rows if row['cnn']!=row['harmonic_v2']]
        comparisons[name]=dict(
            disagreements=len(disagree),
            disagreement_support=dict(cnn_correct=sum(row['cnn']==row['reference'] for row in disagree),
                cnn_incorrect=sum(row['cnn']!=row['reference'] for row in disagree),
                harmonic_correct=sum(row['harmonic_v2']==row['reference'] for row in disagree),
                harmonic_incorrect=sum(row['harmonic_v2']!=row['reference'] for row in disagree),
                both_incorrect=sum(row['cnn']!=row['reference'] and row['harmonic_v2']!=row['reference'] for row in disagree)),
            v1_wrong_v2_correct=[row['id'] for row in rows if row['harmonic_v1']!=row['reference'] and row['harmonic_v2']==row['reference']],
            v1_correct_v2_wrong=[row['id'] for row in rows if row['harmonic_v1']==row['reference'] and row['harmonic_v2']!=row['reference']],
            cnn_wrong_v2_correct=[row['id'] for row in rows if row['cnn']!=row['reference'] and row['harmonic_v2']==row['reference']],
            cnn_correct_v2_wrong=[row['id'] for row in rows if row['cnn']==row['reference'] and row['harmonic_v2']!=row['reference']],
            cnn_wrong_chroma_correct=[row['id'] for row in rows if row['cnn']!=row['reference'] and row['chroma']==row['reference']])
    diagnostics=[]
    for row in sealed['rows']:
        prediction=row['prediction'];windows=prediction.get('window_evidence',{})
        selected=prediction.get('mode_candidates_by_tonic',{}).get(prediction['tonic'],[])
        selected=next((item['components'] for item in selected if item['mode']==prediction['mode']),{})
        diagnostics.append(dict(id=row['id'],pitched_coverage=prediction.get('pitched_coverage'),
            quality_coverage=selected.get('quality_coverage'),chord_tokens=selected.get('token_count'),
            contiguous_runs=selected.get('contiguous_runs'),cadence_count=selected.get('cadence_count'),
            loop=selected.get('loop'),root_fit=selected.get('root_fit'),quality_fit=selected.get('quality_fit'),
            local_valid_bars=windows.get('valid_bars',0),local_decisive_bars=windows.get('decisive_bars',0),
            local_windows=len(windows.get('windows',[])),global_status=prediction['status']))
    (root/'input_diagnostics.json').write_text(json.dumps(dict(rows=diagnostics,
        boundary_note='All methods score the same full decoded audio interval; raw gaps excluded, no reference chord boundaries available',
        candidate_note='Only top1 union oracle is informative here; all 24 keys are enumerated, so full-list oracle would trivially be 100%',
        acoustic_chord_boundary_accuracy=None,bass_accuracy=None),indent=2),encoding='utf-8')
    artifacts.append('input_diagnostics.json')
    known_commands=[
        ['app/prepare_harmonic_key_data.py','--old-split','benchmarks/reports/key_resolver_20261006/splits/giantsteps_split.json','--metadata','benchmarks/external_key_data/original_metadata.zip','--output',str(root/'split.json')],
        ['app/audit_harmonic_compatibility.py','--output',str(root/'compatibility.json')],
        ['app/run_key_resolver_audio.py','--split',str(root/'acoustic_manifest.json'),'--output',str(root/'acoustics'),'--workers','4','--phases','harmonic_dev','harmonic_calibration','harmonic_new_sealed'],
        ['app/run_harmonic_key_profiles.py','--split',str(root/'split.json'),'--output',str(root/'profiles'),'--workers','1'],
        ['-m','pytest','tests/test_harmonic_key_v2.py','-q','--junitxml='+str(root/'tests.xml')]]
    delivery=dict(schema='harmonic_key_v2_delivery',created_epoch=time.time(),objective='Remove mode asymmetry and measure complementary key candidates',
        changed_files=[signature(Path('app')/name) for name in changed]+[signature('tests/test_harmonic_key_v2.py')],
        protected_files=checks,resolver_thresholds=protected['resolver_thresholds'],default_changed=False,
        neural_training=False,hgb_training=False,resolver_training=False,harmonic_rule_parameter_selection=True,
        new_neural_audio_cases=len(acoustics),new_neural_audio_duration_seconds=duration,
        acoustic_worker_seconds_sum=sum(case['elapsed_seconds'] for case in acoustics),
        fixed_profile_distinct_audio_cases=345,fixed_profile_extractions=492,
        profile_note='147 new + 198 known regression distinct inputs; 147 profiles recomputed once after pre-freeze duration-field cache change',
        neural_case_commands=acoustics,preparation_commands=[['C:/Users/33573/miniconda3/python.exe']+argv for argv in known_commands],
        frozen_stage_commands=read('commands.json'),tests=tests,artifacts={name:signature(root/name) for name in artifacts},
        summary_command=[sys.executable]+sys.argv,
        partition_counts=split['partitions'],sealed_summary=sealed['summary'],oracles=sealed['oracles'],
        annotation_coverage={phase:dict(tracks=len(rows),tonic_labels=sum(bool(row['reference']) for row in rows),
            major_minor_family_labels=sum(bool(row['reference']) for row in rows),
            original_labels=dict(Counter(row['original_key'] for row in rows)),
            confidence_levels=dict(Counter(str(row['annotation_confidence']) for row in rows)))
            for phase in ['harmonic_dev','harmonic_calibration','harmonic_new_sealed']
            for rows in [[row for row in split['rows'] if row['harmonic_selected'] and row['harmonic_split']==phase]]},
        pre_selection_rules_unchanged=True,selection_precedes_sealed=contract['frozen_epoch']<sealed['evaluation_epoch'],
        cnn_weight_hashes={item['name']:item['sha256'] for row in sealed['rows'] for item in (row.get('cnn_models') or [])},
        resolver_prerequisite=sealed['resolver_prerequisite'],regressions_and_complementarity=comparisons,
        resources=read('resources.json'),failures=dict(acoustic=[row for row in oldphase['completed'] if 'error' in row],
            billboard=bill['failures'],regressions={name:report['failures'] for name,report in regressions.items()}),
        limitations=['Only four major examples in calibration and four in new sealed; mostly EDM',
            'Upstream CNN pretraining membership not exhaustively known',
            'Metadata artist aliases and recording versions not exhaustively verified',
            'Billboard uses known oracle chord annotations and interpolated bars; no mode or audio accuracy',
            'Harmonic scores/margins are uncalibrated; no disagreement resolver training or deployment',
            'No new chord/bass/extension/structural acoustic accuracy validation',
            'Old 180/old sealed/Beatles17/Xuni are known regressions; Xuni reference is user confirmed'])
    (root/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    def pct(value):return '—' if value is None else f'{100*value:.1f}%'
    def link(name,label):return f'[{label}](benchmarks/reports/harmonic_key_v2_20261006/{name})'
    lines=['## 和声主调分析器 v2：冻结实验（2026-10-06）','',
        '**本轮独立重构和声主调分析器，未修改默认工作台、CNN、和弦模型或裁决器。** '+
        f"新封存 33 首：harmonic v1 {sealed['summary']['harmonic_v1']['overall']['full_key_correct']}/33 → v2 {sealed['summary']['harmonic_v2']['overall']['full_key_correct']}/33；CNN {sealed['summary']['cnn']['overall']['full_key_correct']}/33。"+
        f"CNN∪v2 离线候选上限为 {sealed['oracles']['cnn ∪ harmonic_v2']['correct']}/33。自动裁决仍禁用，本轮没有训练 resolver。",'',
        link('delivery.json','完整交付与哈希')+'；'+link('engine_contract.json','冻结参数、代码 SHA 与全部选模/消融结果')+'；'+link('harmonic_new_sealed.csv','新封存逐曲结果')+'。','',
        '### 根因与新实现','',
        '旧 `detect_key_timeline.py` 的基础拟合给大调 I/IV/V 1.25、vi 1.05，而小调对应匹配为 0.95；自然小调第 V 根音只期待大三和弦，错误压低自然 v。后续 `score_key` 无条件给小调扣 0.12，主音时长/开头分支仅处理大调，并将大调 pair/loop 用于小调。`m7b5` 也先被当成 minor。这些不对称共同解释旧版的大调吸引。原代码保留作对照。','',
        '`app/harmonic_key_v2.py` 先比较 12 个 tonic，再在每个 tonic 内分别比较 major/minor；JSON 同时保存 `tonic_candidates`、`mode_candidates_by_tonic`、`final_key_candidates`。根音匹配、性质匹配、功能级数与时长分离；兼容性没有固定大小调加减分。小调自然 v 与和声小调 V/导音减和弦共存，借用和弦给软支持，未知/sus 性质被掩码。', '',
        '大小调各有终止式、pair 与循环证据，循环按全部旋转归一化；不跨 N/X 空白制造终止式或循环。主音时长是软证据。8 小节/4 小节步长局部窗口，用 12 主音 Viterbi 平滑，再条件判 mode；局部证据与全曲共享和弦，不能当成独立投票。4 小节末端只是弱乐句代理，不代表已识别主歌/副歌。','',
        link('compatibility.json','24 调 × 12 根音 × 7 性质的 2016 项兼容表及旧版探针')+'。固定 [Krumhansl-Kessler chroma profiles](https://essentia.upf.edu/reference/std_Key.html) 作为独立诊断，沿用原 22050 Hz 解码、centered fast chroma、hop=512，用精确时长帧权重。来源内分数不是概率，也没有接入 resolver。','',
        '### 数据、选择与冻结','',
        '在全部 368 条 eligible GiantSteps+ 上重建艺术家/混音者 ID、名字与作品连接组件。旧 180 首及与其相连的 41 首一并排除，剩余 147 首在推理前按组冻结：开发 86 首/76 组（8 大调、78 小调），校准 28 首/23 组（4/24），新封存 33 首/27 组（4/29）。与旧组及跨新划分交叉为零；147 份音频 SHA 彼此唯一，且不与旧 180 重合。别名/未列录音版本与 CNN 上游预训练成员无法穷尽核实。', '',
        '只在开发集预先声明的 12 组 duration/cadence/loop 权重中选择，目标依次为大小调均衡完整准确率、最弱类准确率、总体准确率；并列按固定网格顺序。校准只检查、不回选。代码 SHA/schema/全部门槛和参数已写入 contract，才首次读取新封存成绩。新封存结果之后没有调规则或门槛。', '',
        f"开发选中的主音时长权重 {contract['policy']['tonic_duration_weight']}、终止权重 {contract['policy']['cadence_weight']}、循环权重 {contract['policy']['loop_weight']}；其余固定值见 contract。review margin {contract['policy']['review_margin']} 为工程拒判提示，未经过概率校准。参考采用既有接受的 major/minor-family 口径，保留原始模态亚型；不是对完整音乐调式的评估。",'',
        '### 新封存结果','',
        '| 来源 | 主音 | 调式 | 主音正确条件下调式 | 完整主调 | 大调命中/4 | 小调命中/29 | 预测大/小/缺失 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for method in ['cnn','harmonic_v1','chroma','harmonic_v2']:
        value=sealed['summary'][method]['overall'];classes=value['predicted_modes']
        major=sealed['summary'][method]['major']['full_key_correct'];minor=sealed['summary'][method]['minor']['full_key_correct']
        lines.append(f"| {method} | {pct(value['tonic_accuracy'])} | {pct(value['mode_accuracy'])} | {pct(value['mode_given_correct_tonic'])} | {value['full_key_correct']}/33 ({pct(value['full_key_accuracy'])}) | {major} | {minor} | {classes['major']}/{classes['minor']}/{classes['missing']} |")
    lines+=['','| 来源 | 五度邻调 | 关系大小调 | 同主音大小调 | 其他/缺失 |','|---|---:|---:|---:|---:|']
    for method in ['cnn','harmonic_v1','chroma','harmonic_v2']:
        kinds=sealed['summary'][method]['overall']['confusions']
        lines.append(f"| {method} | {kinds.get('perfect_fifth',0)} | {kinds.get('relative_major_minor',0)} | {kinds.get('parallel_mode',0)} | {sum(v for k,v in kinds.items() if k not in {'exact','perfect_fifth','relative_major_minor','parallel_mode'})} |")
    lines+=['','混淆分类：五度邻调要求同大小调族、主音距离 ±7 半音；关系大小调与同主音大小调分别计数。主音正确条件下的调式准确率分母仅为主音命中的歌曲。','']
    for name,value in sealed['oracles'].items():lines.append(f"- `{name}` oracle：{value['correct']}/{value['n']}（{pct(value['accuracy'])}）。")
    needed=sealed['resolver_prerequisite']['necessary_oracle_condition']
    change=bill['overall']['tonic_change_detection'];false_changes=change.get('reference:False predicted:True',0)
    unchanged=false_changes+change.get('reference:False predicted:False',0)
    lines+=['',f"CNN 错误中 v2 新提供 {len(comparisons['new_sealed']['cnn_wrong_v2_correct'])} 个正确候选；CNN 正确而 v2 错误有 {len(comparisons['new_sealed']['cnn_correct_v2_wrong'])} 首。这是独立分支差异，不是已发生的自动覆盖。离线互补前置条件{'达到' if needed else '未达到'}；{'可以作为后续研究依据，但当前校准与曲风覆盖仍不足，本轮不训练/部署裁决器' if needed else '没有理由重新训练 CNN-vs-harmonic 裁决器'}。oracle 只用于离线评估，不进入推理特征。",'',
        f"新校准集共 {comparisons['calibration']['disagreements']} 首 CNN/v2 分歧，正确性支持 `{comparisons['calibration']['disagreement_support']}`。这是后续门控可用性诊断，没有用这些数值挑门槛。",'',
        '**目前仍不建议重训并校准自动裁决器。** 新封存的互补样本证明分支已有潜在价值，但不能回补本轮开发/校准后继续宣称同一封存验证；新校准仍没有和声纠错正例。下一阶段需要额外独立、跨风格且有纠错正负例的开发/校准材料。','',
        '### A–G 消融（仅新开发/校准）','',
        '| 划分 | 阶段 | 大调完整准确率 | 小调完整准确率 | 总体完整准确率 |','|---|---|---:|---:|---:|']
    for phase,key in [('dev','dev_ablation'),('calibration','calibration_ablation')]:
        for stage,value in contract[key].items():lines.append(f"| {phase} | {stage} | {pct(value['major']['full_key_accuracy'])} | {pct(value['minor']['full_key_accuracy'])} | {pct(value['overall']['full_key_accuracy'])} |")
    lines+=['','A=fit，B=主音时长，C=pair/cadence，D=循环，E=结尾，F=局部/Viterbi，G=弱开头证据。负消融与全部 12 组开发分数均保留在 contract，没有用封存成绩挑阶段。', '',
        '### 已知回归与 Billboard','',
        '| 已知音频 | CNN | harmonic v1 | chroma | harmonic v2 |','|---|---:|---:|---:|---:|']
    for name,report in regressions.items():
        n=report['summary']['cnn']['overall']['n'];values=[report['summary'][method]['overall']['full_key_correct'] for method in ['cnn','harmonic_v1','chroma','harmonic_v2']]
        lines.append('| '+name+' | '+' | '.join(f'{v}/{n}' for v in values)+' |')
    x=regressions['xuni']['rows'][0];prediction=x['prediction'];windows=prediction['window_evidence'];top=prediction['tonic_candidates'][:3]
    lines+=['',f"《虚拟》冻结后：CNN `{x['cnn']}`，chroma `{x['chroma']}`，harmonic v1 `{x['harmonic_v1']}`，v2 `{x['harmonic_v2']}` / `{x['status']}`。v2 主音前三为 `{[(v['tonic'],round(v['score'],4)) for v in top]}`，分差 `{prediction['margins']}`；局部可判定 {windows.get('decisive_bars',0)}/{windows.get('valid_bars',0)} 小节，主音占全部有效小节 `{windows.get('tonic_shares',{})}`。各调条件性质/时长/终止/循环成分与逐窗口原证据保存在 "+link('regressions/xuni/report.json','虚拟完整回归')+'。用户确认 C 是本曲参考，没有据此更改规则。默认仍显示 CNN/旧裁决器的暂定 G，独立实验不会自动修改网页。','',
        '旧 180（含旧 sealed24）、Beatles17 与《虚拟》全部为已知回归，原 CNN/LAB/grid 缓存复用，没有重新声学推理。逐曲改善/退步名单在 delivery，不能称为新盲测。','',
        '《虚拟》全曲 C 的具体优势来自主音和弦时长 21.9%（A 为 15.0%）、结尾落 C、四小节末端 C 支持 32.3%（A 为 19.4%），以及略高的性质拟合。C 的终止式计数实际为 0，A 为 6，不能解释为检测到了强 V→I。局部可判定部分 A 占 64/110、C 占 42/110；相对调窗口多义仍然明显，`resolved` 只代表工程分差通过，不能当作置信保证。','',
        f"Billboard：{bill['overall']['n']} 首成功、{len(bill['failures'])} 首失败；主音 {pct(bill['overall']['tonic_accuracy'])}（旧 {pct(bill['overall']['old_tonic_accuracy'])}），局部时长主音 {pct(bill['overall']['local_tonic_accuracy'])}；supported 局部准确率 {pct(bill['overall']['supported_local_accuracy'])}、覆盖 {pct(bill['overall']['supported_local_coverage'])}。改善 {bill['overall']['corrected']}、退步 {bill['overall']['broken']}。变化检出计数 `{bill['overall']['tonic_change_detection']}`。",'',
        '这是旧已知的参考和弦 + 段落插值网格 → tonic 符号诊断；没有原音频和独立 mode 真值。没有计算大小调准确率，没有从和弦生成 mode 真值再循环打分。转调只比较“是否有变化”，不验证精确边界。失败原件保留，见 '+link('billboard.json','Billboard 逐曲及失败')+'。','',
        f"**局部转调仍明显不可靠：无参考转调的 {unchanged} 首中误报 {false_changes} 首（{pct(false_changes/unchanged if unchanged else None)}）。** 全曲主音略升不能掩盖这项失败，不应将局部时间线或 supported 标签当作可信段落/转调标注。",'',
        '### 仍未解决的性能瓶颈','',
        '消融显示主音时长带来主要改善；cadence 在开发集反而减少 1 首命中，循环和 local/Viterbi 在开发/校准的完整准确率均未增加。结尾证据在校准有收益，弱开头在开发有收益，但小样本不能证明跨风格稳健性。局部多义和弦会在相对调之间换锚，当前平滑与 margin 不能充分抑制假转调。', '',
        f"新封存中 v2 有 {sealed['summary']['harmonic_v2']['overall']['confusions'].get('parallel_mode',0)} 首同主音大小调错误、{sealed['summary']['harmonic_v2']['overall']['confusions'].get('perfect_fifth',0)} 首五度邻调错误；条件调式仍弱于 CNN。chroma 只命中 {sealed['summary']['chroma']['overall']['full_key_correct']}/33，且没有为 CNN 新增正确候选，不能因为它在单曲上成功就加入自动裁决。参考和弦条件下的 Billboard 主音也未接近满分，说明乐理中心判断与声学输入两侧都仍有损失。",'',
        '### 验证、资源与限制','',
        f"- 针对性测试 {tests['tests']-tests['failures']-tests['errors']-tests['skipped']} passed；失败 {tests['failures']}、错误 {tests['errors']}、跳过 {tests['skipped']}。包含全调移调性质、大小调循环全部旋转、自然/和声小调共存、根音/性质分项、未知/空白、时间网格、局部唯一小节归属、主音/mode 指标与封存读取隔离。未重跑历史 256 项或 10 份旧预测。", 
        f"- 新声学分析 {len(acoustics)} 首，总时长 {duration/3600:.2f} 小时，4 个 worker；fixed chroma 147 首及已知回归 198 首。冻结前补齐静音输出时长字段，使 147 份 profile 缓存重算一次，共 492 次提取、345 个不同输入。没有新 CNN/HGB/LR 训练，只有声明过的规则系数选择。", 
        f"- 音频进程树采样 RSS 峰值 {delivery['resources']['peak_sampled_audio_tree_rss']/1024**3:.2f} GiB；采样从批次开始后启动，RSS 汇总可能重复计算共享页，不是全机物理内存峰值。实际阶段命令与耗时见 commands.json；轮询耗时含最多约 10 秒观察间隔。", 
        '- 所有受保护实现、resolver 门槛、冻结 v1/v2 权重/contract 哈希一致。未回放旧输出。',
        '- 新封存与校准各只有 4 首大调，主要为 EDM；广泛跨风格、新录音版本、可信置信校准尚未完成。上游 CNN 预训练重叠未知。',
        '- 本轮未验证和弦/低音/扩展音、声学边界或主歌/副歌准确率。局部证据仍可能混淆相对调、循环多义和借用和弦；固定弱先验不等于完整调性语法。','',
        '所有来源评价同一整曲解码时间支持。raw 空白被排除；输入音高/性质覆盖、窗口与可判定小节见 '+link('input_diagnostics.json','输入诊断')+'。此处候选上限仅使用各来源 top1 并集；枚举全部 24 调的 oracle 会平凡地等于 100%，不报告为能力。没有参考和弦边界，不能给出声学分段上限。','',
        '### 可执行入口','',
        '```powershell',
        'python app/harmonic_key_v2.py --raw raw.lab --neutral neutral_analysis.json --contract benchmarks/reports/harmonic_key_v2_20261006/engine_contract.json --output harmonic_v2.json',
        'python app/key_profile_baseline.py song.mp3 --output profile.json',
        'python app/evaluate_harmonic_key_v2.py evaluate --split benchmarks/reports/harmonic_key_v2_20261006/split.json --acoustics benchmarks/reports/harmonic_key_v2_20261006/acoustics --profiles benchmarks/reports/harmonic_key_v2_20261006/profiles --contract benchmarks/reports/harmonic_key_v2_20261006/engine_contract.json --phase harmonic_new_sealed --output key_evaluation.json',
        '```','',
        '评估重跑只读取冻结缓存；不得在修改后沿用旧 contract。选模入口拒绝覆盖已存在 contract。完整实际运行 argv、输入/权重 SHA、逐曲/曲风成绩、消融与退步名单见交付 JSON。','']
    project=Path('PROJECT_REPORT.md');text=project.read_text(encoding='utf-8');marker='## 和声主调分析器 v2：冻结实验（2026-10-06）'
    if marker in text:raise RuntimeError('Report already summarized; refusing duplicate sections')
    start=text.index('## 实验主调融合')
    header='# 当前实现、性能与瓶颈\n\n更新日期：2026-10-06。最新完成独立和声主调 v2 冻结实验；默认工作台与冻结模型保持原样。使用方法见 [README](README_chord_recognizer.md)。\n\n'
    historical='以下为此前裁决器实验的历史结果；其中旧 180/旧 sealed24 如今均视为已知回归，不是本轮新封存成绩。\n\n'
    project.write_text(header+'\n'.join(lines)+'\n'+historical+text[start:],encoding='utf-8')
    delivery['changed_files'] += [signature(project),signature('README_chord_recognizer.md')]
    (root/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(dict(delivery=str(root/'delivery.json'),report=str(project),protected_unchanged=True))


if __name__=='__main__':main()
