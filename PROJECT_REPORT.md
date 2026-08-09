# 音乐和弦识别工作台项目报告

日期：2026-08-09

## 1. 项目目标

本项目的目标是制作一个本地音乐分析工具，针对上传或本地音频自动完成：

- BPM、调式、拍号识别
- beat/downbeat 与小节网格划分
- 按小节、半小节或拍子标注和弦
- 和弦名与罗马数字级数双显示，例如 `Bm7` / `VIm7`
- 类似 Moises 的网页播放器，在播放时同步高亮当前和弦
- 支持人工校正、曲谱导入、单元格/小节复制粘贴
- 通过公开标注数据集建立可量化评测，持续提升准确率

目前项目已经从最初的模板匹配式和弦识别，演进为：

```text
音频
  |
成熟 ACR 模型候选：madmom / Chordino / Large-Vocab
  |
beat/downbeat 小节网格
  |
Chord Studio 融合层
  |
调性、罗马数字、和声单位、段落展示、人工校正
  |
HTML 可视化播放器
```

## 2. 当前核心能力

### 2.1 音频分析与可视化

主程序：

- `app/chord_studio.py`
- `app/analyze_with_madmom.py`

输出内容：

- `_analysis.json`：完整结构化分析结果
- `_studio.html`：可播放、可编辑的本地网页
- `_bar_chords.csv` / `_bar_chords.txt`：可读和弦表
- `_madmom.lab`：madmom 原始和弦输出

网页端已经支持：

- 播放时同步高亮当前小节/当前拍
- 和弦名 / 罗马数字级数切换
- 点击某一拍进行单独编辑
- 输入 `VIm`、`VIm7`、`IV/VI`、`1`、`6m` 等级数写法校正
- 单格复制粘贴、小节复制粘贴
- 曲谱文本导入并自动铺排
- 片段拆半、拆拍、合并
- 毫秒级同步校准

### 2.2 小节与节拍处理

已支持：

- madmom downbeat 网格
- 外部 beat/downbeat 标注导入
- 参考 beat 文件中的 downbeat 标签保真
- 弱起、不完整小节、连续 downbeat 的处理
- `3/4`、`4/4`、`6/8` 等拍号
- `6/8` 下按两个大拍拆分的 `compound-half`
- 局部 BPM 与 tempo segment 输出，适合后半段渐慢歌曲

关键脚本：

- `app/downbeat_bridge.py`
- `app/retime_analysis_to_downbeats.py`
- `app/downbeat_external_fusion.py`
- `app/evaluate_beats.py`

### 2.3 和弦融合层

当前稳定路径是：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0
```

融合层目前的策略：

- madmom 作为主 ACR 候选源
- Chord Studio 负责对齐到 beat/downbeat 网格
- 默认 `harmonic-unit auto`
- 整小节为主，只有在小节内两个半小节候选足够明确时才拆成半小节
- 默认关闭全局调内加分，避免误伤借用和弦、局部转调和复杂色彩
- 防止把已经融合过的 JSON 再作为本地模板证据读入，避免反馈污染
- 保守继承同根 `7/m7` 色彩，不轻易把普通大三和弦染成 `maj7`

## 3. 数据集与评测体系

### 3.1 当前评测集

当前已建立 Beatles / Isophonics 评测集：

- 6 张专辑
- 83 首歌曲
- 包含专家 chord reference
- 包含 beat/downbeat reference
- 音频已切分到 `benchmarks/audio/isophonics/The Beatles/...`

当前主 manifest：

- `benchmarks/manifest.beatles_selected6_83.csv`

当前稳定报告：

- `benchmarks/reports/beatles_selected6_no_feedback_auto_final.json`
- `benchmarks/reports/beatles_selected6_no_feedback_auto_final.csv`

### 3.2 当前稳定指标

83 首 Beatles 评测集，当前稳定基线：

| 指标 | 数值 |
| --- | ---: |
| Root accuracy | 0.8776 |
| Triad accuracy | 0.8508 |
| Beat F1 | 0.9999 |
| Downbeat F1 | 1.0000 |
| Beat median absolute error | 约 0.001 ms |
| Downbeat median absolute error | 约 0.001 ms |

简化说：

- 节拍/小节对齐在有参考 beat/downbeat 的受控评测中已经非常稳定
- 和弦根音准确率约 88%
- 三和弦质量准确率约 85%
- 后续瓶颈主要在和弦候选质量与复杂和声理解

## 4. 外部模型接入情况

### 4.1 madmom

madmom 是当前主力模型。

特点：

- CNN/CRF 和弦识别
- 输出稳定
- 与当前融合层最匹配

当前 madmom 原始输出在 83 首上的平均表现约为：

| 模型 | Root | Triad |
| --- | ---: | ---: |
| madmom 原始 `.lab` | 0.822 | 0.794 |
| madmom + Chord Studio 融合 | 0.878 | 0.851 |

说明融合层带来了明显提升，主要来自：

- 小节网格对齐
- 和声单位约束
- 防噪声平滑
- 调性与罗马数字层辅助

### 4.2 Chordino

已完成 Sonic Annotator + Chordino 工具链检查：

- `app/check_chordino_toolchain.py`
- `app/generate_chordino_from_manifest.py`

当前 Chordino 可以正常生成 `.lab` 文件。

实测结论：

| 模型 | Root | Triad |
| --- | ---: | ---: |
| Chordino 原始输出 | 0.768 | 0.713 |

直接等权加入会显著拉低结果，低权重 0.05 基本持平但没有稳定净收益。

结论：

- Chordino 暂不建议默认启用
- 更适合作为诊断源或候选对照

### 4.3 ISMIR2019 Large-Vocabulary Chord Recognition

已接入：

- `app/generate_large_vocab_chords_from_manifest.py`

当前已完整跑完 Beatles 83 首。

实测结论：

| 模型 | Root | Triad |
| --- | ---: | ---: |
| Large-Vocab 原始输出 | 0.768 | 0.730 |

低权重融合扫描：

| 额外权重 | Root | Triad |
| --- | ---: | ---: |
| baseline | 0.877637 | 0.850844 |
| 0.02 | 0.877723 | 0.850861 |
| 0.05 | 0.877468 | 0.850103 |
| 0.08 | 0.877205 | 0.850026 |
| 0.10 | 0.877181 | 0.850039 |
| 0.15 | 0.876914 | 0.850020 |

结论：

- 该模型整体不如 madmom
- 但在约 20 首歌曲上局部优于 madmom
- 权重 0.02 只有极小提升，不足以默认启用
- 更适合后续研究“模型门控”：判断什么时候该信任第二模型

已继续实现第一版实验性门控：

- 整首层面计算 Large-Vocab 与 madmom 的兼容度
- 如果整首兼容度低于阈值，则禁用该首歌的第二模型
- 小节/半小节层面判断主模型是否足够明确
- 只有主模型不够明确，或第二模型与主模型同根/兼容时，才低权重加入第二模型

当前最佳实验配置：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70
```

对应结果：

| 配置 | Root | Triad |
| --- | ---: | ---: |
| 稳定基线 | 0.877637 | 0.850844 |
| Large-Vocab 门控实验 | 0.878687 | 0.851839 |

逐曲差异：

- 9 首提升
- 2 首下降
- 72 首不变

这说明门控方向是有效的，但当前提升仍较小，暂时应视为实验配置，而不是完全替代稳定基线。

相关诊断报告：

- `benchmarks/reports/largevocab_raw_vs_fused_diag.csv`
- `benchmarks/reports/gate_large_w005_agree070_vs_base_diff.csv`

### 4.4 片段级错误诊断与 bass-aware 实验

新增片段级诊断脚本：

- `app/diagnose_chord_segments.py`

它会对每个错误片段输出：

- 起止时间
- reference chord / prediction chord
- reference roman / prediction roman
- madmom、Large-Vocab、Chordino 同时刻候选
- 错误类型，例如 `root_error`、`quality_error`、`seventh_color_error`、`slash_bass_error`

当前 triad 级诊断报告：

- `benchmarks/reports/segment_errors_gate_large_w005_agree070.csv`

主要错误时长分布：

| 错误类型 | 累计时长 |
| --- | ---: |
| root_error | 约 959 s |
| quality_error | 约 279 s |
| missed_chord | 约 186 s |
| false_chord | 约 17 s |

exact 级诊断报告：

- `benchmarks/reports/segment_errors_exact_gate_large_w005_agree070.csv`

exact 级别下，`seventh_color_error` 和 `spelling_or_bass_display` 也非常显著，说明七和弦色彩和 slash/bass 显示是后续体验优化的重要方向。但如果只看 triad accuracy，当前最大瓶颈仍然是根音错误。

已生成独立 template/bass 证据目录：

- `benchmarks/manifest.beatles_selected6_83_template_bass.csv`
- `benchmarks/template_bass_predictions/`

并新增融合入口：

- `--base-evidence-analysis`
- `--base-evidence-dir`

这允许融合层从独立 template JSON 读取 `bass_root` / `bass_confidence`，同时不把已经融合过的主预测再次作为本地证据，避免反馈污染。

当前 bass-aware 扫描结果：

| 配置 | Root | Triad |
| --- | ---: | ---: |
| 门控实验，无 bass evidence | 0.878687 | 0.851839 |
| 门控实验，bass-root-weight 0.03 | 0.878707 | 0.851859 |

bass 证据目前只有极小收益，接近噪声级别；而片段错误诊断显示根音错误时长没有明显下降。因此这一步证明了接口可用，但当前 bass 特征还不够强，不建议把 bass-root 权重进一步放大。

### 4.5 边缘漏标修复实验

片段诊断显示，`missed_chord` 中有大量片段并不是外部模型没有识别出和弦，而是最终 beat/downbeat 网格没有覆盖歌曲开头或结尾的区域，导致评测时预测为 `N`。典型情况：

```text
reference: E
prediction: N
madmom: E
largevocab: E
chordino: E7
```

因此新增实验性边缘补全：

- `--edge-fill`
- `--edge-fill-min-duration`
- `--edge-fill-min-score`

它只在第一拍前或最后一拍后存在足够长空白，且外部模型给出明确非 `N` 和弦时，才向 `events` 时间轴补一个 `edge` 片段。它不会强行改动已有小节卡片，因此能保留纯鼓 intro 的空白显示，只在模型确实听到和弦时补全。

当前推荐实验配置：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0 --base-evidence-dir .\benchmarks\template_bass_predictions --bass-root-weight 0.03 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70 --edge-fill --edge-fill-min-score 0.50 --edge-fill-min-duration 0.50
```

对应报告：

- `benchmarks/reports/beatles_selected6_gate_bass_edge_experimental.json`
- `benchmarks/reports/gate_bass_edge_vs_base_diff.csv`
- `benchmarks/reports/segment_errors_gate_bass_edge.csv`

结果：

| 配置 | Root | Triad |
| --- | ---: | ---: |
| 稳定基线 | 0.877637 | 0.850844 |
| 门控 + bass 0.03 | 0.878707 | 0.851859 |
| 门控 + bass 0.03 + edge-fill | 0.878977 | 0.852278 |
| 门控 + bass 0.03 + segmented edge-fill | 0.879655 | 0.853077 |

逐曲差异为 27 首提升、8 首下降、48 首不变。`missed_chord` 时长从约 185.5 秒降到约 142.7 秒；副作用是 `false_chord` 从约 16.7 秒升到约 56.1 秒。总体净收益为正，但仍建议作为实验推荐配置，而不是默认无条件开启。

进一步测试发现，`summary` 模式会把整个边缘空白压成一个平均和弦，遇到包含多个和弦变化的 intro/outro 时仍会漏标。新增 `--edge-fill-mode segments` 后，边缘区域会保留外部模型的和弦分段。当前更好的实验配置：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0 --base-evidence-dir .\benchmarks\template_bass_predictions --bass-root-weight 0.03 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70 --edge-fill --edge-fill-mode segments --edge-fill-min-score 0.50 --edge-fill-min-duration 0.75
```

对应报告：

- `benchmarks/reports/beatles_selected6_gate_bass_edge_segments_experimental.json`
- `benchmarks/reports/gate_bass_edge_segments_vs_base_diff.csv`
- `benchmarks/reports/segment_errors_gate_bass_edge_segments.csv`

逐曲差异为 54 首提升、7 首下降、22 首不变。`missed_chord` 时长降到约 65.6 秒，`false_chord` 约 37.5 秒。相比 summary edge-fill，segmented 模式既减少了漏标，也减少了错误补全，是当前更值得保留的实验推荐。

### 4.6 重复段落一致性实验

新增实验性重复段落修正：

- `--repeat-consensus-refine`
- `--repeat-phrase-bars`
- `--repeat-min-support`
- `--repeat-min-share`
- `--repeat-max-score-gap`
- `--repeat-max-current-score`

它只在同一 4/8 小节循环位置多数重复为某和弦，并且当前片段自身候选分数也支持该和弦时，替换低把握离群结果。它不会凭空创造没有出现在当前候选里的和弦。

当前实验结果：

| 配置 | Root | Triad |
| --- | ---: | ---: |
| segmented edge-fill | 0.879655 | 0.853077 |
| segmented edge-fill + repeat consensus | 0.879090 | 0.852391 |

在当前 83 首 Beatles 上，repeat-consensus 相比 segmented edge-fill 反而略低。原因是规则过粗，只在少数曲目产生变化，且会误伤个别非严格循环段落。因此暂时保留为研究开关，不作为推荐配置。

## 5. 中文流行数据集扩展

### 5.1 POP909

已从官方 GitHub 获取 POP909 标注包：

- `benchmarks/raw/pop909/POP909.zip`
- `benchmarks/raw/pop909/POP909/`

该包包含 909 首中文流行歌曲的 MIDI 与对齐标注，但不包含完整商业原音频。每首包含：

- `*.mid`
- `beat_audio.txt`
- `chord_audio.txt`
- `key_audio.txt`

新增导入脚本：

- `app/import_pop909.py`

导入命令：

```powershell
python .\app\import_pop909.py .\benchmarks\raw\pop909\POP909 -o .\benchmarks\manifest.pop909.csv --import-dir .\benchmarks\imported\pop909
```

导入结果：

- `benchmarks/manifest.pop909.csv`
- `benchmarks/imported/pop909/`
- 共 909 首，0 首跳过

由于 POP909 包不含原音频，它当前最适合用于中文流行和声进行统计、Roman numeral progression prior、symbolic/chord 语言模型，以及后续配合合法音频或 MIDI 合成音频建立中文测试集。

### 5.2 中文流行和声先验

已基于 POP909 生成中文流行 progression prior：

```powershell
python .\app\progression_stats.py .\benchmarks\manifest.pop909.csv -o .\benchmarks\reports\pop909_progression_prior.json
```

输出：

- `benchmarks/reports/pop909_progression_prior.json`

高频转移包括：

| 转移 | 计数 |
| --- | ---: |
| V -> I | 5497 |
| IIm -> V | 3373 |
| Vm -> Im | 2481 |
| I -> V | 2424 |
| Im -> IVm | 2416 |
| VII -> Im | 2407 |
| V -> VIm | 2360 |
| IV -> V | 2327 |

### 5.3 POP909 Prior 跨域测试

虽然 POP909 是中文流行数据，而 Beatles 83 首不是中文流行，但仍做了副作用测试。当前最佳 segmented edge-fill 配置下，加入 POP909 prior：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\pop909_progression_prior.json --progression-prior-strength 0.04 --diatonic-shadow-weight 0 --diatonic-bonus 0 --base-evidence-dir .\benchmarks\template_bass_predictions --bass-root-weight 0.03 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70 --edge-fill --edge-fill-mode segments --edge-fill-min-score 0.50 --edge-fill-min-duration 0.75
```

结果：

| 配置 | Root | Triad |
| --- | ---: | ---: |
| segmented edge-fill | 0.879655 | 0.853077 |
| segmented edge-fill + POP909 prior 0.04 | 0.879560 | 0.853448 |

相对 segmented edge-fill，11 首提升、7 首下降、65 首不变。相对稳定基线，56 首提升、10 首下降、17 首不变。

这说明 POP909 prior 作为弱和声语言模型没有明显跨域副作用，并能带来小幅 triad 提升。但它真正的价值需要在中文流行音频 challenge set 上验证。

## 6. 已解决的关键问题

### 6.1 前端交互问题

已完成：

- 去掉小节顶部混乱的大号和弦组合显示
- 保留小节内四个/多个拍子框
- 点击某一拍即可编辑该拍
- 支持罗马数字输入校正
- 支持复制单元格和复制小节
- 支持曲谱导入后自动生成整曲标注

### 6.2 节拍与小节对齐问题

已完成：

- 保留外部 beat/downbeat 标签，不再强行按固定 4 拍循环推断
- 修复弱起和不完整小节导致的小节相位错误
- 支持 `6/8` 复合拍处理
- 对《成都》这类 6/8 歌曲，小节显示对齐明显改善
- 对后半段渐慢歌曲，输出 local BPM / tempo segments

### 6.3 和弦准确率问题

已完成：

- 从 template chroma 识别升级到 madmom 外部 ACR
- 建立 Isophonics/Beatles 评测集
- 加入 no-feedback 机制，防止重复融合自我强化
- 默认关闭会误伤的调内强加分
- 引入保守七和弦色彩继承
- 支持多模型候选源输入

## 7. 当前主要瓶颈

### 7.1 候选 ACR 质量仍是上限

目前融合层可以把 madmom 原始 triad 约 0.794 提升到 0.851，说明融合层是有效的。

但当外部模型在某段听错根音时，融合层只能部分纠正，不能凭空恢复正确和弦。

典型难点：

- slash chord，例如 `D/F#`
- 七和弦、属七、借用和弦，例如 `I7`
- 局部转调
- bass 与上方和声冲突
- 人声、吉他泛音、混响导致 chroma 混杂
- 伴奏稀疏时根音缺失

### 7.2 需要模型门控，而不是简单投票

Chordino 和 Large-Vocab 都不是整体更强，但它们在少数歌曲/片段上确实更准。

所以后续重点不是：

```text
madmom + Chordino + Large-Vocab 简单平均
```

而是：

```text
判断当前片段哪个模型更可信
```

需要利用：

- 模型间一致性
- 局部 confidence
- 和声持续性
- 小节位置
- 是否符合常见进行
- 是否与前后段落循环一致
- bass/chroma 局部证据

### 7.3 复杂和声功能仍需增强

当前系统已能输出罗马数字，但“像音乐家一样理解和声进行”还没有完全成熟。

后续需要加强：

- 常见流行歌进行概率
- 2/4/8 小节循环结构
- 主歌/副歌段落重复
- `IV-V-vi`、`vi-IV-I-V`、`I-V-vi-IV` 等常见片段偏好
- 小节末尾和声延续概率
- 不轻易把延续的 vi 错改为 IV

## 8. 下一步建议

### POP909 中文流行扩展进展

已经接入 POP909 标注并生成完整 manifest：

- `benchmarks/manifest.pop909.csv`
- `benchmarks/manifest.pop909_synth.csv`
- `benchmarks/raw/pop909/POP909/index_POP909.csv`

909 首 MIDI 合成音频已经渲染到：

- `benchmarks/audio/pop909_synth`

前 100 首在 reference beat/key 条件下完成 smoke benchmark：

- 报告：`benchmarks/reports/pop909_synth_first100_reference_grid.json`
- Root：约 0.683
- Triad：约 0.598
- Beat/Downbeat：约 0.992

这组结果主要说明两点：

- POP909 标注管线已经跑通，可用于中文流行和声先验统计。
- pretty_midi 合成音频不能替代真实商业音频评测；当前 root error 明显偏高，容易被合成钢琴声部权重、转位低音和稀疏曲尾影响。

新增真实音频链接工具：

```powershell
python .\app\link_pop909_audio.py .\benchmarks\manifest.pop909.csv ".\benchmarks\audio\pop909_real" -o .\benchmarks\manifest.pop909_real.csv
```

用户后续只需要把合法取得的 POP909 对应音频放入目录，并在文件名中保留编号、歌名或歌手，即可生成真实音频评测 manifest。

### POP909 真实音频 001-034 批次

已下载并整理用户提供的 B 站音频链接，跳过无可靠音源的 `pop909_018`：

- 下载清单：`benchmarks/audio_sources/pop909_bilibili_links_001_034.csv`
- 下载日志：`benchmarks/reports/pop909_bilibili_download_log.csv`
- 音频目录：`benchmarks/audio/pop909_real`
- 已下载：33 首
- 链接 manifest：`benchmarks/manifest.pop909_real.csv`
- 本批次 manifest：`benchmarks/manifest.pop909_real_001_034_downloaded.csv`

初始 reference beat/key 评测：

- 报告：`benchmarks/reports/pop909_real_001_034_reference_grid.json`
- Root：约 0.398
- Triad：约 0.359

诊断结果：

- 移调扫描基本没有显著改善，说明主要问题不是整体调性不一致。
- 时间偏移扫描显示多首歌存在 1.5-9.5 秒级别的起点差异。
- 部分曲目通过音频 offset 能显著提升，例如：
  - `pop909_025`：Triad 0.130 -> 0.568
  - `pop909_007`：Triad 0.297 -> 0.582
  - `pop909_026`：Triad 0.161 -> 0.397

已新增 manifest 行级 `audio_offset` 支持：

- 修改文件：`app/generate_predictions_from_manifest.py`
- 校准 manifest：`benchmarks/manifest.pop909_real_001_034_offset.csv`
- 校准报告：`benchmarks/reports/pop909_real_001_034_offset_reference_grid.json`

校准后整体结果：

- Root：约 0.419
- Triad：约 0.379

结论：

这批真实音频不能直接作为训练集使用，必须先做音频与 POP909 标注的自动对齐。下一步应实现 `align_pop909_audio.py` 一类的预处理：用 chroma/beat/chord 片段扫描最佳时间偏移，生成 shifted beat/chord reference 或 aligned manifest，再进入模型训练/门控学习。

已实现第一版保守筛选流程：

- 新脚本：`app/build_aligned_manifest_from_shift_scan.py`
- accepted manifest：`benchmarks/manifest.pop909_real_001_034_aligned_accepted.csv`
- rejected manifest：`benchmarks/manifest.pop909_real_001_034_aligned_rejected.csv`
- strict manifest：`benchmarks/manifest.pop909_real_001_034_aligned_strict.csv`

筛选策略：

- `best_triad >= 0.50`：进入 accepted，作为可复核评测数据。
- `best_triad >= 0.55`：进入 strict，作为低风险和声先验统计数据。
- 低于阈值的音频暂不进入训练，避免把版本不匹配/错位标注学进模型。

当前结果：

- accepted：14 首，Root 约 0.638，Triad 约 0.592
- strict：10 首，Root 约 0.664，Triad 约 0.612

已基于对齐后的参考标注生成中文流行和声先验：

- `benchmarks/reports/pop909_real_001_034_aligned_strict_progression_prior.json`
- `benchmarks/reports/pop909_real_001_034_aligned_accepted_progression_prior.json`

这批数据呈现明显小调倾向，常见转移包括 `VII -> Im`、`VI -> VII`、`Im -> VI`、`Im -> IVm` 等。由于样本数还小，目前只建议作为观察性先验或很弱的辅助权重，不应替代完整 POP909/Beatles 统计。

### 第一优先级：建立模型门控层

目标：

在 madmom、Large-Vocab、Chordino 和本地证据之间，不做简单平均，而是学习何时信任谁。

建议实现：

```text
每个候选片段
  |
提取特征：
  - madmom/chordino/largevocab 是否一致
  - 当前和弦持续时间
  - 前后和声转移概率
  - 是否落在强拍/小节起点
  - 是否符合段落循环
  - 是否与本地 chroma/bass 匹配
  |
门控打分
  |
选择或加权候选
```

可以先用规则实现，再逐步训练轻量模型。

### 第二优先级：错误案例驱动优化

优先分析：

- `I'm Looking Through You`
- `The Word`
- `Tell Me What You See`
- `Tomorrow Never Knows`

这些是 Large-Vocab 局部优于当前融合结果的曲目，适合研究“何时第二模型更可信”。

同时关注 Large-Vocab 严重错误的曲目：

- `Anna (Go To Him)`
- `Boys`
- `Another Girl`
- `Nowhere Man`
- `Love You To`
- `Yellow Submarine`

这些能帮助识别“何时绝对不能信任第二模型”。

### 第三优先级：扩大评测集

当前 Beatles 83 首已经足够做局部优化，但风格偏 60 年代摇滚/流行。

建议继续加入：

- 中文民谣/流行，尤其是吉他伴奏歌曲
- 现代流行歌曲
- 6/8 歌曲
- 有渐慢/自由速度的歌曲
- 和弦较简单、人工可快速校验的歌曲

建议每批新增 20-50 首，并保留：

- 音频路径
- 曲名
- 调式
- 拍号
- 参考和弦或人工校正结果

## 9. 当前可用命令

分析单首歌曲：

```powershell
python .\app\analyze_with_madmom.py "song.mp3" --key G --meter 4/4 --display roman
```

6/8 歌曲：

```powershell
python .\app\analyze_with_madmom.py "song.mp3" --key D --meter 6/8 --harmonic-unit compound-half --display roman
```

跑 83 首稳定评测：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0
python .\app\run_benchmark.py .\benchmarks\manifest.beatles_selected6_83.csv -o .\benchmarks\reports\beatles_selected6_no_feedback_auto_final
```

检查 Chordino：

```powershell
python .\app\check_chordino_toolchain.py
```

生成 Large-Vocab 输出：

```powershell
python .\app\generate_large_vocab_chords_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\largevocab\beatles_selected6_raw --force
```

## 10. 总结

当前项目已经具备一个可用的本地音乐和弦识别工作台雏形，并且已经建立起可持续优化的评测闭环。

最重要的进展是：

- 节拍/小节对齐已显著稳定
- 前端校正体验已经从“逐个手调”提升为可批量编辑
- 和弦识别从模板匹配升级到外部 ACR + 和声融合
- Beatles 83 首评测达到 root 约 88%、triad 约 85%
- Chordino 和 Large-Vocab 已接入并完成实测

下一阶段的核心不是继续堆模型，而是建立“模型门控 + 段落结构 + 和声进行概率”的判断层。也就是说，让程序不仅听到局部和弦，还要开始理解：这个位置、这个段落、这个小节里，哪个和弦更像真实音乐会选择的答案。
