# 本地音乐和弦识别工作台

这个工具会分析本地音频，并输出 BPM、调式、拍号、小节/拍子级和弦，以及一个可播放并同步高亮当前和弦的 HTML 界面。

支持常见音频格式：`mp3`、`m4a`、`wav`、`flac`。

如果换到新电脑，先安装依赖：

```powershell
python -m pip install -r requirements_chord_studio.txt
```

## 当前目录结构

- `app\`：程序源码，主程序是 `app\chord_studio.py`
- `projects\panopticon\`：Panopticon 音频、正式分析结果和测试结果
- `projects\BV1abRqB8EPb\`：另一个 B 站音频及前 60 秒分析结果
- `projects\BV1ff4y1779y\`：简单歌曲测试音频和同步测试结果
- `archive\cache\`：Python 缓存文件

## 快速使用

在当前文件夹打开 PowerShell：

```powershell
python .\app\chord_studio.py "音频文件.m4a"
```

更推荐的高准确率路径是使用 madmom 的 CNN+CRF 和弦模型，再接入本项目的拍点网格、罗马数字显示和校正界面：

```powershell
python .\app\analyze_with_madmom.py "音频文件.m4a" --key E --meter 4/4 --display roman
```

如果已知 BPM，可以一并给出，速度会更快且小节网格更稳定：

```powershell
python .\app\analyze_with_madmom.py "音频文件.m4a" --key E --meter 4/4 --bpm 161.29 --display roman
```

如果一小节内经常有两个和弦，例如 `I-V | VIm-IV | V-I`，可以直接使用半小节融合：

```powershell
python .\app\analyze_with_madmom.py "音频文件.m4a" --key G --meter 4/4 --harmonic-unit half-bar --display roman
```

如果已经生成过 madmom `.lab`，可以复用它来快速尝试不同小节粒度：

```powershell
python .\app\analyze_with_madmom.py "音频文件.m4a" -o ".\projects\song_halfbar" --key G --meter 4/4 --harmonic-unit half-bar --madmom-lab ".\projects\song_madmom.lab"
```

生成结果：

- `_madmom.lab`：madmom 原始和弦识别结果
- `_analysis.json`：融合到拍子/小节后的 Chord Studio 数据
- `_studio.html`：可播放、可校正、可切换和弦名/罗马数字的界面

也可以用批处理入口：

```powershell
.\recognize_chords.bat "音频文件.m4a"
```

生成结果：

- `_bar_chords.txt`：按小节阅读的和弦表
- `_bar_chords.csv`：适合 Excel 打开，包含 `chord`、`roman`、`bass_root`、`bass_confidence` 等列
- `_analysis.json`：给程序或网页使用的完整分析数据
- `_studio.html`：类似 Moises 的本地播放器界面，可切换“和弦名/级数”，小节卡片内只显示各拍和弦，并带毫秒级同步校准

## 常用修正

自动 BPM 不准时：

```powershell
python .\app\chord_studio.py "song.m4a" --bpm 128
```

自动调式不准时：

```powershell
python .\app\chord_studio.py "song.m4a" --key F#m
```

默认用罗马数字级数显示：

```powershell
python .\app\chord_studio.py "song.m4a" --display roman
```

## 准备评测音频

公开数据集只提供标注，不提供商业音频。拿到合法的整张专辑音频后，可以用时间戳表自动切成单曲，并回填评测清单里的 `audio` 路径。例如 `Please Please Me`：

```powershell
python .\app\split_album_by_timestamps.py ".\path\to\Please Please Me full album.m4a" --segments ".\benchmarks\audio_sources\please_please_me_segments.csv" --out-dir ".\benchmarks\audio\isophonics\The Beatles\01 - Please Please Me" --format mp3 --overwrite --update-manifest ".\benchmarks\manifest.public_dev.csv"
```

`With the Beatles` 的整张专辑也可以用同样流程切分：

```powershell
python .\app\split_album_by_timestamps.py ".\path\to\With The Beatles full album.mp3" --segments ".\benchmarks\audio_sources\with_the_beatles_segments.csv" --out-dir ".\benchmarks\audio\isophonics\The Beatles\02 - With the Beatles" --format mp3 --overwrite --update-manifest ".\benchmarks\manifest.isophonics_beatles_all.csv"
```

切分完成后，前两张 Beatles 专辑共 28 首可以用 `benchmarks\manifest.beatles_first2_28.csv` 做小型回归评测。

## 当前评测结论

在 `Please Please Me` 专辑 14 首、Isophonics 专家标注、参考 beat grid + 专家 key 的受控评测下：

| 方法 | Root accuracy | Triad accuracy |
| --- | ---: | ---: |
| 本项目原 template 模型 | 0.560 | 0.515 |
| madmom 直接 `.lab` | 0.828 | 0.813 |
| madmom + Chord Studio 网格/HTML 融合 + Beatles 进行先验 | 0.873 | 0.866 |

在 `Please Please Me` + `With the Beatles` 共 28 首、参考 beat/downbeat 标签保真、专家 key 的受控评测下，当前 `madmom + Chord Studio 融合 + Beatles 进行先验` 为：

| Report | Root accuracy | Triad accuracy | Beat F1 | Downbeat F1 |
| --- | ---: | ---: | ---: | ---: |
| `benchmarks\reports\beatles_first2_labeled_grid_prior_s0p045.json` | 0.886 | 0.860 | 1.000 | 1.000 |

这版修复了参考 beat 文件里弱起、不完整小节或连续 downbeat 标签被强行按固定 4 拍循环的问题。对于小节相位错误，优先检查 beat/downbeat 标签是否被保留下来，而不是只看全局 BPM。

继续接入 `A Hard Day's Night` 后，前三张 Beatles 专辑共 41 首可以用 `benchmarks\manifest.beatles_first3_41.csv` 评测：

| Report | Root accuracy | Triad accuracy | Beat F1 | Downbeat F1 |
| --- | ---: | ---: | ---: | ---: |
| `benchmarks\reports\beatles_first3_labeled_grid_prior_s0.json` | 0.880 | 0.857 | 1.000 | 1.000 |

这次还修复了评测器对 Isophonics `Point` 标记的处理：`Point` 是事件标记，不是和弦，不能参与 chord accuracy 计算。41 首参数扫描显示，当前 `--progression-prior-strength 0` 的 triad 略好，说明 Beatles 的特殊进行较多，过强的通用进行先验会误伤局部和弦。

继续接入 `Help!`、`Rubber Soul`、`Revolver` 后，当前已下载 6 张 Beatles 专辑共 83 首，可用 `benchmarks\manifest.beatles_selected6_83.csv` 评测：

| Report | Root accuracy | Triad accuracy | Beat F1 | Downbeat F1 |
| --- | ---: | ---: | ---: | ---: |
| `benchmarks\reports\beatles_selected6_labeled_grid_bar_prior_s0_final.json` | 0.864 | 0.834 | 1.000 | 1.000 |
| `benchmarks\reports\beatles_selected6_labeled_grid_auto_prior_s0_final.json` | 0.883 | 0.854 | 1.000 | 1.000 |
| `benchmarks\reports\beatles_selected6_no_feedback_auto_final.json` | 0.878 | 0.851 | 1.000 | 1.000 |

83 首上新的 `auto + no-feedback` 是当前稳定基线。它默认保持整小节，只有当一个小节内部两个半小节的外部候选都很明确且根音不同，才局部拆成半小节。`beat` 级过碎，会放大局部候选噪声。这次扫描显示，全局调内质量加分会轻微误伤借用和弦、局部转调和大小调色彩，因此默认关闭 `diatonic_bonus`。融合脚本现在也会避免把已经融合过的 `external-fusion` JSON 再当作本地模板证据读入，防止重复评测时发生自我强化。七和弦补色保持保守：只在同根且本地证据足够强时继承 `7/m7`，不自动把普通大三和弦染成 `maj7`。

融合层已支持多个外部 ACR 候选源。主目录仍放 madmom 结果；如果后续接入 Chordino、BTC 或其他模型，只要输出同名 `.lab` 文件，就可以作为额外目录加入：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --external-chords-dir-extra .\benchmarks\external_chords\chordino\beatles_selected6
```

Chordino 接入准备：

```powershell
python .\app\check_chordino_toolchain.py
python .\app\generate_chordino_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\chordino\beatles_selected6 --force
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --external-chords-dir-extra .\benchmarks\external_chords\chordino\beatles_selected6 --external-chords-extra-weight 0.05
```

当前 `tools\sonic-annotator-1.7-win64\...` 已可运行；还需要系统可见的 Vamp Plugin Pack / NNLS Chroma Chordino 插件。检查脚本会确认 `vamp:nnls-chroma:chordino:simplechord` 是否出现在 Sonic Annotator 的 `--list` 输出里。

实测 Chordino 在当前 Beatles 83 首评测集上弱于 madmom-only 融合，直接等权加入会把 triad 从 0.851 拉低到约 0.808；低权重 0.05 基本持平但没有净收益。因此 Chordino 目前更适合作为对照/诊断候选源，不建议默认启用。后续如果接入 BTC 等更强模型，可以复用同一个 `--external-chords-dir-extra` 接口。

已额外接入并测试 `ISMIR2019-Large-Vocabulary-Chord-Recognition` 深度模型，生成脚本是：

```powershell
python .\app\generate_large_vocab_chords_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\largevocab\beatles_selected6_raw --force
```

这个模型在 83 首 Beatles 评测集上的原始输出为 root 0.768、triad 0.730，整体低于 madmom 原始输出；但它在约 20 首歌上局部优于 madmom。作为第二候选源进行低权重融合时，简单加权几乎没有净收益：`--external-chords-extra-weight 0.02` 只带来极小提升，0.05 以上开始拉低结果。因此它不适合直接默认加入。

目前已加入实验性的第二模型门控：当整首层面 Large-Vocab 与 madmom 兼容度过低时禁用第二模型；在单个小节/半小节内，只有主模型不够明确或二者同根兼容时，才允许第二模型以低权重参与。当前最佳实验配置：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70
```

对应报告 `benchmarks\reports\beatles_selected6_gate_large_w005_agree070_experimental.json`：root 0.8787、triad 0.8518。相对稳定基线 root 0.8776、triad 0.8508 有小幅提升；逐曲差异为 9 首提升、2 首下降、72 首不变。这个提升还不够大，所以暂时标记为实验配置，而不是无条件默认配置。

现在还可以生成片段级错误诊断，直接查看哪些时间段错了、错成什么、各个外部模型在同一时刻给了什么候选：

```powershell
python .\app\diagnose_chord_segments.py .\benchmarks\manifest.beatles_selected6_83.csv -o .\benchmarks\reports\segment_errors_gate_large_w005_agree070.csv --madmom-dir .\benchmarks\external_chords\madmom\beatles_selected6 --largevocab-dir .\benchmarks\external_chords\largevocab\beatles_selected6_raw --chordino-dir .\benchmarks\external_chords\chordino\beatles_selected6 --level triad
```

如果要研究七和弦、slash chord 和低音显示，使用 exact 级别：

```powershell
python .\app\diagnose_chord_segments.py .\benchmarks\manifest.beatles_selected6_83.csv -o .\benchmarks\reports\segment_errors_exact_gate_large_w005_agree070.csv --madmom-dir .\benchmarks\external_chords\madmom\beatles_selected6 --largevocab-dir .\benchmarks\external_chords\largevocab\beatles_selected6_raw --chordino-dir .\benchmarks\external_chords\chordino\beatles_selected6 --level exact
```

已新增独立 bass/template 证据接口。先生成不覆盖主预测的 template/bass JSON：

```powershell
python .\app\generate_predictions_from_manifest.py .\benchmarks\manifest.beatles_selected6_83_template_bass.csv --force --use-reference-key --use-reference-beat-grid --default-meter 4/4 --feature-mode fast --harmonic-unit beat --bass-weight 0.75
```

然后在融合层读取它作为 bass evidence：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0 --base-evidence-dir .\benchmarks\template_bass_predictions --bass-root-weight 0.03 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70
```

目前 bass-root-weight 0.03 的报告 `benchmarks\reports\beatles_selected6_gate_large_bass003_experimental.json` 为 root 0.8787、triad 0.8519，只比不带 bass 的门控实验略高，收益接近噪声。因此 bass 接口已经可用，但当前低频特征还不能强力解决根音错误。

片段诊断还发现一类更直接的问题：部分歌曲开头/结尾超出 beat 网格覆盖范围，最终预测为 `N`，但外部模型其实在这些边缘区域给出了明确和弦。可以启用实验性边缘补全：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0 --base-evidence-dir .\benchmarks\template_bass_predictions --bass-root-weight 0.03 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70 --edge-fill --edge-fill-min-score 0.50 --edge-fill-min-duration 0.50
```

报告 `benchmarks\reports\beatles_selected6_gate_bass_edge_experimental.json`：root 0.8790、triad 0.8523。相对稳定基线，27 首提升、8 首下降、48 首不变。`missed_chord` 明显减少，但 `false_chord` 会增加，所以这个功能适合评测和需要完整时间轴的场景；对于明确纯鼓/无和弦 intro，仍建议保留空白或人工标 `N`。

更好的边缘补全方式是 segmented 模式。它不会把整个边缘空白压成一个平均和弦，而是保留外部模型在 intro/outro 里的和弦分段：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\isophonics_beatles_progression_prior.json --progression-prior-strength 0 --diatonic-shadow-weight 0 --diatonic-bonus 0 --base-evidence-dir .\benchmarks\template_bass_predictions --bass-root-weight 0.03 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70 --edge-fill --edge-fill-mode segments --edge-fill-min-score 0.50 --edge-fill-min-duration 0.75
```

报告 `benchmarks\reports\beatles_selected6_gate_bass_edge_segments_experimental.json`：root 0.8797、triad 0.8531。相对稳定基线，54 首提升、7 首下降、22 首不变；`missed_chord` 降到约 65.6 秒，`false_chord` 约 37.5 秒。这是当前最好的实验推荐配置。

也新增了重复段落一致性实验开关 `--repeat-consensus-refine`，但它在当前 83 首 Beatles 上不如 segmented edge-fill，暂时不建议启用。

## 中文流行数据集

已获取并导入 POP909 标注包。这个包不含完整商业原音频，但包含 909 首中文流行歌曲的 MIDI、音频时间轴 beat/chord/key 标注，非常适合训练中文流行和声语言模型。

```powershell
python .\app\import_pop909.py .\benchmarks\raw\pop909\POP909 -o .\benchmarks\manifest.pop909.csv --import-dir .\benchmarks\imported\pop909
python .\app\progression_stats.py .\benchmarks\manifest.pop909.csv -o .\benchmarks\reports\pop909_progression_prior.json
```

已生成：

- `benchmarks\manifest.pop909.csv`
- `benchmarks\imported\pop909\`
- `benchmarks\reports\pop909_progression_prior.json`

POP909 prior 作为弱和声语言模型接入后，在 Beatles 83 首上做跨域副作用测试也有小幅提升：

```powershell
python .\app\generate_external_fusion_from_manifest.py .\benchmarks\manifest.beatles_selected6_83.csv .\benchmarks\external_chords\madmom\beatles_selected6 --harmonic-unit auto --progression-prior .\benchmarks\reports\pop909_progression_prior.json --progression-prior-strength 0.04 --diatonic-shadow-weight 0 --diatonic-bonus 0 --base-evidence-dir .\benchmarks\template_bass_predictions --bass-root-weight 0.03 --external-chords-dir-extra .\benchmarks\external_chords\largevocab\beatles_selected6_raw --external-chords-extra-weight 0.05 --extra-gating conservative --extra-gating-min-track-agreement 0.70 --edge-fill --edge-fill-mode segments --edge-fill-min-score 0.50 --edge-fill-min-duration 0.75
```

报告 `benchmarks\reports\beatles_selected6_pop909_prior_s004_experimental.json`：root 0.8796、triad 0.8534。它真正的价值还需要用中文流行音频 challenge set 验证。

所以目前默认优化方向是：让成熟 ACR 模型负责“听到了什么和弦”，本项目负责 beat/downbeat 对齐、罗马数字/功能分析、段落化展示和人工高效校正。

自动拍号不准时：

```powershell
python .\app\chord_studio.py "song.m4a" --meter 3/4
```

如果小节线或拍位总是对不齐，尤其是 `3/4`、`6/8`、弱起或复合拍歌曲，可以启用 madmom 的神经网络 downbeat 追踪：

```powershell
python .\app\chord_studio.py "song.m4a" --beat-source madmom --meter 6/8
```

如果已经有一版和弦识别结果，只想替换成更准的 beat/downbeat 网格，可以先导出 madmom downbeat，再快速重排原标注：

```powershell
python .\app\downbeat_bridge.py "song.m4a" -o ".\projects\song_downbeats_6-8.json" --beats-per-bar 6
python .\app\retime_analysis_to_downbeats.py ".\projects\song_analysis.json" ".\projects\song_downbeats_6-8.json" -o ".\projects\song_retimed" --meter 6/8
```

如果已有外部和弦模型输出，例如 madmom/Chordino/BTC 的 `.lab`，可以直接在 downbeat 网格上做快速候选融合。这个路径不会重新提取 chroma，适合先快速得到一版更稳定的小节级标注：

```powershell
python .\app\downbeat_external_fusion.py ".\projects\song_analysis.json" ".\projects\song_downbeats_6-8.json" ".\projects\song_madmom.lab" -o ".\projects\song_external_fused" --key D --meter 6/8 --harmonic-unit bar
```

`--harmonic-unit half-bar` 会把 `4/4` 拆成 `1-2 / 3-4`，适合 `176451` 这类一小节内可能换两次和弦的流行歌；`--harmonic-unit compound-half` 会把 `6/8` 拆成 `1-3 / 4-6` 两个大拍；`bar` 则更适合民谣、抒情歌这种整小节换和弦的情况。

融合层会保守继承本地 template 模型中的同根属七和弦色彩，例如 D 调里较可靠的 `D7` 会显示为 `I7`，`A7` 会显示为 `V7`。可以用下面参数调节继承阈值，数值越高越保守：

```powershell
python .\app\downbeat_external_fusion.py ".\projects\song_analysis.json" ".\projects\song_downbeats_6-8.json" ".\projects\song_madmom.lab" -o ".\projects\song_external_fused" --key D --meter 6/8 --harmonic-unit compound-half --dominant-seventh-threshold 0.55
```

对于后半段有渐慢或自由速度的歌曲，madmom beat grid 会保留真实拍点时间；最终 JSON 还会写入每小节 `local_bpm` 和整体 `tempo_segments`，HTML 顶部 BPM 会显示全局 BPM 及局部范围。

现在还可以从公开标注统计真实流行歌和声转移概率，并作为融合层的弱先验。当前默认的一键 madmom 流程会自动使用 `benchmarks\reports\isophonics_beatles_progression_prior.json`，如果这个文件存在：

```powershell
python .\app\progression_stats.py .\benchmarks\manifest.isophonics_beatles_all.csv -o .\benchmarks\reports\isophonics_beatles_progression_prior.json
python .\app\analyze_with_madmom.py "song.m4a" --key D --meter 6/8 --harmonic-unit compound-half --display roman
```

也可以手动指定先验强度。当前在 `Please Please Me` 14 首回归集上，`0.045` 表现最好：

```powershell
python .\app\analyze_with_madmom.py "song.m4a" --key G --meter 4/4 --harmonic-unit half-bar --progression-prior ".\benchmarks\reports\isophonics_beatles_progression_prior.json" --progression-prior-strength 0.045
```

如果自动识别把慢歌判断成 2 倍 BPM，可以强制只比较半速候选：

```powershell
python .\app\chord_studio.py "song.m4a" --tempo-mode half
```

如果一首歌的和弦常在半小节变化，可以显式使用半小节和声单位：

```powershell
python .\app\chord_studio.py "song.m4a" --harmonic-unit half-bar
```

如果某些非重复结构的音乐被重复段先验带偏，可以关闭二次细化：

```powershell
python .\app\chord_studio.py "song.m4a" --no-repeat-refine
```

如果某些音乐的半小节换和弦非常自由，不适合流行歌两小节循环的短语位置先验，可以关闭短语细化：

```powershell
python .\app\chord_studio.py "song.m4a" --no-phrase-refine
```

生成后直接打开播放器：

```powershell
python .\app\chord_studio.py "song.m4a" --open
```

## 人工校正与导出

生成的 `_studio.html` 现在可以直接校正和弦。点击某个拍子框会选中这一拍，例如点第 49 小节第 3 拍的 `I` 后，保存只会修改这一拍；点击小节空白区域或右侧和弦列表则会选中对应片段。支持 `N`、`D`、`Bm7`、`D/F#`、`C:maj`、`A:min` 这类真实和弦名，也支持 `I`、`VIm`、`VIm7`、`IV/VI`、`1`、`6m` 这类调内级数写法。页面会同步更新和弦名与罗马数字级数。

右侧“曲谱导入”面板可以粘贴或上传文本和弦谱，再从指定小节开始自动铺满整首。支持常见写法：

```text
| D | Bm | G | A |
| I | vi | IV | V |
176451
```

在当前调内，`1 6 4 5`、`I vi IV V` 会自动转换成实际和弦。若一小节内写多个和弦，例如 `| D Bm | G A |`，会自动按半小节或拍子拆分；也可以在“单位”里指定“小节 / 半小节 / 拍”。

如果某处不对齐，可以选中片段后用“拆半”“拆拍”细分到更小单位，再局部修改。相邻相同和弦可以用“合并”整理。

标注时也可以用内部剪贴板快速复用：选中某一拍后点“复制格 / 粘贴格”，或选中任意小节内的拍/片段后点“复制小节 / 粘贴小节”。快捷键是 `Ctrl+C` / `Ctrl+V` 复制粘贴当前格，`Ctrl+Shift+C` / `Ctrl+Shift+V` 复制粘贴当前小节；输入框获得焦点时仍保留系统复制粘贴。

右侧“导出”面板可以下载：

- `LAB`：`start end chord` 标注，可作为人工参考答案或外部模型候选再次输入
- `JSON`：带完整小节、拍子、事件信息的校正版本

校正后可以用评测脚本比较下一版模型是否真的进步：

```powershell
python .\app\evaluate_chords.py ".\annotations\song_corrected.lab" ".\projects\song_analysis.json"
```

只分析某一段：

```powershell
python .\app\chord_studio.py "song.m4a" --offset 90 --duration 60
```

如果开头有人声、噪声或长前奏，可以从音乐主体开始分析：

```powershell
python .\app\chord_studio.py "song.m4a" --offset 16 --key D --meter 4/4
```

如果开头是纯鼓组或没有明确和声，不要截掉音频；可以保留小节但标为空和弦：

```powershell
python .\app\chord_studio.py "song.m4a" --blank-intro-bars 4
```

默认模型会自动使用流行歌功能和声先验，例如倾向于 `I - V/7 - vi - IV - V - I`、`IV - V - vi - IV - V - I` 这类路径；低音判断也会同时考虑根音和转位低音。也可以切回旧的局部模板模型做对照：

```powershell
python .\app\chord_studio.py "song.m4a" --analysis-model template
```

如果已经知道歌曲常用进行，可以在旧模板模型里作为人工提示传入。例如 D 大调中 `176451` 会按流行歌常见写法倾向为 `D - A/C# - Bm - G - A - D`：

```powershell
python .\app\chord_studio.py "song.m4a" --analysis-model template --key D --meter 4/4 --progressions 176451,456451
```

如果已经用 Chordino、madmom、BTC 或其他自动和弦识别模型生成了时间轴和弦，可以作为外部候选融合进默认功能和声模型。文件支持 `.lab`/`.txt` 的 `start end chord`，CSV 的 `start,end,chord`，以及含 `events` 或 `chords` 的 JSON：

```powershell
python .\app\chord_studio.py "song.m4a" --external-chords ".\model_outputs\song.lab" --external-weight 0.35
```

外部模型的常见标签会被自动归一化，例如 `C:maj` -> `C`、`A:min` -> `Am`、`G:7` -> `G7`。`--external-weight` 是软权重，建议先在 `0.25` 到 `0.50` 之间比较；权重太高会让外部模型的局部误判压过段落结构。

`--external-chords` 可以重复传入多次，用于多模型融合：

```powershell
python .\app\chord_studio.py "song.m4a" --external-chords ".\model_outputs\song_madmom.lab" --external-chords ".\model_outputs\song_btc.lab" --external-weight 0.30
```

当前机器上已经额外安装了一个独立的 madmom 环境：`C:\Users\33573\.codex\acr_envs\madmom39`。如果需要重建它，可以运行：

```powershell
.\setup_madmom_env.ps1
```

madmom 在当前 Windows 环境里不能直接读取 m4a，所以项目提供了一个桥接脚本：先用主环境解码音频，再把 numpy 音频交给 madmom 的 CNN+CRF 和弦识别器输出 `.lab`。

```powershell
python .\app\madmom_bridge.py "song.m4a" -o ".\model_outputs\song_madmom.lab"
python .\app\chord_studio.py "song.m4a" --external-chords ".\model_outputs\song_madmom.lab" --external-weight 0.35
```

Chordino 可以通过 Vamp Plugin Pack 和 Sonic Annotator 输出和弦时间轴；BTC-ISMIR19 的 `test.py` 也会输出 `.lab` 文件。它们的结果不需要改主程序，直接作为 `--external-chords` 输入即可。

更简单稳定的三和弦识别：

```powershell
python .\app\chord_studio.py "song.m4a" --mode triads
```

更多复杂和弦：

```powershell
python .\app\chord_studio.py "song.m4a" --mode rich
```

降低贝斯对和弦判断的影响：

```powershell
python .\app\chord_studio.py "song.m4a" --bass-weight 0.05
```

让和弦变化更自由，减少时序平滑：

```powershell
python .\app\chord_studio.py "song.m4a" --transition-strength 0.05 --smooth 1
```

## 准确度建议

这个工具使用 chroma 音级特征、低频贝斯根音估计、真实 beat 时间追踪、半速/倍速候选比较、小节相位候选、调式联合评分、相对大小调中心复核、功能和声转移矩阵、常见流行歌进行先验、两小节短语位置先验、重复段候选共识、重复段一致性细化和 Viterbi 序列推断。声学打分会同时考虑和弦音覆盖率、根音/三音/五音支撑、非和弦音比例和低音根音支撑。它适合生成扒歌初稿，但复杂编曲仍建议人工校对。

如果是一整场演出或歌曲合集，先切成单曲再识别，准确度会明显更好。若 BPM 或调式识别偏了，先用 `--bpm`、`--key`、`--meter` 修正后再生成一次。

如果 HTML 高亮整体早了或晚了几十到几百毫秒，打开 `_studio.html` 后调顶部“同步”滑块即可；这个校准会按当前音频记住。

如果识别结果反复出现 `vii°`，但实际是流行歌里常见的低音下行 `I - V/7 - vi`，默认会把它清理成属和弦转位；若确实想保留严格的减七级和弦，可加 `--strict-diminished`。

如果大调里出现 `ii/IV` 这类与 `IV6` 高度近似的结果，默认会优先按流行谱习惯清理为 `IV`，尤其是后面接 `V` 的时候。例如 G 调里 `Am/C - D` 会更倾向输出为 `C - D`。

如果一首歌虽然与大调共享音阶，但主体明显以相对小调的 `vi` 为中心，默认会自动重跑相对小调模型。例如 A 大调集合里反复出现 `F#m - D - E - F#m`，会更倾向标为 F#m 小调的 `i - VI - VII - i`。小调里的 `iv/VI` 歧义也会优先清理为 `VI`。

如果某些和弦明显被低音经过音带偏，可以调低 `--bass-weight`。如果和弦变化被压得过于保守，可以降低 `--transition-strength` 或 `--smooth`。

## 评测

如果有人工和弦标注，可以用评测脚本量化准确率，而不是只凭肉眼判断。标注支持 `.lab`/`.txt` 的 `start end chord` 格式，也支持包含 `start,end,chord` 列的 CSV：

```powershell
python .\app\evaluate_chords.py ".\annotations\song.lab" ".\projects\song_analysis.json"
```

脚本会输出根音准确率和简化三和弦准确率。后续如果要训练神经网络，建议先用这个脚本固定评测集，再比较每次模型改动的真实收益。

## Benchmark 评测集

当前目录已经建立了 `benchmarks\` 工作区，用来逐步接入公开数据集和本地人工校正样本：

- McGill Billboard：主要评测 chord / root / triad，并统计常见和声转移
- Isophonics：评测 chord + beat/downbeat + key，最适合完整系统
- Ballroom：专门评测 tempo / beat / bar
- SALAMI：后续用于段落结构和 progression context
- RWC-Pop：有访问权限后用于流行音乐泛化测试

批量评测入口：

```powershell
python .\app\run_benchmark.py .\benchmarks\manifest.csv -o .\benchmarks\reports\latest
```

单独评测 beat/downbeat：

```powershell
python .\app\evaluate_beats.py ".\annotations\song.beats" ".\projects\song_analysis.json"
```

`benchmarks\manifest.example.csv` 是模板。公开数据集的音频多数有版权限制，所以这里只管理本地路径和标注，不把音频打包进项目。

如果要扩大端到端评测集，优先按 `benchmarks\audio_sources\isophonics_beatles_album_checklist.md` 下载整张专辑音频；完整逐曲清单在 `benchmarks\audio_sources\isophonics_beatles_all_audio_checklist.md`。

## POP909 中文流行评测

当前已经接入 POP909 的 chord / beat / key 标注，并生成了 909 首 MIDI 合成音频用于流程验证：

```powershell
python .\app\import_pop909.py .\benchmarks\raw\pop909\POP909 -o .\benchmarks\manifest.pop909.csv --import-dir .\benchmarks\imported\pop909
python .\app\render_pop909_midi_audio.py .\benchmarks\manifest.pop909.csv .\benchmarks\raw\pop909\POP909 --audio-dir .\benchmarks\audio\pop909_synth --output-manifest .\benchmarks\manifest.pop909_synth.csv
```

合成音频只适合做 pipeline smoke test 和和声先验统计，不代表真实中文流行音频表现。前 100 首 reference beat/key 测试结果在 `benchmarks\reports\pop909_synth_first100_reference_grid.json`：Root 约 0.683，Triad 约 0.598。

如果已经自行取得 POP909 对应的真实音频，把文件放到一个目录里，文件名里包含 `001`、`pop909_001`、歌名或歌手即可自动链接：

```powershell
python .\app\link_pop909_audio.py .\benchmarks\manifest.pop909.csv ".\benchmarks\audio\pop909_real" -o .\benchmarks\manifest.pop909_real.csv
```

然后用真实音频 manifest 重新生成预测和评测：

```powershell
python .\app\generate_predictions_from_manifest.py .\benchmarks\manifest.pop909_real.csv --use-reference-key --use-reference-beat-grid --feature-mode fast --harmonic-unit auto
python .\app\run_benchmark.py .\benchmarks\manifest.pop909_real.csv -o .\benchmarks\reports\pop909_real_reference_grid
```
