# Chord Studio：本地和弦识别与标注工作台

> **先读**：`D:\ChordStudio\README.md`（当前冻结模型、目录地图、归档说明）。
> 现在在用的模型是 **`chord_retained10_v80`（音频入口 v88）**，权重 `retained10.npz`，
> sha256 `6a37737d…eceef`，随桌面版安装包发布，不放在本仓库里。
> 本仓库的 `benchmarks\models\_legacy_202608\` 全是历史实验模型，不要混淆。

分析本地音频，生成和弦、小节/拍点、调性及可播放、可修改的离线页面。当前实现、可信性能与优先瓶颈统一见 [PROJECT_REPORT](PROJECT_REPORT.md)。本文只保留使用方法，不再叠加历史审计日志。

## 选择入口

| 任务 | 入口 | 输出 |
|---|---|---|
| 网页上传音乐 / B 站链接一键识别 | `start_harmony_web.bat` | 常驻单页工作台、任务进度、分析 JSON、独立修正 LAB |
| 分析新歌曲并打开工作台 | `recognize_harmony.bat` / `app/analyze_with_madmom.py` | 分析 JSON、HTML、和弦表 |
| 打开已有分析并继续人工标注 | `open_chord_studio.bat` / `app/open_studio.py` | 当前样式的工作台，不重识别 |
| 使用三源候选实验模型 | `app/recognize_harmony.py` | 带来源记录的 `harmony.lab` |
| 单独使用大词表识别器 | `app/generate_audio_only_largevocab_labs.py` | Harte 和弦 LAB |

**注意两个同名入口不同：根目录 `.bat` 是默认工作台；`app/recognize_harmony.py` 是实验模型。** 新候选模型尚未替换工作台默认路线。

### 离线主调前三候选排序实验

新增 `app/train_key_top3_reranker_v1.py` 是独立的回顾性开发训练入口，比较原 CNN（M0）、CNN 特征 LR（M1）、增加全曲和声特征的 LR（M2）。固定 CNN 前三候选、5 外折/3 内折，使用已有冻结缓存，不启动声学模型或工作台自动覆盖。实际成绩、误改及限制见 [项目报告](PROJECT_REPORT.md)；模型输出为未经校准的排序分数。

已实际运行的训练/评价命令：

```powershell
python -B -m app.train_key_top3_reranker_v1 run --output benchmarks/reports/key_top3_rerank_20261006_v1
```

已完成的目录拒绝覆盖。重现时指定新的输出目录，先准备/冻结，再运行同一个小网格；不修改旧 protocol：

```powershell
python -B -m app.train_key_top3_reranker_v1 prepare --output benchmarks/reports/key_top3_rerank_reproduction
python -B -m app.train_key_top3_reranker_v1 run --output benchmarks/reports/key_top3_rerank_reproduction
python -B -m pytest tests/test_key_top3_reranker_v1.py -q
```

准备需要本机 `key_review_data_20261006` 的用途登记、`harmonic_key_v2_20261006` 的 contract/四份完整预测报告及其指向的原音频、CNN/LAB/中性网格。缺失或身份不符会列入新 manifest；无有效材料则报真实错误，不重新下载或伪造缓存。历史源码与模型须保持原位。

应用已训练的离线模型：

```powershell
python -B -m app.key_top3_reranker_v1 --model benchmarks/reports/key_top3_rerank_20261006_v1/models/M2_full.json --cnn cnn.json --harmonic harmonic_v2.json --output ranked_key.json
```

`--harmonic` 可省略，缺失来源保留有效性掩码；`M1` 只使用 CNN 字段。应用入口校验特征/实现与冻结上游 contract，无法替代新歌曲输入来源身份的核对。本轮《虚拟》两模型均仍选 G，没有上线默认策略。

## 日常使用

### 浏览器工作台（新增）

双击 `start_harmony_web.bat`，打开 **http://127.0.0.1:8765**。选择或拖入音频，立即试听，然后点击“开始分析”；也可以切换到“B 站链接”，输入 `bilibili.com/video/…` 或 BV 号，点击“获取音频并分析”。支持 `?p=2` 等分 P 参数，不自动下载整个合集。网页显示实际执行阶段和运行日志；任务后台串行执行，关闭页面后仍可从“我的音乐”重新打开。

新页面默认明确选择“多来源和声 · 冻结 v1”，也可选择单来源 madmom 或显式的实验 v2。原来的离线工作台默认路线保持不变。v2 B52 与 v1 B66 训练规模不同，来源一致性不代表准确率。

结果包括波形、拍点/小节、主调、段落相似性、和声循环、各来源原始区间和候选提案。主调可手动修改，和弦名称与级数可切换；编辑框支持 `C#m7`、`V7`、`6m` 等记法。候选来自有时间重叠的独立声学来源，不显示没有导出的模型分数。A/B 是相似段落标记，不能当作已确认的主歌/副歌。

结果页可独立打开“和声主调 v2 · 辅助审阅”。原自动主调与冻结全曲建议并列显示；开关不会自动覆盖，原 `fused` 判断也不会换算法。点击“采用此建议”或选择自定义主调会立即保存人工审阅，跨来源统一级数，重新打开仍保留；“恢复自动估计”恢复原自动结果。原始和弦及人工和弦修订不移调。局部候选默认折叠为未验证，仅用于定位播放。缺少匹配原始音频/LAB/中性网格或契约失效时显示不可用，不抄用别的歌曲结果。显示改调不重新识别音频。

“主调审阅 JSON”导出运行级自动结果、建议、人工选择和历史事件，是新增人工改调的权威记录。“分析 JSON”继续导出不可变的原始模型结果；各和弦来源的旧修改 JSON 中保留的主调仅为当时显示设置，不能替代较新的运行级审阅记录。

“先听辨再看建议”入口提供独立标注页：先填写录音版本、时段、主音/可未知的大小调族、依据及标注者，锁定后显式查看建议。预测曝光和修订留痕；此功能只隐藏本页预测，不能证明从未见过其他预测，也不会把人工采用建议当成金标准。导出标注 JSON 可通过 `python -B -m app.import_key_annotations --input key_annotation.json --output imports/listening_001.jsonl` 导入；JSONL/CSV 同样支持，输出拒绝覆盖。数据用途和待补材料见 [数据协议](benchmarks/reports/key_review_data_20261006/DATA_PROTOCOL.md)。

点击“应用”更新浏览器草稿，再点击“保存修改到项目”写入本机。各来源分别保存修改，原始 LAB 不覆盖；修正 LAB 保留原起止时间文本。导出链接读取上次服务器保存的版本，“导出当前草稿”可保存尚未提交的修改。两个窗口同时保存会检测版本冲突。新分析使用新的运行目录，旧输出保留，旧修改不会自动套用到新边界。

首次在其他环境安装：

```powershell
.\setup_harmony_web.ps1 -ModelPython "C:\路径\到\现有模型环境\python.exe"
.\start_harmony_web.bat
```

Web 依赖装在独立的 `.tmp/harmony_web_env`，模型继续使用原 Python 环境及原 madmom/Chordino/大词表工具链。脚本记录模型解释器位置；也可通过 `HARMONY_MODEL_PYTHON` 覆盖。模型权重和 contract 仍需原位保留，安装 Web 依赖不会下载训练数据或重训模型。

开发启动与针对性验证：

```powershell
.tmp/harmony_web_env/Scripts/python.exe -B app/start_web.py --port 8765
.tmp/harmony_web_env/Scripts/python.exe -B -m pytest tests/test_web_workbench.py -q
# 已完成项目的 UI 验证，不重跑声学模型：
.tmp/harmony_web_env/Scripts/python.exe -B -m app.web.check_browser <project_id>
```

服务绑定 `127.0.0.1`，单进程队列；默认接受不超过 512 MB、30 分钟的音频。B 站仅获取公开可访问的音频流，不自动读取浏览器登录信息；不可访问的视频会给出失败状态，可改用本地上传。服务器重启时中断任务标记为可重试，不冒充继续运行。

代码边界为 `frontend/` 固定页面、`app/web/server.py` HTTP 接口、`jobs.py` 任务队列、`worker.py` 现有 CLI 编排。模型不生成页面。接口包括 `/api/upload`、`/api/bilibili`、`/api/projects/{id}/analyze`、`/api/jobs/{id}`、项目 `result`/`audio`/`corrections`/`export`。写接口需要 `X-Harmony-Client: studio`，页面已自动发送。

数据保存在 `projects/web/<project_id>/`：音频与 `project.json`；每次分析位于 `runs/<job_id>/`，包含 `analysis.json`、各来源 LAB、模型来源侧车及 `corrections/<source_id>/corrections.json` / `corrected.lab`。任务命令、耗时、失败和日志位于 `projects/web/jobs/`。统一 JSON 使用 `sources[].rows` 保存完整区间，以 `default_source` 标记当前模型，另含 `key_analysis`、`beats`、`bars`、`waveform`、`structure` 和输入/模型哈希。

实现使用 [FastAPI 文件上传](https://fastapi.tiangolo.com/tutorial/request-files/)、[Starlette 文件及范围响应](https://starlette.dev/responses/)和 [yt-dlp](https://github.com/yt-dlp/yt-dlp) 的公开接口。

### 独立和声主调 v2 与 chroma 诊断（2026-10-06）

这两个入口输出独立 JSON；默认工作台仍使用原 CNN/冻结裁决器。v2 将主音与条件大小调分开，保留 12 主音、各主音的两个调式、24 个组合候选及局部窗口证据。分数不是校准概率。

```powershell
python app/harmonic_key_v2.py --raw raw.lab --neutral neutral_analysis.json --contract benchmarks/reports/harmonic_key_v2_20261006/engine_contract.json --output harmonic_v2.json
python app/key_profile_baseline.py song.mp3 --output chroma_profile.json
```

`raw.lab` 必须是独立原始和弦，`neutral_analysis.json` 来自不带 `--key`、带 `--all-key-candidates` 的分析。原始音频和这些输出必须对应同一录音。chroma 使用已有 22050 Hz / hop 512 前端，缓存核对音频及实现哈希。省略 `--contract` 仅运行未选模的工程默认规则，不能将其结果混入冻结实验。

规则、schema 或特征语义改变后必须新建 contract，不能沿用已冻结制品。开发集选择只读取新开发/校准分区，已有 contract 不允许覆盖。新封存、消融、旧歌曲回归与互补 oracle 见 [PROJECT_REPORT](PROJECT_REPORT.md) 和 [交付索引](benchmarks/reports/harmonic_key_v2_20261006/delivery.json)。oracle 是离线诊断，不参与推理；本轮没有重训或启用 disagreement resolver。

### 实验主调融合（2026-10-06，保留历史实现）

网页“主调判断”选择 **分歧裁决 · 实验**，或在 CLI 使用：

```powershell
python app/analyze_with_madmom.py "song.mp3" --key-source fused -o projects/song_fused
```

先读取独立原始和弦与 CNN 证据，运行不带 `--key` 的独立分析，再检查全曲和声及多小节稳定性。若最终主调与独立分析不同，最多再运行一次；人工 `--key` 始终优先。网页的和弦来源本身与主调无关，只需一次独立调性分析。此模式不强制添加 Chordino/大词表。

页面显示 CNN 与全曲和声两路主证据，以及联合分析、局部时间线等辅助证据。主证据一致且质量检查通过时接受；有分歧时才运行两个小型逻辑回归 gate，只有一方通过校准门槛才裁决。没有兼容模型、两方均通过或均未通过时为 `ambiguous`，显示的是**暂定主调**。可判定小节占比、未判定占比、候选在可判定部分的占比、全部有效小节支持率分别展示。

输出使用 `key_fusion`、`key_source=key_disagreement_resolver_v2` 与 `key_estimation.madmom_cnn`。旧 CNN 原始证据保留；逻辑回归输出是判别分数，不是有可靠误差保证的概率。四种路径为 `agreement`、`disagreement_resolved_cnn`、`disagreement_resolved_harmonic`、`ambiguous`。旧门槛仅保存在 `legacy_policy`/`legacy_gate_checks` 供追溯，不参与新裁决。

默认仍为 CNN；完整对照、回归和部署判断见 [PROJECT_REPORT](PROJECT_REPORT.md#实验主调融合2026-10-06)。旧项目不会自动重算或覆盖人工标注。实验模型默认读取 `benchmarks/models/key_resolver_v2.json`；CLI 可用 `--key-resolver 路径` 指定冻结制品。实现哈希或 CNN 权重不匹配时，必须重新生成证据并训练/校准，不能沿用旧门槛。

本轮制品的校准分歧样本中没有和声纠错正例，两个自动裁决门槛均禁用；当前只能接受质量合格的主证据一致结论，分歧继续标为待审核。《虚拟》尚未自动改成 C，不建议将 fused 设为默认。

离线评估入口：

```powershell
python app/evaluate_global_key_fusion.py --manifest key_eval.csv --output benchmarks/reports/key_eval
```

CSV 字段为 `id,cnn,neutral,raw,reference`，可加 `audio` 记录音频哈希；路径相对 CSV 所在目录。`neutral` 必须来自不带 `--key`、带 `--all-key-candidates` 的 studio 分析；`raw` 必须是与主调无关的原始 LAB。参考只参与最后评分，不进入推理。默认标记为回归集，`--dataset-kind held-out` 仅适用于调用者已经独立封存的数据，程序不会替代数据隔离审查。

分歧裁决器实验入口（训练只读取 development/calibration）：

```powershell
python app/download_key_resolver_data.py --output benchmarks/external_key_data
python app/prepare_key_resolver_data.py --root benchmarks/external_key_data --output benchmarks/reports/key_resolver_20261006/splits --limit 180
python app/run_key_resolver_audio.py --split benchmarks/reports/key_resolver_20261006/splits/giantsteps_split.json --output benchmarks/reports/key_resolver_20261006/giantsteps --workers 2
python app/train_key_disagreement.py --split benchmarks/reports/key_resolver_20261006/splits/giantsteps_split.json --cases benchmarks/reports/key_resolver_20261006/giantsteps --output benchmarks/reports/key_resolver_20261006/resolver.json
python app/evaluate_key_disagreement.py --split benchmarks/reports/key_resolver_20261006/splits/giantsteps_split.json --cases benchmarks/reports/key_resolver_20261006/giantsteps --model benchmarks/reports/key_resolver_20261006/resolver.json --output benchmarks/reports/key_resolver_20261006/sealed_test.json
python app/evaluate_key_disagreement.py --manifest key_eval.csv --model benchmarks/reports/key_resolver_20261006/resolver.json --output benchmarks/reports/key_resolver_20261006/regression.json
python app/evaluate_billboard_key_reasoning.py --split benchmarks/reports/key_resolver_20261006/splits/billboard_split.json --output benchmarks/reports/key_resolver_20261006/billboard/report.json
```

已冻结 split 与模型拒绝覆盖；新实验请使用新目录。`run_key_resolver_audio.py --evidence-only` 只从校验通过的缓存重建和声证据，不会重新运行声学模型。GiantSteps+ 是公开音频的电子音乐试点；按艺术家/作品分组的 resolver 隔离不等于已验证 CNN 上游预训练完全无重叠。Billboard 入口只用公开参考和弦及插值小节测试主音推理，不是音频识别或完整大小调准确率。

### 原有离线入口操作

把音频拖到 `recognize_harmony.bat`，或执行：

```powershell
python app/analyze_with_madmom.py "song.mp3" -o projects/song --open
```

已有分析 JSON 可拖到 `open_chord_studio.bat`；重新打开时无需重跑模型：

```powershell
python app/open_studio.py "已有_analysis.json" --open
```

也可直接打开[现有歌曲工作台索引](benchmarks/reports/session_20260926/index.html)。页面支持播放跟随、波形定位、A/B 循环、变速、标签编辑、撤销/重做、段落建议及导出修订 JSON。浏览器本地草稿不等于项目文件，重要修改请使用页面导出功能保存。[工作台操作细节](docs/studio_ui_20260926.md)。

标签保真修复适用于新生成的页面。已有 HTML 内嵌的是旧版脚本；要使用新版编辑导出，请先保存修订，再用 `open_chord_studio.bat` 打开相应分析 JSON，无需重识别音频。

默认使用 madmom 的和弦、调性与拍格；轻量结构模块提供中性段落和循环建议。可显式传入已知调性、拍号，例如 `--key E --meter 4/4`。参考拍点或人工调性参与的结果应标为辅助分析，不能算完全自动识别成绩。

## 实验模型

```powershell
python app/recognize_harmony.py "song.mp3" --model benchmarks/reports/session_20260926_accuracy2/multisource_candidate_reliability/selected_train66_head.joblib --output-dir projects/song_candidate
```

此入口串行运行 madmom、Chordino、ISMIR2019 大词表模型，并用 NNLS 音符证据和 HGB 模型选候选。默认保护原始根音与时间边界；`--no-preserve-root` 才允许完整候选头修改根音。输出为 `harmony.lab`、来源侧车及 `result.json`，不是已验证的工作台最终融合结果。

模型文件旁的 `.contract.json` 必须保留。已有三源输出时可传 `--raw-lab`、`--chordino-lab`、`--largevocab-lab` 复用；特征缓存按音频和工具身份核验。详细成绩、限制与单独应用模型的命令见[模型实验说明](docs/model_accuracy2_20260927.md)。

独立大词表入口：

```powershell
python app/generate_audio_only_largevocab_labs.py --audio "song.mp3" --output-dir projects/song_largevocab
```

结果在 `projects/song_largevocab/labs/audio.lab`。工作台导入已保留显式 slash 低音和未知标签 X；Harte 的相对低音会转为音名，如 `G:maj/3 → G/B`。后续拍格融合仍可能改变区间或标签，页面结果不能直接继承原始 LAB 的准确率。修复对照见[项目报告](PROJECT_REPORT.md)。

## 独立 v2 实验入口

v2 不接替上述 frozen v1 和工作台入口。新增源码、139 维 schema、权重和 contract 单独保存；源码或特征语义改变后必须重新训练并重新生成契约。当前可用的 `semantic_e1_B52_v2.joblib` 只用于已知数据诊断，分数未经置信校准。结果与限制见 [PROJECT_REPORT](PROJECT_REPORT.md#本轮-v2-实施与实际结果)。

使用现成音频、三源 LAB 和整曲 NNLS NPZ，应用已训练的 v2 头；不读取参考、不自动提取缺失缓存：

```powershell
python app/apply_multisource_candidates_v2.py --audio song.mp3 --raw raw.lab --chordino chordino.lab --largevocab largevocab.lab --notes notes.npz --model benchmarks/reports/model_v2_20261003/semantic_e1/semantic_e1_B52_v2.joblib --output projects/song_v2/harmony.lab
```

以上 `song.mp3` 等是待替换的实际输入路径。输出须为新路径，同时保存来源侧车。E1 可执行复现实验如下，已有目录不会覆盖：

```powershell
python app/run_semantic_experiment_v2.py --baseline-report benchmarks/reports/model_v2_20261003/policy_diagnostic/dev14/report.json --output benchmarks/reports/semantic_e1_repeat
```

E2 掩码目标训练需先准备新的分组 manifest：JSON 行包含 `id/work_id/artist_id/recording_id/domain/grouping_evidence/split`，以及 `audio/raw_madmom/chordino/largevocab/notes/reference` 六条输入路径；相对路径以 manifest 所在目录为基准。`split` 为 `fit/development/calibration/sealed_test/known_regression`；同一作品、艺术家、录音或音频哈希不得跨组，fit/development 均须有至少两个有来源说明的风格。配置 JSON 必须显式提供七项 `weights`、`margin`、`reference_coverage_policy`，没有默认调权。本轮没有这套新的完整数据，也没有运行下列训练：

```powershell
python app/experiment_protocol_v2.py freeze-splits --manifest grouped_manifest_v2.json --output frozen_splits_v2.json
python app/experiment_protocol_v2.py freeze-config --protocol frozen_splits_v2.json --configuration selection_config.json --output frozen_config_v2.json
python app/train_multisource_candidate_reliability_v2.py --protocol frozen_splits_v2.json --configuration frozen_config_v2.json --baseline-v1-model benchmarks/reports/session_20260926_accuracy2/multisource_candidate_reliability/selected_train66_head.joblib --output benchmarks/reports/masked_e2
```

冻结步骤会给所有输入建立 SHA；封存参考只作不透明字节哈希，不解析标签。训练入口仅读 fit/development 的参考，保留校准与封存组，输出五类对照、字段覆盖、误改时长及退步曲目。它拒绝 frozen v1 已见训练音频作为新评估音频，但外部预训练集完整来源仍需另行核验。

可选六组原声学概率导出与同 raw 边界聚合：

```powershell
python app/generate_audio_only_largevocab_labs_v2.py --audio song.mp3 --output-dir projects/song_posteriors --export-posteriors
python app/aggregate_largevocab_posteriors_v2.py --posteriors projects/song_posteriors/posteriors/audio.npz --raw-lab raw.lab --output projects/song_posteriors/raw_aggregated.json
```

NPZ 包含 triad/bass/7/9/11/13 概率、类别映射、帧采样位置、采样率/hop 与完整来源哈希。省略 `--export-posteriors` 保持原 LAB 推理路线；缓存模式、音频、权重或前端不符会拒绝复用。聚合另支持 `--anchors largevocab` 或 `--anchors union` 配合 `--largevocab-lab`，用于分别研究边界因素，不会改写概率来源名称。目前只完成 8 秒片段的实现验证，没有后验增强模型准确率结论。

## 环境

本机现有环境已安装，无需因文档整理重新安装。迁移到新电脑时，主环境依赖：

```powershell
python -m pip install -r requirements_chord_studio.txt
```

- madmom 使用独立 Python 3.9 环境。`setup_madmom_env.ps1` 要求本机预先安装 Python 3.9，默认环境位于用户目录 `.codex/acr_envs/madmom39`。
- 三源实验额外依赖本地 Sonic Annotator、NNLS Chroma Vamp 插件、PyTorch 与 `tools/models/` 内的 ISMIR2019 源码和五个权重。不要将其 `cache_data/*.sdict` 当作可删缓存。
- 已验证的实际包版本保留在 `benchmarks/reports/session_20260926_accuracy2/runtime_environment.json`；仅装通用 requirements 不代表已装齐所有可选后端。
- 可选 All-In-One 段落专家使用 `.tmp/structure_env`；该目录是环境，不能按临时文件清空。首次安装可运行 `setup_structure_env.ps1`，然后使用 `--structure allin1`。其主歌/副歌仅作建议，不自动替换人工标注或现有拍格。

## 文件与维护

- `app/`：当前入口、公共模块、仍有调用链的研究工具。
- `app/studio_assets/`：离线页面模板、样式、交互。
- `benchmarks/audio/`、`raw/`、`references/`、`imported/`：音源与参考材料。
- `benchmarks/models/`、`tools/models/`：模型资源。
- `benchmarks/reports/`：关键评估、展示页，以及因旧路径依赖仍原位保存的训练候选 CSV；不能整目录当日志删除。
- `projects/` 与人工修订 JSON：用户工作成果。
- `.tmp/nnls_seventh_features/` 等：可被训练报告引用的特征证据，不是全部可以丢弃。

本次整理归档了历史说明、一次性实验脚本和旧自动输出，清除了已完成结构分析的中间音轨/频谱；原音频、人工标注、模型、最终结果和当前页面路径均保留。历史文件在 `archive/cleanup_20261003/history.zip`，原路径、SHA 和动作见[清理清单](archive/cleanup_20261003/manifest.json)。恢复个别旧文件的方法见 [PROJECT_REPORT](PROJECT_REPORT.md#历史内容与恢复)。

维护检查按改动范围执行，不例行重复全套测试、声学推理或旧预测逐字节复现。
