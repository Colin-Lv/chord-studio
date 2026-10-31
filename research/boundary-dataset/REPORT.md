# POP909 和弦切换边界数据集

用 POP909 真实歌曲的**真实和弦标注**与**真实拍/小节网格**合成“边界精确已知”的
带标签 MIDI + 音频数据集，用于训练和弦切换边界检测器（同时可直接用于和弦识别）。

所有标签来自 POP909 标注，**没有任何自编和声进行**；所有音高都由和弦音集合生成，
`check.py` 复核为 0 错音。

---

## 1. 规模一览

| 项 | 值 |
|---|---|
| 歌曲数 | 48（26 大调 / 22 小调，24 个调性各≥3 首） |
| 样本数 | 144（48 首 × 3 种织体） |
| 音频总长 | 标签时间轴 6189.9 s ≈ 103.2 min；wav 实际 6305.3 s ≈ 105.1 min（含尾音）。48 kHz / 单声道 / PCM16，605 MB |
| 单条时长 | 34.0 – 50.0 s，均值 43.0 s |
| 速度范围 | 57.3 – 124.0 BPM，均值 89.7（慢 45 / 中 48 / 快 51 条） |
| 正例边界 | 3267（平均 22.7 条/样本） |
| 负例点 | 7935（bar 390 / half_bar 1440 / beat 4005 / melody 2100；平均 55.1 个/样本） |
| 正:负 | ≈ 1 : 2.4 |
| 音符总数 | 33622（和弦声部 31522，旋律 2100） |
| 加旋律的样本 | 44（30.6 %，block 15 / arpeggio 15 / strum 14） |
| MIDI / 标签 / 音频 | 144 / 144 / 144 个文件 |

## 2. 数据来源（重要：路径与文件与任务描述不一致）

任务给的 `D:\ChordStudio\dialogue\benchmarks\external_chords\pop909` **不存在**。
在 `D:\ChordStudio` 下找到两份 POP909，都只有部分文件：

| 位置 | 内容 | 是否可用 |
|---|---|---|
| `D:\ChordStudio\_archive\2026-10-30\benchmarks\raw\pop909\POP909\<sid>\` | **官方 POP909 完整版**：`beat_midi.txt` `chord_midi.txt` `key_audio.txt` + 原始 MIDI（909 首） | ✅ 实际使用 |
| `D:\ChordStudio\Dialogue\benchmarks\imported\pop909\pop909_<sid>\` | 只有 `chords.lab` + `beats.beats`（909 首），**没有 beat_midi/key_audio** | ❌ 未使用 |

为什么不用 `chords.lab`：

1. 它是**另一套标注**（Harte 风格，含 `Bmaj7`、`Emaj(9)`），与官方 `chord_midi.txt`
   的序列并不一致（例：001 首官方 `B:maj C#:maj Bb:min …`，`chords.lab` 是
   `Bmaj7 C# A#m7 …`，段数 152 vs 123）；混用会导致“标签与拍网格不来自同一套分析”。
2. 它的时间轴是**音频时间**，而 `beat_midi.txt` 是 **MIDI 时间**，两者不可直接配对
   （001 首首个和弦 2.554 s vs 2.722 s）。

因此统一使用**官方 POP909 的 MIDI 时间轴三元组** `chord_midi.txt + beat_midi.txt +
key_audio.txt`：和弦边界与拍点在同一时间轴上**逐样本完全重合**（全库 124 805 个和弦段，
边界到最近拍点的最大残差 5×10⁻⁷ s），量化无损失。`generate.py` 同时支持
`chords.lab`/`beats.beats` 布局作为后备（自动识别，key 缺失时用 Krumhansl 模板推断）。

SoundFont：任务给的 `D:\ChordStudio\research\midi-gen\render\FluidR3_GM.sf2` 不存在，
实际使用 `D:\ChordStudio\assets\soundfonts\FluidR3_GM.sf2`（148 MB，同一份 FluidR3 GM）。

## 3. 目录结构

```
D:\ChordStudio\research\boundary-dataset\
├── generate.py            # POP909 -> MIDI + labels
├── render.py              # MIDI -> 48 kHz 单声道 wav
├── check.py               # 自动校验（错音/对齐/进行合理性/音频）
├── dataset.json           # 样本清单 + 全局统计（144 条）
├── plan_selected.json     # 选中的 48 首及其窗口（起止拍、调、BPM、乐段）
├── render_manifest.json   # 渲染清单（时长、峰值、耗时）
├── check_report.md        # 校验报告（人读，含 10 条随机进行）
├── check_report.json      # 校验汇总（机读）
├── check_details.json     # 每条样本的完整校验明细
├── midi/   pop909-<sid>-<texture>[-mel].mid        144
├── labels/ pop909-<sid>-<texture>[-mel].events.json 144
├── audio/  pop909-<sid>-<texture>[-mel].wav         144  (48 kHz, mono, PCM16)
└── REPORT.md
```

## 4. 生成流程

### 4.1 选歌与窗口（`generate.py`）

- 解析 909 首：拍网格（`beat_midi.txt` 第 1 列时间、第 2 列半小节脉冲、第 3 列小节线；
  兼容 `beats.beats` 的 1..4 拍序号）、和弦段、调性（`key_audio.txt`）。
- 同和弦（同根音+性质+低音，含等音异名）的相邻段合并。
- 在**真实小节线**上枚举 30–58 s 的窗口，要求：
  - 窗口内无 `N`（无和弦）段，逐拍连续覆盖；
  - ≥4 个和弦段、≥3 个不同根音；
  - **不以 V 开头**（`V`/`v`/`V7`… 均拒绝）；
  - 调外根音 ≤2 个且占比 ≤20 %，且必须“可解释”（见 4.4）；
  - 不含调外的减/增和弦；
  - 时长按 `34 + (sid % 5)×4` 秒分档，取评分最高窗口（评分 = 和弦根音多样性 + 时长贴合 − 调外惩罚）。
- 906/909 首存在合法窗口；按 (调式 × 速度档) 分层 + 主音多样性挑选 48 首，
  其中 8 首强制来自“含 1–2 个可解释调外和弦”的窗口（次级属和弦/调式交替），避免全库过于“白”。

### 4.2 量化

和弦边界落在拍点上（POP909 本身如此），生成时按**拍→tick**（960 tick/拍）直接落网格：
小节线/半小节/拍点全部精确；速度取该曲中位拍长（BPM 写为整数 µs/tick，标签时间由
同一 BPM 反算，MIDI 时间与标签时间误差 < 0.05 ms）。

### 4.3 三种织体（均由和弦音构成，绝不出现非和弦音）

| 织体 | 拍点行为 | 说明 |
|---|---|---|
| `block` 柱式 | 每拍同时奏低音+上方和弦音（根音+3~4 音） | 和弦开始时根音与构成音同一下手，钢琴/吉他扫弦式 |
| `arpeggio` 分解 | 拍点奏根音，拍后半拍（+0.5 拍）奏一个和弦音，循环 | 低音持续 + 分解音型，符合“拍点根音、之后分解” |
| `strum` 扫弦 | 拍点起低音，上方音按 7 ms/弦铺开（≤26 ms），换和弦必为下扫 | 换和弦的根音严格落在拍点，附加起音不破坏 30 ms 对齐 |

每拍都重击（正例与负例在**起音能量上完全同构**），检测器必须依赖和声变化而不是起音，
这是本数据集的核心设计。

音色（GM，轮换）：钢琴 0、钢弦吉他 25、尼龙吉他 24、电钢琴 4、弦乐 48。
其中 block/arpeggio 在 5 种音色间轮换（各约 9–10 条），strum 用吉他（钢弦 20 / 尼龙 18）
与钢琴滚奏（10）：

| 织体 | 音色分布 |
|---|---|
| block（48） | piano 10 / nylon 10 / EP 10 / steel 9 / strings 9 |
| arpeggio（48） | EP 10 / strings 10 / nylon 10 / piano 9 / steel 9 |
| strum（48） | steel 20 / nylon 18 / piano 10 |

### 4.4 旋律（30 % 样本，作为干扰源）

- 只在和弦音 + 调内音中取音；调内非和弦音若与任和弦音相距 1 个半音（b9 / 下方导音）则禁用
  → 实际相当于“和弦音 + 五声化调内音”。
- 节奏刻意与伴奏错开：55 % 的拍在**后半拍**（+0.5）起音，18 % 在拍点起音，音长 0.22–1.1 拍。
- 旋律起音全部计入 `negative_points`（`kind="melody"`），用于学习“旋律变、和声不变 → 非边界”。

### 4.5 标签 schema（`labels/*.events.json`）

```jsonc
{
  "sample_id": "pop909-021-block",
  "source_song": "021", "source_dir": "...", "source_chord_file": "chord_midi.txt",
  "key": "Eb:maj", "mode": "maj", "tonic_pc": 3, "scale_pcs": [0,2,3,5,7,8,10],
  "bpm": 57.0001, "time_signature": "4/4", "duration": 37.8947, "beats": 36,
  "texture": "block", "instrument": {"program": 0, "name": "Acoustic Grand Piano"},
  "melody": false, "melody_instrument": null,
  "chords": [{"label":"Ab:maj","start":0.0,"end":2.1053,"start_beat":0,"end_beat":2,
              "roman":"IV","root_pc":8,"root_name":"Ab","quality":"maj",
              "pcs":[0,3,8],"bass_pc":8,"bass_name":"Ab"}, ...],
  "boundaries":     [{"time":2.1053,"beat":2,"from":"Ab:maj","to":"Bb:maj","roman":"V"}, ...],
  "boundaries_sec": [2.1053, 4.2105, ...],
  "negative_points":[{"time":1.0526,"beat":1,"kind":"beat","chord":"Ab:maj","has_onset":true}, ...],
  "progression_roman":  ["IV","V","I","VIsus4", ...],
  "progression_labels": ["Ab:maj","Bb:maj","Eb:maj","C:sus4", ...],
  "n_chromatic_chords": 0, "chromatic_chords": [],
  "stats": {"n_chords":18,"n_boundaries":17,"n_negatives":18, ...}
}
```

- `boundaries` = **正例**：真实换和弦点（秒 + 拍号 + 前后和弦 + 罗马数字）。
- `negative_points` = **负例**：同一和弦内部的**小节线（bar）/ 半小节（half_bar）/ 拍点（beat）**
  以及旋律起音（melody）；`has_onset` 表示该时刻是否真有伴奏起音（本数据集全部为 true）。

### 4.6 渲染（`render.py`）

`render_sf2.py --help` 显示它按**轨道名**（`CHORDS`/`PAD`/`MELODY`…）映射 GM 音色，
且固定 44.1 kHz 立体声、release 取 SoundFont 原值——直接用它会让 5 种音色全部塌缩成钢琴，
且 FluidR3 钢弦吉他声明了 13–25 s 的 release（渲染慢 20 倍、和弦互相糊在一起）。
因此 `render.py`：

- 直接复用同一套采样器 `research/midi-gen/render/sf2.py`（`SoundFont.render`），
  按 MIDI 里每条轨道的 **GM program** 渲染，直接输出 **48 kHz 单声道** PCM16；
- 对每个 program 缓存 zone、把 release 上限截到 0.6 s、音符尾部加 60 ms 淡出（避免爆音）；
- 提供 `--engine cli`：原样调用 `render_sf2.py --no-stems`，再把输出下混 + 重采样到 48 kHz 单声道
  （用于对照验证，默认不走这条）。
- 144 条 / 105 分钟音频渲染耗时 **64 s**（远低于 20 分钟预算，单条平均 0.44 s，最慢 1.24 s）。

## 5. 校验结果（`check.py --audio`）

| 校验项 | 结果 |
|---|---|
| MIDI 文件 / 标签 / 音频缺失 | 0 / 0 / 0 |
| 和弦声部错音（音高 ∉ 当前和弦） | **0 / 31522 个音符** |
| 旋律越界（不在调内且不在和弦内） | **0 / 2100** |
| 旋律半音冲突（b9 / 导音贴和弦音） | **0** |
| 正例边界与 MIDI 起音偏差 | 最大 **0.050 ms**（阈值 30 ms），超阈值 0 条 |
| 负例点与起音偏差 | 超阈值 0 条 |
| 以 V 开头的进行 | **0 / 144** |
| 调外根音 | 24 条样本、27 个和弦：**3 次级属和弦 + 24 调式交替 + 0 无法解释** |
| 非本调性音质（次级属/交替候选，非错误） | 90 条样本（如小调 IV 大、I 大（皮卡迪三度）、大调 III/II/VI 大、小调 v 小） |
| 音频 | 144 文件，采样率全部 48000，全部单声道，0 异常 |

**进行合理性**：全部 144 条进行均以主/下属/六级等非属和弦开头，且每条最多 2 个可解释调外和弦，
无“无法解释”的调外和弦——`check.py` 的 suspicious 列表为空，因此无需剔除样本再渲染。

补充声学交叉验证（一次性抽查 8 条样本、178 个和弦段，FFT chroma）：
和弦音 chroma 能量是非和弦音的 **2.86 倍**（中位 2.73，p10 1.81），说明音频实际发音与标签和弦一致
（音色/音高渲染正确）。

### 10 条随机进行（`check_report.md` 同款，seed=11）

| # | sample | 调 | 织体 | 罗马数字 |
|---|---|---|---|---|
| 1 | pop909-632-arpeggio-mel | Ab:min | arpeggio+mel | i VI v iv v VI v iv v i VII VI III iv i V I |
| 2 | pop909-882-strum-mel | D:maj | strum+mel | III I IV IVsus2 IV VI vi IV ii IV V Isus2 I vi IV |
| 3 | pop909-643-strum | Db:maj | strum | I V vi v I IV iii ii V I **bVI** V I V VIsus2 V IV iii ii vi V I V vi I7 IV |
| 4 | pop909-717-arpeggio-mel | E:maj | arpeggio+mel | I iii vi V IV ii V iii vi iii vi ii I Vsus4 I iii vi V IV ii V I iii vi iii vi |
| 5 | pop909-322-block-mel | E:min | block+mel | III VII v i7 v7 i7 iv VII III VII v7 i VI III VII III IIIsus4 VII III VII |
| 6 | pop909-260-strum | B:maj | strum | ii V I Isus2 ii7 IIsus2 VIsus2 vi ii IIIsus4 III vi Imaj7 ii V V7 Imaj7 Isus2 I ii7 vi IV V vi V |
| 7 | pop909-717-strum | E:maj | strum | I iii vi V IV ii V iii vi iii vi ii I Vsus4 I iii vi V IV ii V I iii vi iii vi |
| 8 | pop909-647-arpeggio | Bb:min | arpeggio | #vi I VI V III i v VI III i7 v IV V i v VI III i v IV |
| 9 | pop909-197-block | C:maj | block | viihdim7 vi7 I V6 V7sus4 IV IVsus2 I I ii7 V7sus4 Vsus2 VIsus2 II7sus4 I Isus2 I III VII7sus4 vi7 v I Vsus2 IV I ii7 Isus4 V Vsus4 IIsus4 ii7 II7sus4 IV V7sus4 Vsus4 V7sus4 I I6 I ii7 I ii7 IV |
| 10 | pop909-632-block | Ab:min | block | i VI v iv v VI v iv v i VII VI III iv i V I |

（明细、全部 144 条进行、每条错误明细见 `check_report.md` / `check_details.json`。）

## 6. 已知问题 / 使用注意

1. **只用官方 `chord_midi.txt` 标注**：`chords.lab` 版本（imported 目录）是另一套更“爵士化”的
   标注且时间轴不同，未使用；因此和弦词汇是 POP909 的大三/小三/7/大7/小7/sus/6/dim/hdim，
   没有 9/11/13。`generate.py` 对未知扩展会退化为三和弦族（仅在后备布局下可能触发）。
2. **sus 和弦会带来“色彩边界”**：POP909 常把一个和声拆成 `sus2→maj` 两段（例 `Eb:sus2→Eb:maj`）。
   本数据集按真实标注保留为两个和弦 → 会产生“根音不变、性质变化”的边界。这类边界是真实标注，
   但比换根音的边界更微妙；若只想训练“根音变化”检测器，可按 `root_pc` 过滤 `boundaries`。
3. **时间完全量化**：所有起音严格落在 1/2 拍网格，无 humanize / swing / 微时序抖动。
   标签因此精确（0.05 ms），但缺少真实演奏的时序噪声；建议训练时自行加时间抖动/重采样增强。
4. **旋律是合成干扰，不是 POP909 原旋律**：为保证“0 错音”，旋律由和弦音+调内音随机游走生成，
   风格简单；它不是原曲旋律，只承担“和声不变但旋律变化”的负例作用。
5. **音色轮换是纹理感知的**：`strum` 只用吉他/钢琴，不用弦乐与电钢，因此总分布
   吉他偏多（钢弦 38 / 尼龙 38 / 钢琴 29 / 电钢 20 / 弦乐 19）。如需严格均匀可改
   `pick_instrument()`。
6. **调性标注偶有可疑**（POP909 `key_audio.txt` 本身的问题）：本数据集用“调外根音 ≤2 且可解释”
   过滤，选中的 48 首没有无法解释的调外和弦；但后备布局（无 key 文件）用模板推断，
   严格程度会下降。
7. **罗马数字是“简单规则”**：按音级查表（大调自然音阶 / 小调自然音阶），调外音用 `b/#` 前缀
   （如小调 `#III`、大调 `bVII`）。等音异名（`Gb:min` vs `F#:min`）按音高类处理。
8. **渲染细节**：FluidR3 的钢琴/吉他/弦乐 release 声明为 0，直接渲染会爆音 → `render.py` 加
   60 ms 尾部淡出；钢弦吉他 release 13–25 s 被截到 0.6 s（否则渲染慢 20 倍且和弦糊）。这些是
   渲染层处理，不改 MIDI/标签。
9. **音频峰值统一归一到 0.89**，没有做响度（LUFS）归一；训练前建议自行做 RMS/LUFS 归一。
10. 只生成 48 首（任务上限 60），可通过 `--songs 60` 扩容；`--chromatic-songs` 控制调外和弦配额。

## 7. 复现

```powershell
$py = "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe"   # 已装 mido/pretty_midi/soundfile

& $py D:\ChordStudio\research\boundary-dataset\generate.py --songs 48 --chromatic-songs 8
& $py D:\ChordStudio\research\boundary-dataset\render.py
& $py D:\ChordStudio\research\boundary-dataset\check.py --audio --n-print 10
```

- `generate.py` 会自动在候选路径里找 POP909（可用 `--pop909-root` 指定）；
- `render.py` 自动在候选路径里找 SoundFont（可用 `--soundfont` 指定），支持
  `--engine cli` 走原 `render_sf2.py`；
- `check.py` 可用 `--prune` 把可疑样本从 `dataset.json` 剔除（本数据集没有触发）。

未修改 `C:\Users\33573\AppData\Local\Programs\Chord Studio\backend` 下任何文件，
也未改动 `research/midi-gen/render/` 下的既有脚本。
