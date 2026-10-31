# 音轨分离 · 验收清单

用途：把「目标里的每一条」和「能证明它的证据」对上，方便你逐条核对或复验。
所有证据都是本机跑出来的，脚本和输出都在仓库里。

## 1. 逐条对照

| 目标要求 | 状态 | 证据 |
|---|---|---|
| 先调研并选定开源分离方案（不重训） | ✅ | `RESEARCH.md` §1–3；对比了 demucs / ft / 6s / BS-RoFormer / SW / Mel-RoFormer，六首以上实测 |
| 后端实现分离 | ✅ | `app/web/stems_worker.py`；GPU 15–75 s/首，失败自动回退 CPU |
| 每轨自动标注（乐器/能量/BPM/调性） | ✅ | 每轨 `rms_db / active_ratio / centroid_hz / register / first_last`；BPM 与小节来自拍点或 librosa；调性取和声分析 |
| 类 FL Studio 网格页面 | ✅ | `/stems`；`shots/final_v2.png`、`shots/drums_expanded.png` |
| 点击播放各分轨 | ✅ | 四/六轨同步，实测 `currentTime` 偏差 0.000 s（`REPORT.md` §2.3） |
| 静音/独奏 | ✅ | M/S 按钮，`aria-pressed` 与音频 `muted` 同步，实测通过 |
| 按小节显示活跃块 | ✅ | 每列一小节、列内 16 格十六分音符；小节上方还有和弦 |
| 显示 BPM 与音轨信息 | ✅ | 运输条 BPM/调性/小节数；左下轨道信息面板 |
| 音轨分离与和声识别是两个按钮 | ✅ | 工作台并排两个按钮；`shots/workbench_home.png` |
| 产出可试听的第一版 | ✅ | 应用在跑；另有 `D:\ChordStudio\data\exports\虚拟_模型对比\` 的 9 个 wav |
| 用户验收分离效果 | ⏳ 等你 | 见下方 5 分钟脚本 |

## 2. 五分钟验收脚本

**A. 应用内（推荐）**

1. 打开工作台 → 《虚拟》→ 点右上模型下拉
2. 选「标准（快）」→ 点副歌某小节 → 勾「循环小节」→ 独奏人声
3. 切「人声优先」→ 同一小节再听
4. 切「六轨精细」→ 独奏「钢琴」「吉他」，听钢琴是否被分到吉他轨
5. 回到「标准」→ 点「细分鼓组」→ 独奏底鼓/军鼓，看是否落在拍上

**B. 播放器里（不想开应用时）**

打开 `D:\ChordStudio\data\exports\虚拟_模型对比\`：

1. 先听 `原曲_30s.wav` 建立参照
2. 依次听四档的 `_人声.wav`，记下哪一档钢琴残留最少
3. 再听四档的 `_伴奏.wav`，记下人声残留最少的那一档

## 3. 已知限制（不要当成 bug）

1. **两轨模型（人声优先）不分离乐器**：没有歌声的混音上，它的「伴奏」就是原曲本身。
2. **六轨模型的钢琴轨偏闷**：合成对照里丢 11% 能量、谱质心低 400 Hz；路由是对的。
3. **单件乐器缺席时归属会错**：只有钢琴的歌，它可能把钢琴放进「吉他」轨。
4. **鼓件里 ride/crash 不可信**：底鼓/军鼓可用，镲片会互相塌。
5. **分离不改善和弦识别**：受控实验里四种输入最多差 2 帧，差距来自下拍相位。

## 4. 复验命令

```powershell
$py = "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe"
# 三档模型的无真值指标（重构残差、跨轨泄漏）
& $py research\separation\eval\stem_eval.py --mix <原曲.wav> --stems "D:\ChordStudio\data\stems\<sha>\htdemucs\wav"
# 评测 harness 自测
& $py -m pytest -q research\separation\eval\tests
# 十首 MUSDB 汇总（需先跑过 musdb_bench.py）
& $py research\separation\aggregate_musdb.py
```
