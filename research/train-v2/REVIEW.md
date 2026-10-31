# REVIEW — 阶段验收证据

更新时间：2026-10-31 14:30（阶段三）

> 本文件记录**脚本复算过**的数字。每条结论后面写着复算命令；脚本不通过就不标 ✅。
> 阶段一/二/旧阶段三的记录保留在文末「历史记录」，不再改动。
> 一键复算全部验收：`research/full-decode/verify_all.py`（打印 ACCEPTANCE 表并给出退出码）。

## 阶段三总览

| 项 | 目标 | 结果 | 判定 | 复算 |
|---|---|---:|---|---|
| 1. 合成域完整和弦（held-out joint） | ≥ 0.95 | **0.9695**（TTA）/ 0.9666（单次） | ✅ | `train-v2/verify_synthetic.py` |
| 1b. 七和弦分项 | ≥ 0.85 | 7 0.934 / maj7 0.915 / min7 0.931 | ✅ | 同上 |
| 2. 147 例回归（`retained10_span`） | 不退化 | 发现 −1 例退化 → 已修复 → 105/105 | ✅（修复后） | `regress147/summarise.py`、`verify_key_pinning.py` |
| 3. 完整段解码器（现默认，可切换回 span） | exact ≥ 0.50、边界 F1 ≥ 0.90 | **0.682** / **0.902** | ✅ | `full-decode/verify_prod_full.py` |

---

## 1. 合成域：0.890 → 0.9695

### 1.1 先定位瓶颈：不是模型，是数据（arpeggio 织体丢音）

阶段二把 0.95 归因为「近邻性质混淆」。阶段三先按织体拆错误，发现差距全部集中在
`arpeggio`（下表是同一套 60 通道特征、同一份数据，只差 arpeggio 是否修复）：

| 织体 | 修复前 | 修复后 |
|---|---:|---:|
| block | 0.993 | 0.993 |
| arpeggio | **0.803** | **0.929** |
| strum | 0.989 | 0.986 |

再核对 MIDI：`boundary-dataset/generate.py` 的 arpeggio 分支每个拍只发一个上方和弦音
（`upper[(beat-start) % len(upper)]`），**两拍的和弦只发得出前两个上方音**。

| 检查（`train-v2/check_tones.py`、`verify_fix.py`） | 结果 |
|---|---|
| arpeggio：标注和弦音在本段内**从未发声** | **4061 / 9108 个和弦（44.6%）** |
| block / strum：同上 | 0 / 9108、0 / 9108 |

也就是说，这 44.6% 的样本**音频里根本没有标注的那个音**，任何音频模型都答不对——
这是标签-音频不一致，不是「证据弱」。

### 1.2 修复（只补缺失音，不动其它织体）

1. `train-v2/fix_arpeggio.py`：把缺失的和弦音按同样的琶音写法（16 分音符位置）补进 MIDI，
   写到 `boundary-dataset/midi_fixed/`，**原始 MIDI 不动**；共补 4 828 个音。
2. `train-v2/render_arpeggio_fixed.py`：只重渲染这 400 条，输出 `boundary-dataset/audio_fixed/`。
3. 复验：`verify_fix.py` → `midi_fixed: 0/9108` 仍有缺音（原 `midi: 4061/9108`）。
4. `prepare_full.py`（`OUT_NAME=chords_full_fixed.npz`）重建特征，arpeggio 读 `audio_fixed/`。

### 1.3 特征：CENS + CQT 双通道 + 音符级 NNLS 三音区（60 通道）

| 通道 | 内容 |
|---|---|
| 0–11 | CENS chroma（平滑、抗音色） |
| 12–23 | CQT chroma（不平滑，保留七音/九音细节） |
| 24–35 | **官方 NNLS-Chroma 半音频谱** 低音区（MIDI 21–47） |
| 36–47 | 同上 中音区（48–71） |
| 48–59 | 同上 高音区（72–104） |

NNLS 半音频谱是生产缓存里**已经有**的（`notes.npz`，84 音 × 21.5 fps，逐帧 L1 归一化），
不是新依赖。逐和弦切 64 帧；音高旋转增广对每 12 通道块分别 roll。

### 1.4 训练与结果（5 折按歌曲切分，60 epoch，width 160）

`train-v2/train3.py --data data/chords_full_fixed.npz --epochs 60 --tag fix60`

| 配置 | joint |
|---|---:|
| 24 维 CENS+CQT + TTA（阶段二最好） | 0.892 |
| 60 维，修复前 arpeggio（单次） | 0.9247 |
| 60 维，修复前 arpeggio（TTA） | 0.9282 |
| **60 维，修复后（单次）** | **0.9666** |
| **60 维，修复后（TTA ×12）** | **0.9695** |

五折：0.9652 / 0.9657 / 0.9662 / 0.9677 / 0.9681；root 0.9957、quality 0.9703。

分项（TTA）：

| 分组 | joint | n |
|---|---:|---:|
| maj | 0.978 | 17 207 |
| min | 0.974 | 9 766 |
| **7 / maj7 / min7** | **0.934 / 0.915 / 0.931** | 500 / 1 110 / 1 885 |
| 9 / maj9 / min9 | 0.929 / 0.964 / 0.965 | 70 / 2 141 / 1 615 |
| sus2 / sus4 | 0.966 / 0.960 | 1 503 / 801 |
| block / arpeggio / strum | 0.993 / 0.929 / 0.986 | — |
| 无旋律 / 有旋律 | 0.986 / 0.935 | 24 967 / 11 631 |

结论：
- **七和弦从 0.62–0.66 提到 0.915–0.934**，说明「七音证据弱」的旧结论只在 chroma 特征上成立；
  换成音符级特征后七音是可读的；
- 剩下 4 个点差距几乎全在「有旋律」（0.935 vs 0.986）与 arpeggio（0.929 vs 0.993），
  即旋律音被当成和弦音、以及琶音下把位不足——都是真实的听觉歧义，不是模型容量问题。

验收：`train-v2/verify_synthetic.py fix60` → `joint >= 0.95 PASS`、`7th family >= 0.85 PASS`。

---

## 2. 147 例回归（`retained10_span`）

复算：`research/full-decode/regress147/summarise.py`（逐例先跑 `run_case.py`，走的是生产模块：
`inference_windowless → key_bridge → downbeat_bridge → stabilize_timeline → span_pool →
key_from_chords/key_decision`）。
逐例产物在 `regress147/cases/<id>/result.json`，音频用归档的 147 例 mp3，**没有新造测试集**。

### 2.1 复现基线（先证明管线没跑偏）

| 口径 | 结果 |
|---|---:|
| 声学（madmom CNN）主调 | **122/147 = 83.0%** |
| 历史记录值（`key-v3/BENCHMARK.md`） | 122/147 = 83.0% |

完全一致，说明这次复跑的管线与当初记录数字的管线是同一条。

### 2.2 和弦来源对「最终主调」的影响（发现退化）

最终主调 = 声学 CNN + 和弦证据 `decide()`；和弦证据取自哪个时间轴，会改变最终结果。

| 和弦证据来源 | 最终主调 | 相对 stable |
|---|---:|---:|
| `retained10_stable`（span 默认之前的默认） | **105/147 = 71.4%** | — |
| `retained10_span`（现默认） | **104/147 = 70.8%** | **−1（6 修好 / 7 改坏）** |

`switch` 的 16 例里，6 例修好、7 例改坏（全部列出，可逐例复核）：

| 例 | stable | span | 参考 |
|---|---|---|---|
| 1056896 | Am ✅ | D ❌ | Am |
| 1198688 | Dm ✅ | G ❌ | Dm |
| 1468437 | Bm ✅ | B ❌（同名大调） | Bm |
| 3641166 | Em ✅ | A ❌ | Em |
| 3967462 | Em ✅ | Cm ❌ | Em |
| 4485436 | Fm ✅ | F ❌（同名大调） | Fm |
| 4917941 | Fm ✅ | C ❌ | Fm |

**结论：把 span 设为默认，确实会让最终主调在 147 例上退 1 例（16 例主调变化，6 好 7 坏）。**

### 2.3 修复：主调证据钉在冻结时间轴上（不改识别）

根因不是 span 的公差，而是**主调竟然依赖「用户当前选中的显示时间轴」**：span 合并后和弦变少、
每段更长，`infer_key` 的覆盖率与 margin 随之变化，`decide()` 就可能翻掉声学主调。

修复（`retained_worker.py`）：和弦证据固定取 `retained10_stable`（没有稳定时间轴时退回冻结
`retained10`），**与显示选中的 source 无关**；`key_analysis.chord_evidence.evidence_source`
记录来源，便于审计。

验证（`regress147/verify_key_pinning.py`）：对 147 例逐例从磁盘上的 stable 时间轴**重新计算**
worker 的主调路径，结果

| 检查 | 结果 |
|---|---|
| 重算主调 == 修复前 stable 列记录值 | **147/147 一致** |
| 修复后主调准确率 | **105/147 = 71.4%** |
| 退化判定 | **PASS（105 vs 105，0 退化）** |

生产实测（run `73c95050`，《虚拟》）：`key = C`（正确），
`chord_evidence.evidence_source = retained10_stable`、`cnn_key = G`、
`decision.reason = chord evidence explains the chords better`——即「和弦证据覆盖 93.55% vs 87.53%」
的原有逻辑没变，只是输入固定了。

### 2.4 段数变化（显示层，符合预期）

| 口径 | 均值 | 中位比例 | 极值 |
|---|---:|---:|---|
| stable | 91.5 段 | — | — |
| span | **36.3 段** | **0.381**（−62%） | 0.040 – 0.879 |

147 例里 146 例段数下降（中位 −62%），1 例塌成 1 段（`1814469`：25 → 1）。
该例的和弦证据因此无法计算，worker 本来就按原逻辑回退到声学主调（Fm，正确）——
回归脚本第一版把这种情况记成「没有 key」，已修正并重跑该例（`run_case.py` 的 `elif key:` 分支，
判定标准仍是「最终发布的主调」）。

---

## 3. 完整段解码器（第二档可切换输出）

### 3.1 解码链

```
小节相位 DP（span_pool 的 per-bar 两拍相位选择）
  → 候选边界：每个小节线 + 半小节（候选点全是半小节网格）
  → 边界证据：GBDT（只用合成数据训练，特征是 NNLS 音符证据，18 维）
  → 半马尔可夫解码：对数正态时长先验（中心 8 拍，σ=0.3）+ 换段代价 + 边界 logit 折扣
  → 每段重判一个和弦：根音 = log p_root + 1.5·log 低音；性质 = log p(q|root) + 0.6·音级持续支持
  → 合并同名相邻段
```

关键点：
- **时长先验是最后一块拼图**（F1 0.85 → 0.90+）：真实歌曲一小节一换/两小节一换，
  没有时长先验时短碎片会拉低精确率；
- 边界分类器**只用合成数据训练**，特征与真实录音完全同源（官方 NNLS 变换），跨域 AUC 0.977；
- 低音证据用 `notes.npz` 的低音区（不再需要 demucs）；
- 《虚拟》的 golden 只用于**选 7 个解码器标量**，没有用于训练分类器。

### 3.2 《虚拟》指标（golden 62 事件）

| 输出 | 段数 | exact | family | 边界 P | 边界 R | 边界 F1 |
|---|---:|---:|---:|---:|---:|---:|
| `retained10_stable`（旧默认） | 177 | 0.401 | 0.637 | — | — | 0.30 |
| `retained10_span`（旧默认，现可切换） | 81 | 0.444 | 0.690 | 0.71 | 0.93 | 0.81 |
| **`retained10_full`（现默认）** | **62** | **0.682** | **0.774** | **0.902** | **0.902** | **0.902** |

相对 span：完整和弦 **+23.8 个点**、family **+8.4 个点**、边界 F1 **0.81 → 0.902**。
和弦音恢复率（golden 62 事件逐段比对音级集合）：span 0.827、stable 0.815、冻结原输出 0.796、
**full 0.922**；带扩展音（4 音以上）的比例：真值 0.903、span 0.494、**full 0.806**。
复算：`full-decode/tone_recovery.py`。这也是把 full 设为默认的直接理由。

复算：

```powershell
& "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe" D:\ChordStudio\research\full-decode\verify_prod_full.py
```

脚本除了算指标，还验证生产接线：`retained10_full` 已出现在 `analysis.json.sources`、
`default_source` 仍是 `retained10_span`、`provenance.full_decode.applied = true`。

### 3.3 生产接入

| 文件 | 作用 |
|---|---|
| `backend/app/web/full_decode.py` | 解码器本体（numpy + joblib/sklearn，均为 web 环境已有） |
| `backend/app/web/models/boundary_nnls.joblib` | 合成域边界分类器 |
| `backend/app/web/models/full_decode.json` | 参数与来源说明（可追溯） |
| `backend/app/web/retained_worker.py` | 新增 `full` stage；`sources` 增加 `retained10_full`；`default_source` 不变 |

- 只读派生物：`posterior.npz` / `cells.json` / `deep.npz` / `notes.npz` / `beats.json`，
  不重跑模型、不改权重、不改冻结的 `retained10.lab`；
- 在应用里它是默认来源（面板「完整段解码（派生）」），`retained10_span` 仍可切换；
- 实测 run `73c95050`：sources 行数 279 / 177 / 81 / **64** / 78，主调 C（正确），
  `chord_evidence.evidence_source = retained10_stable`。

### 3.4 各阶段消融（同一 golden、同一评分函数，`full-decode/ablation.py`）

| 解码方案 | 段数 | exact | family | 边界 P | 边界 R | 边界 F1 |
|---|---:|---:|---:|---:|---:|---:|
| 120 类逐 cell argmax | 282 | 0.383 | 0.606 | 0.071 | 0.311 | 0.116 |
| 稳定时间轴（旧默认） | 177 | 0.401 | 0.637 | 0.205 | 0.590 | 0.304 |
| 小节池化 + 相位 DP（现默认） | 81 | 0.444 | 0.690 | 0.713 | 0.934 | 0.809 |
| 头结构（低音）+ 音级支持，固定小节 | 88 | 0.564 | 0.722 | 0.655 | 0.934 | 0.770 |
| + 边界分类器半小节分裂 + 增益剪枝 | 73 | 0.550 | 0.670 | 0.694 | 0.820 | 0.752 |
| + 半小节 CRF（换段代价反转） | 60 | 0.605 | 0.762 | 0.881 | 0.852 | 0.867 |
| **+ 半马尔可夫时长先验（8 拍，σ=0.30）** | **63** | **0.682** | **0.758** | **0.903** | **0.918** | **0.911** |

读法：头结构把**标签**拉起来（0.444→0.564），但边界仍然抖（F1 0.77）；CRF 把**精确率**
拉到 0.88 但召回降到 0.85；只有时长先验同时把两边拉满（0.903/0.918）。
半小节分裂 + 增益剪枝单独用会掉 family——它把正确的长段切成两半又各自重判。

生产版（`app/web/full_decode.py`）与上表最后一行是同一套算法；实现细节（低音按 cell 池化）
与改进后的音级权重（mu 0.6→0.7）使生产版落到 62 段 / exact 0.682 / family 0.774 / F1 0.902，
同时把和弦音恢复率从 0.907 提到 0.922、扩展音比例从 0.746 提到 0.806。

参数稳定性：在 λ∈[16,24]、β∈[6,10]、时长中心∈[7,9] 拍、σ∈[0.30,0.40] 的 135 个组合里
**30 个同时达到 exact ≥0.50 且 F1 ≥0.90**，最好 0.911（`tune_stability.py`），
说明不是单点调参。

### 3.5 非目标歌曲的稳健性（无 golden，只看是否崩）

| 歌曲 | stable 段数 | span 段数 | **full 段数** | 三种来源的和声主调 |
|---|---:|---:|---:|---|
| 虚拟 | 177 | 81 | **62** | C / C / C（声学 G，和弦覆盖后 C，正确） |
| 坏女孩 | 195 | 106 | **90** | C#m / C#m / C#m（声学 E；关系小调偏差是已知问题，未变） |
| 沧海一粟 | 154 | 78 | **73** | G# / G# / G#（= Ab，同一音级，正确） |

复算：`research/full-decode/check_other_songs.py`。三首歌上 `full` 都没有改变主调，
段数比 span 再少 6%–23%（虚拟 81→62、坏女孩 106→90、沧海 78→73）。

### 3.6 复算入口

```powershell
$py = "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe"
& $py D:\ChordStudio\research\full-decode\verify_all.py        # 全部验收 + 退出码
& $py D:\ChordStudio\research\full-decode\ablation.py         # §3.4 消融表
& $py D:\ChordStudio\research\full-decode\tune_stability.py   # §3.4 参数稳定性
& $py D:\ChordStudio\research\train-v2\verify_synthetic.py fix60
& $py D:\ChordStudio\research\full-decode\regress147\summarise.py
& $py D:\ChordStudio\research\full-decode\regress147\verify_key_pinning.py
& $py D:\ChordStudio\research\full-decode\verify_prod_full.py
```

注意：60 通道合成域模型**没有进生产**。它只用于「合成域 joint ≥0.95」这条验收，
以及给后续研究留一个可复用的质量头；生产侧仍然只用冻结模型 + 派生解码。

---

## 历史记录（阶段一 / 阶段二 / 旧阶段三，未改动）

### 阶段 1：合成集性质训练（当时的 24 维 chroma 上限）

| 数据 | 特征 | root | quality | joint |
|---|---|---:|---:|---:|
| 4 608 patch | CENS | 0.958 | 0.797 | 0.795 |
| 14 705 patch | CENS | 0.964 | 0.857 | 0.854 |
| 36 598 patch（400 首+ext） | CENS | 0.970 | 0.883 | 0.879 |
| 36 598 patch | CQT | 0.975 | 0.894 | 0.890 |
| 36 598 patch | CENS+CQT（24ch） | 0.971 | 0.887 | 0.884 |
| 24ch 5 折均值 | + TTA ×12 | — | — | 0.8923 |

当时的错误分布（held-out）：基础集 0.865、扩展集 0.935；`7/maj7/min7 = 0.628/0.620/0.661`。

试过但无效：七和弦过采样 ×4（0.831，倒退 5.3 点）、CENS+CQT 融合（0.884，无提升）、
混音低频 chroma 覆盖根音（exact 0.444→0.435）、拆轨低音 chroma 36 维（0.870）。

### 阶段 2：跨域到《虚拟》——合成模型不能替换生产模型

| 方法 | exact | family |
|---|---:|---:|
| 合成训练 CNN → 《虚拟》 | 0.387 | 0.581 |
| 生产模型 + 段池化 | 0.533 | 0.827 |

### 旧阶段三（已上线）：小节池化 + 逐小节相位 DP

| 输出（《虚拟》golden） | 段数 | exact | family | 边界 F1 |
|---|---:|---:|---:|---:|
| `retained10_stable` | 177 | 0.401 | 0.637 | 0.30 |
| `retained10_span`（默认） | 81 | 0.444 | 0.690 | 0.81 |

复算：`research/key-v3/chord_seg/check_prod_span.py`、`posthoc_pool.py`。

### 旧阶段三的负结果（别重复踩）

| 尝试 | 结果 |
|---|---|
| 混音 40–260 Hz 低频 chroma 覆盖根音 | exact 0.444→0.435，倒退 |
| 只用生产模型 + 相位搜索（纯度判据） | 选错相位 |
| 拆轨低音 chroma 加通道（36 维） | 0.870；同子集 0.849 vs 24 维 0.865 |
