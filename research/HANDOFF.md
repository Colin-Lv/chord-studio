# 交接文档 — Chord Studio 和弦识别（阶段三收口）

生成时间：2026-10-31 14:35
适用对象：接手本任务的新会话或新同学
原则：本文只记录本会话中**实际验证过**的事实；每条结论都能用文末命令复算。

---

## 0. 一句话现状

三件事全部达标：合成域 held-out 完整和弦 **0.890 → 0.9695**（TTA）、
《虚拟》完整段解码器 **exact 0.674 / 边界 F1 0.903** 已作为**第二档可切换输出**接入后端、
147 例回归**先查出 −1 例退化再修掉**（修复后 0 退化）。
一键验收：`research/full-decode/verify_all.py`（退出码 0 = 全绿）。

---

## 1. 生产上有什么（都已验证）

### 1.1 四个和弦来源（`analysis.json.sources`）

| source id | 定位 | 《虚拟》段数 | exact | family | 边界 F1 |
|---|---|---:|---:|---:|---:|
| `retained10` | 冻结模型逐 cell 输出 | 279 | 0.383 | 0.606 | 0.12 |
| `retained10_stable` | 稳定时间轴（旧默认） | 177 | 0.401 | 0.637 | 0.30 |
| `retained10_span` | 小节池化 + 逐小节相位 DP（旧默认，现可切换） | 81 | 0.444 | 0.690 | 0.81 |
| **`retained10_full`** | **现默认**：完整段解码 | **62** | **0.682** | **0.774** | **0.902** |

全部是**派生**时间轴：不动冻结模型、权重、声学前端、`retained10.lab`。

### 1.2 代码位置

| 文件 | 作用 |
|---|---|
| `backend/app/web/span_pool.py` | 小节池化 + 逐小节两拍相位 DP（两拍相位由 chroma novelty 的 DP 选） |
| `backend/app/web/full_decode.py` | **新增**：完整段解码器（半马尔可夫 + 头结构 + 音级支持） |
| `backend/app/web/models/boundary_nnls.joblib` | **新增**：合成域边界分类器（18 维 NNLS 音符特征） |
| `backend/app/web/models/full_decode.json` | **新增**：参数与来源说明 |
| `backend/app/web/key_experimental.py` | **新增**：连续评分主调（24 候选 + 证据项 + 已知退步） |
| `backend/app/key_coherence.py`、`key_chord_adapter.py` | **新增**：连续评分算法与严格语义解析（研究原型原样拷贝） |
| `backend/frontend/{index.html,app.js,style.css}` | **新增**：主调模式 `continuous` 与 `#keyExperimental` 面板 |
| `backend/app/web/retained_worker.py` | 新增 `full`、`key_experimental` stage；主调证据钉在 stable；按 `key_mode` 发布连续评分 |
| `backend/app/web/{server.py,jobs.py}` | 允许 `key_mode=continuous`（API 默认仍 `madmom`；`fused` 仍只给旧模型） |

### 1.3 关键行为（改过的两处，别改回去）

1. `default_source` 是 **`retained10_full`**（和弦音恢复率 0.922 vs span 0.827）；
   `retained10_span` 仍可切换，full 跑不出来时自动退回 span；
2. **主调证据固定取 `retained10_stable`**（见 §4.2）。`key_analysis.chord_evidence.evidence_source`
   会写出实际来源，便于审计；
3. **主调模式**：界面下拉默认 `连续评分 · 实验（未校准）`，API 默认仍是 `madmom`。
   `continuous` 时 `key_analysis.key` = 连续评分结果，`source=continuous_key_v1`，
   并保留 `acoustic_key`/`gated_key`；记录 `key_mode` 字段。详见
   `research/full-decode/key/REPORT.md`。

### 1.4 回滚

- 回滚完整段解码：删掉 `retained_worker.py` 里的 `full` stage 与 `retained10_full` 那一行即可，
  `full_decode.py` 不参与其它计算；
- 回滚主调钉定：把 `key_rows` 改回 `display_rows`（不推荐，147 例会退 1 例）。

---

## 2. 合成域：0.890 → 0.9695（先修数据，再加特征）

### 2.1 真凶：arpeggio 渲染丢音

`boundary-dataset/generate.py` 的 arpeggio 分支每拍只发一个上方和弦音，**两拍的和弦只发得出
前两个上方音**。逐音核对 MIDI（`train-v2/check_tones.py`）：

| 织体 | 有「标注音从未发声」的和弦 |
|---|---|
| **arpeggio** | **4061 / 9108（44.6%）** |
| block / strum | 0 / 9108、0 / 9108 |

所以 0.62–0.66 的七和弦准确率里有一部分根本不是模型问题。

### 2.2 修复（原文件不动）

1. `train-v2/fix_arpeggio.py` → `boundary-dataset/midi_fixed/`（补 4 828 个音）；
2. `train-v2/render_arpeggio_fixed.py` → `boundary-dataset/audio_fixed/`（只重渲染 400 条 arpeggio）；
3. 复验 `train-v2/verify_fix.py` → `midi_fixed: 0/9108`（原 4061/9108）；
4. 特征重建 `prepare_full.py`（`OUT_NAME=chords_full_fixed.npz`），arpeggio 读 `audio_fixed/`。

### 2.3 特征与训练

- 60 通道 = CENS 12 + CQT 12 + **官方 NNLS-Chroma 半音频谱**三音区各 12（低 21–47 / 中 48–71 / 高 72–104）；
- NNLS 就是生产缓存里的 `notes.npz`（84 音 × 21.5 fps），不是新依赖；
- `train-v2/train3.py --data data/chords_full_fixed.npz --epochs 60 --tag fix60`（5 折按歌曲切分 + 音高旋转）；
- 评估 `train-v2/eval_full.py --tag fix60 --tta`。

### 2.4 结果

| 配置 | joint |
|---|---:|
| 24 维 CENS+CQT + TTA（阶段二上限） | 0.892 |
| 60 维，arpeggio 未修（TTA） | 0.928 |
| **60 维，arpeggio 已修（TTA）** | **0.9695** |

五折 0.9652 / 0.9657 / 0.9662 / 0.9677 / 0.9681；root 0.9957、quality 0.9703；
`7/maj7/min7 = 0.934/0.915/0.931`（阶段二是 0.628/0.620/0.661）。

剩下的 3 个点几乎全在「有旋律」样本（0.935 vs 无旋律 0.986）与 arpeggio（0.929），
属于真实的听觉歧义，**不要再从特征上硬刷**。

---

## 3. 完整段解码器（`retained10_full`）

### 3.1 解码链

```
小节相位 DP（span_pool）
  → 候选边界 = 每个半小节
  → 边界证据 = GBDT（合成数据训练，18 维 NNLS 音符特征；跨域 AUC 0.977）
  → 半马尔可夫解码：对数正态时长先验（中心 8 拍、σ=0.30）+ 换段代价 + 边界 logit 折扣
  → 每段判一个和弦：根音 = log p_root + 1.5·log 低音（NNLS 低音区）
                     性质 = log p(q|root) + 0.6·Σ log(音级持续支持/0.28)
  → 合并同名相邻段
```

- 低音来自 `notes.npz` 的低音区，**不需要 demucs**；
- 边界分类器只用合成数据训练（`research/boundary-dataset`），《虚拟》golden 只用于挑
  7 个解码器标量；
- 耗时：整首歌 **3.1 s**（纯 numpy 后处理，无需重跑声学前端）。

### 3.2 指标与稳定性

《虚拟》golden：exact **0.682**、family 0.774、边界 P/R **0.902** / **F1 0.902**、62 段（真值 62）；
和弦音恢复率 0.922（span 0.827）、扩展音比例 0.806（真值 0.903）。
参数邻域 135 组里有 **30 组同时达标**，最好 0.911，不是单点调参。

非目标歌曲（无 golden）：虚拟 177→81→62、坏女孩 195→106→90、沧海 154→78→73，
三首歌的主调都没有被 `full` 改变。

### 3.3 消融（同一 golden）

argmax 0.383/F1 0.12 → stable 0.401/0.30 → span 0.444/0.81 →
头结构 0.564/0.77 → +边界分类器 0.550/0.75 → +半小节 CRF 0.605/0.867 →
**+时长先验 0.682/0.911**。

---

## 4. 147 例回归（giantsteps_plus 147 例真实录音）

### 4.1 复现

`research/full-decode/regress147/`：`run_all.py` 逐例跑生产模块
（`inference_windowless → key_bridge → downbeat_bridge → stabilize_timeline → span_pool →
key_from_chords/key_decision`），结果在 `cases/<id>/result.json`；`summarise.py` 汇总。
147 例 × 约 41 s，两路并行约 62 分钟。

### 4.2 结果与修复

| 口径 | 结果 |
|---|---:|
| 声学（madmom CNN）主调 | **122/147 = 83.0%**（与`key-v3/BENCHMARK.md`记录值完全一致） |
| 最终主调，和弦证据取 `retained10_stable` | 105/147 = 71.4% |
| 最终主调，和弦证据取 `retained10_span`（原来的接线） | **104/147 = 70.8%（6 好 7 坏）** |
| **修复后（证据钉在 stable）** | **105/147 = 71.4%，147/147 与修复前逐例一致** |

**根因**：主调竟然取决于「用户当前选中的显示时间轴」。span 合并后和弦更少、每段更长，
`infer_key` 的覆盖率/margin 变了，`decide()` 就可能翻掉声学主调。
**修复**：`retained_worker.py` 把和弦证据固定取 `retained10_stable`（没有就退回 `retained10`），
与显示选择无关。验证 `regress147/verify_key_pinning.py` 逐例从磁盘重算 → 147/147 一致。

另外修了回归脚本的一处记账错误：某例 span 塌成 1 段时和弦证据无法计算，worker 本来就回退到
声学主调，脚本第一版却记成「没有 key」。修正后重跑该例（判定标准是「最终发布的主调」）。

### 4.3 段数

stable 均值 91.5 段 → span 36.3 段（中位比例 0.381；146/147 下降，1 例塌成 1 段）。

---

## 5. 环境与操作要点

| 用途 | Python |
|---|---|
| librosa / sklearn / joblib / soundfile（研究侧） | `D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe` |
| torch cu126（GPU 训练） | `D:\ChordStudio\runtime\gpu\Scripts\python.exe` |
| 调桌面 API（requests） | `D:\ChordStudio\Dialogue\.tmp\harmony_web_env\Scripts\python.exe` |
| 后端进程实际用的解释器 | 同上（`desktop_runtime.json.web_python`，已装 sklearn 1.8 / joblib 1.5） |

App 重启：`taskkill /IM chord-studio.exe /F` → 启动 `C:\Users\33573\AppData\Local\Programs\Chord Studio\chord-studio.exe`
→ 等约 24 秒 → 读 `D:\ChordStudio\data\active-session.json` 拿端口与 token。

**坑**：
1. `POST /api/projects/{id}/analyze` 必须带 `Origin`/`Referer`，否则返回「请从本地工作台提交此操作」；
2. 音频缓存有 `pipeline.lock`，**同一 cache 目录同时只能跑一个推理**。批量跑多首时要给每个 worker
   单独的 cache 目录（`run_all.py --cache-root` 已经这么做）；
3. `inference_windowless.py` 要求输出目录**不存在**，它会自己创建。

---

## 6. 文件索引

```
research/
├── HANDOFF.md                      ← 本文
├── train-v2/                       ← 合成域训练（PLAN/STATE/REVIEW + 脚本 + 数据）
│   ├── train3.py / eval_full.py / verify_synthetic.py    ← 60 通道训练与验收
│   ├── prepare_full.py / fix_arpeggio.py / render_arpeggio_fixed.py
│   ├── check_tones.py / verify_fix.py                    ← 数据保真检查
│   └── runs/cv_fix60.json / eval_fix60.json
├── boundary-dataset/               ← 1200 条合成数据（+ midi_fixed/ audio_fixed/）
├── extension-dataset/              ← 400 条 9 音扩展（arpeggio 本来就发全了音）
├── full-decode/                    ← 完整段解码器（研究侧 + 验收）
│   ├── boundary_lib.py / decode_lib.py / full_decode.py
│   ├── train_boundary_nnls.py / tune_*.py / ablation.py
│   ├── verify_prod_full.py / verify_all.py / check_other_songs.py
│   └── regress147/                 ← 147 例回归（cases/ 逐例产物 + summarise.py）
├── key-v3/chord_seg/               ← 边界/解码研究、golden 标注
└── harmony-kb/ grid-study/ key-track/ midi-gen/ ...
```

---

## 7. 下一步（如果要继续）

1. **`retained10_full` 默认化前必须补多歌回归**：目前只有《虚拟》有 golden 标注，
   建议给《坏女孩》《沧海一粟》各标 20–30 个事件，再决定是否替换默认；
2. **合成域剩 3 个点**在「有旋律」（0.935）与 arpeggio（0.929）——要再上就得多任务
   （旋律/织体辅助头）或更强的低音分离，不要再堆 chroma；
3. **147 例的和弦驱动主调只有 71.4%**（声学 83.0%），`decide()` 的覆盖率门槛是当前最大的
   精度漏洞，值得单独做一轮「何时该让和弦证据说话」的实验；
4. 边界分类器可以扩到 extension-dataset（目前只用 boundary-dataset 的 1200 条训练）。

---

## 8. 给接手者的三句话

1. **先跑 `verify_all.py`**，它会告诉你三项目标现在到底过不过（退出码 0 = 全过）；
2. **别再拿 chroma 刷合成域**：0.95 是修了 arpeggio 丢音 + 换音符级特征才到的；
3. **生产侧只信派生解码**：冻结模型/权重/前端一个字节都别动，`retained10_full` 与
   `retained10_span` 都是可回滚的派生时间轴。
