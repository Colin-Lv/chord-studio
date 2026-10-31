# 连续评分主调 · 前端接入记录

日期：2026-11-01。对象：安装版 `Chord Studio`（`C:\Users\33573\AppData\Local\Programs\Chord Studio`）。
上游：`D:/ChordStudio/Dialogue/docs/dsh_key_handoff_20261101.md`、`key_audit_errors_20261101.md`、
`key_stage3_report_20261101.md`。

## 0. 为什么之前"主调判断还是默认 CNN"

上一轮接入只加了**只读面板**（`key_analysis.experimental`），但主调链路本身没变：

| 环节 | 之前 |
|---|---|
| 前端下拉 | 只有 `madmom`（CNN · 当前默认）与 `fused`（仅旧模型） |
| 后端校验 | `jobs.submit`：`retained10` 只允许 `key_mode="madmom"`，其余直接报错 |
| worker | `retained_worker` 完全不读 `key_mode`，永远走「CNN + 和弦覆盖」 |
| 结果字段 | `analysis.json` 不记录主调用了哪种方式 |

所以界面上写的是"CNN · 当前默认"，连续评分只是面板里的一段旁证。**这一轮把它变成一条真正的主调模式**。

## 1. 改了什么

| 文件 | 改动 |
|---|---|
| `backend/app/web/server.py` | `key_mode` 允许 `continuous`（API 默认仍是 `madmom`，不影响脚本/旧客户端） |
| `backend/app/web/jobs.py` | `retained10` 允许 `madmom`/`continuous`，仍拒绝 `fused`；非 `retained10` 拒绝 `continuous` |
| `backend/app/web/retained_worker.py` | 读 `job.key_mode`；`continuous` 时把连续评分结果作为发布主调，保留 `cnn_key`/`gated_key`，追加警告；结果写 `key_mode` |
| `backend/app/key_coherence.py` | 研究原型的连续评分（原样拷贝，未改算法） |
| `backend/app/key_chord_adapter.py` | 严格语义解析适配层（`chord_semantics_v3`，原样拷贝） |
| `backend/app/web/key_experimental.py` | 新增：24 候选排序 + 证据项 + 解析统计 + 已知退步 + 警告 |
| `backend/frontend/index.html` | 主调下拉新增 `continuous`（排在第一位并默认选中）；新增 `#keyExperimental` 面板 |
| `backend/frontend/app.js` | `renderKeyExperimental()`；读取 `result.key_mode`；主调来源显示「连续评分 · 实验（未校准）」；等音（G#/Ab）按音级判定 |
| `backend/frontend/style.css` | 面板与候选表样式 |

发布主调由 `key_analysis.key` 决定（前端本来就取这个字段），所以选择 `continuous` 后
**主调、级数记号、候选列表全部按连续评分走**，不是旁注。

## 2. 实测（三首歌，`key_mode=continuous`）

| 歌曲 | 声学 CNN | 覆盖裁决（旧） | **连续评分（现发布）** | 分差 | 解析和弦 |
|---|---|---|---|---:|---:|
| 虚拟 | G | C | **C** | +0.233 | 175 / 177 |
| 坏女孩 | E | E | **E** | +0.670 | 188 / 195 |
| 沧海一粟 | Ab | Ab | **G#**（同音级） | +3.217 | 151 / 154 |

《虚拟》的 C 与 stage-3 报告的分值完全一致：C −1.1690、G −1.4019、Am −4.6025，差 0.232886。
`madmom` 模式仍然可复现旧行为（虚拟 C 由覆盖裁决给出、坏女孩 E、沧海 Ab），回归没有被破坏。

## 3. 界面上能看到的

`#keyExperimental` 面板（在有主调结果时出现）：

- 标题：`连续评分 · 实验（当前主调来源）`，或未采用时 `连续评分 · 实验（第二档）`；
- 一行摘要：实验主调、是否与当前一致、与第二名的分差、CNN 原判、覆盖裁决值；
- 24 候选中的前 6：总分 / 声学项 / 和声项 / 调内占比 / 主和弦占比 / 正格到达 / 属回次数；
- 「采用实验主调」按钮：与手动改调走同一条 `/key-review` 保存路径，当前已一致时自动隐藏；
- 解析统计（标签数、可解析数、N/X/缺失）与**两条已知退步**（191347、853834）；
- 四条警告：分数非概率、147 例为已暴露回归集（123/147，改对 3 改错 2）、权重在看过评测后设定、不改变默认。

## 4. 与审计约束的关系（必须一起读）

- **没有**把实验模块说成已验证：结果里 `status=experimental_not_default`，UI 标题写「实验」，
  警告逐条展示，已知退步写在面板里；
- **API 默认仍是 `madmom`**，只有桌面界面的下拉默认选到 `continuous`；旧脚本、旧链接行为不变；
- 主调证据仍钉在 `retained10_stable`（上一轮的修复没动），连续评分与覆盖裁决共用同一条时间轴；
- 严格解析（`key_chord_adapter`）已接进链路，POP909 的三例退步（024/505/513）不因此消失，
  也没有删样本；
- 没有为保住《虚拟》的 C 调新增门控或权重搜索：分差是原公式算出来的。

## 5. 复算

```powershell
$py = "D:\ChordStudio\Dialogue\.tmp\harmony_web_env\Scripts\python.exe"
& $py D:\ChordStudio\research\full-decode\key\verify_integration.py   # 后端+结果+资产+渲染 全部断言
node D:\ChordStudio\research\full-decode\key\renderer_harness.js <run>\analysis.json
& $py D:\ChordStudio\research\full-decode\key\check_experimental_block.py
```

`verify_integration.py` 覆盖：`key_mode` 生效、发布主调 = 连续评分、`source=continuous_key_v1`、
声学/覆盖值保留、24 候选、警告存在、三条 payload 的 DOM 渲染、静态资产已更新。

## 6. 已知限制（不因为接入而消失）

1. 191347（D#m→A#m）：反复回归奖励饱和把 A#m 抬上来；
2. 853834（Em→E）：发射势阶段已偏向 E 大，时间模型只是放大；
3. 分数未校准、147 例已反复查看、权重在看过评测后设定 —— 需要按作品隔离的新集才能宣称泛化；
4. `key_mode=continuous` 只在 `retained10` 档位开放；`v1/v2` 仍走原分歧裁决。

下一步仍是交接文档 §6 的顺序：先做结构证据（乐句归一化/去重）与真实声部证据，再谈这两例。
