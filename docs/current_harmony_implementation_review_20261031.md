# Chord Studio 当前和声识别实现检查

检查日期：2026-10-31（Asia/Shanghai）。本次只检查实现、校验文件并复算缓存；未修改部署代码、权重、已有分析或旧评测输出。

## 1. 当前真正部署的版本

- 当前工作区 `C:\Users\33573\Desktop\对话` 是 `D:\ChordStudio\Dialogue` 的 junction；其中大量实现属于早期实验，不能据此判断桌面版当前模型。
- 桌面后端：`C:\Users\33573\AppData\Local\Programs\Chord Studio\backend`。
- 当前模型：**chord_retained11_v90**；桌面配置标记音频入口 **v88**，底层音频契约名称仍沿用 v82。
- 权重：`backend\inference\model\retained10.npz`，文件名沿用旧版，大小 **230,907 字节**。
- 实测 SHA-256：`cf44143ee0b5747ba2962b8aac7349bd9ee4fd2489cc63872dc45f8273fd31bd`。
- 权重与 `model\manifest.json`、`desktop_runtime.json` 一致。相对 v80，只有 32 个数组变化：4 个 quality 头各 6 个数组，以及 4 个 router 各 2 个数组。source 和 N 头未变。
- 四个 quality 头均逐数组匹配 `D:\ChordStudio\research\training\v90\quality_<seed>.npz`。

## 2. 实际调用链与结构

桌面 `app\web\retained_worker.py` 调用 `inference\audio_predict.py`，完成音频解码、音频缓存校验、madmom CNN/CRF 时间锚点、DeepChroma 与 NNLS 特征提取，再调用 `inference\model\predict.py::Retained10`。

模型输入为每个 cell 的 149 维特征、10 Hz 的 12 维 DeepChroma，以及每 cell 4 维能量。source 对 12 个候选根音分别旋转输入，运行共享的 **149→96→48→10** 网络，拼成 120 类；并非单个 48→120 输出层。

quality 残差头以 hidden 48 维与旋转 DeepChroma 12 维拼成 60 维输入，运行 **60→32→10**，修正各根音下的质量 logits。四种子集成，N 头另作无和弦判断。词表为 12 根音 × maj/min/7/maj7/min7/9/maj9/min9/sus2/sus4，另加 N；不具备完整未知和弦检测。

v90 将每个 router 的末层权重清零、bias 设为 3，实测专家混合权重均为 **0.9525741339**。当前是固定比例混合，已不随片段自适应。

逐 cell 输出保存为 `retained10.lab`；随后生成独立的 `retained10_stable.lab`，处理同根音类型抖动、短段合并和小节线吸附。后者是界面默认来源。madmom 主调、拍点及结构分析是另行运行的模块，不属于 v90 和弦头。

当前主调还接入 `app\key_from_chords.py` 和 `app\key_decision.py`：用和弦证据、4-5-6 规则及音级覆盖率决定是否改写 CNN 主调。`D:\ChordStudio\research\key-v3` 是更新的独立实验；未发现桌面 worker 调用该 v3 分支。

## 3. 本次实际验证

通过部署代码验证模型清单 12 个文件、音频入口清单 7 个文件，以及运行工具和 madmom 模型身份。示例推理成功，后验形状为 54×120，输出 15 段。

随后直接使用已安装模型，在内存中重跑已有音频特征；按现有 `evaluate.py::frame_scores` 的 0.1 秒网格、逐片段等权口径复算：

| 数据 | v80 exact | v80 仅打开 router | v90 exact | v90 root |
|---|---:|---:|---:|---:|
| 25 个 chordfix 合成片段 | 89.1575% | 94.1673% | 94.2550% | 95.6613% |
| 24 个 POP909 留出渲染片段 | 60.2593% | 65.2315% | 67.7963% | 80.7870% |

第一组 19 个提升、5 个下降、1 个不变；第二组 24/24 相对 v80 提升。第一组新残差相对“v80 仅开 router”的净提升约 **0.088 个百分点**；第二组约 **2.565 个百分点**。两组收益大部分来自打开 router，但新残差在留出渲染上也有额外改善。

真实项目最新运行的 v90 缓存回放与原 prediction.json 的 segments 完全相等：

- 《坏女孩》：run `5bef14fc2b9745f98c860d0fe94095d4`，319 段，当前主调 E。
- 《虚拟》：run `bc928edd2c5c460784154d969e07d499`，279 段，当前主调 C。

这证明当前模型与已有结果一致，不证明真实录音标签正确。本次未重新解码整曲、未重训、未运行与桌面实现不同的旧仓库全量测试。

## 4. 检查发现

### A. 优先补真实录音评估：固定 router 的泛化效果尚无充分证据

`research\training\V90_REPORT.md:118–125` 已明确承认真实 OOF 未复测，并记载旧残差在真实 OOF 上曾有 root 下降。当前合成/留出渲染结果能复现，但两首真实歌曲只有段数与抖动代理指标，无法确认固定 95.26% 专家权重对真实根音、质量和 N 的净效果。应在独立真实录音及明确作品隔离的数据上比较 v80、v80-on、v90，分别报告 root、质量、N、整标签及逐曲退步。

### B. 版本说明和组件溯源未同步

`D:\ChordStudio\README.md:14` 与仓库 `README_chord_recognizer.md:4` 仍称 v80 为当前模型。部署 `audio_manifest.json:4–5` 仍记录旧模型 manifest/权重哈希；模型 manifest 的 `components` 仍将已替换的 quality/router 标成旧组件路径与哈希，并保留“没有改变产品默认”的旧限制文字。

整包权重哈希和桌面配置是正确的，当前校验因此可以通过；问题是组件来源记录不能准确描述 v90，旧音频清单的模型身份字段也不是当前运行时强校验项。应把父版本来源与当前组件身份分开，更新部署说明和 manifest。

### C. 打包脚本无法可靠保证四种子完整与固定基线

`D:\ChordStudio\research\training\make_bundle.py:19` 将当前安装目录当作 SRC；安装升级后再次执行同一命令，基线可能已经改变。第 49–51 行在指定 quality 文件缺失时直接 continue，仍能生成标为 v90 的包，并且只更新整个 NPZ 哈希，未更新组件哈希。

应显式指定并核验不可变的 source bundle，要求四个种子及全部数组齐全后再写包；同时记录各新组件身份。此问题不等于本次部署缺权重：本次已证明四个种子都正确装入。

### D. 稳定时间轴的 slash 低音保真存在缺口

部署 `app\web\stabilize_timeline.py:283–284` 读入原 LAB 后丢弃标签，只把 cell/后验送入 stabilize；第 244 行未传 slash 时全部置空。最小复现：相同 cell `C:maj/3` 与确定的 C:maj 后验，原 decode 输出 `C:maj/E`，稳定化输出 `C:maj`，即使只有一个长段、无需合并也会丢低音。

建议从 cell 的显式低音解析并传入 slash，补保真检查。当前标准 madmom 锚点通常不携带 slash，因此本次未证明两首真实歌曲因此受损；这是已复现的接口保真缺陷。

### E. 训练说明与实际代码有差异

- `train_v90.py:183` 实际 batch=64，报告第 72 行写 4000。
- 模块说明宣称含反事实一致性损失，但 train_seed 实际只有 chord CE 与 root CE；去旋律样本只参与普通数据增强。
- 新数据总计 118 段，其中实际训练 **94 段 / 17,695 cells**，留出 **24 段 / 4,529 cells**；并非 118 段全部训练。
- `train_v90.py:156` 算出 labeled 掩码但未用于采样；训练池有 1,421 个全零 chord 目标 cell，仍参与归一化与采样，对两项监督损失不贡献标签梯度。
- metrics.json 的最大 torch/numpy parity 误差为 **3.8147e-6**，报告写 1.9e-6。仍属很小数值差异，但记录应准确。
- 四份旧 fold 元数据合计 2,136 段，只有 552 个唯一 id；142,386 cells 不是互不重复的新样本量。重复本身不能据此判定留出泄漏，但后续复验需保留作品/录音身份并重新建立隔离。

### F. 主调 4-5-6 规则的最终生效范围有限

`app\key_decision.py:117` 要求和弦候选比 CNN 的音级覆盖率高至少 3 个百分点。关系大小调使用相同音级集，因此此条件无法区分它们。复现 A:maj–B:maj–C#:min 循环：guard_456 把 C#m 候选改成 E，但当 CNN=C#m 时，两个覆盖率都为 100%，最终 decide 仍保留 C#m。

这是当前保守覆盖率裁决的能力边界；不能把“guard 已判大调”理解成“最终一定能修正关系调”。本次《坏女孩》的 CNN 本身为 E，所以最终 E 不能单独证明规则完成了关系调纠错。

## 5. 可继续工作的入口

- 已部署推理：`C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\inference\model\predict.py`
- 音频前端：同 backend 下 `inference\audio_predict.py`
- 桌面编排与派生时间轴：同 backend 下 `app\web\retained_worker.py`、`stabilize_timeline.py`
- 训练/评估：`D:\ChordStudio\research\training\train_v90.py`、`make_bundle.py`、`evaluate.py`
- 原升级记录：`D:\ChordStudio\research\training\V90_REPORT.md`
- v80 回滚原件：`D:\ChordStudio\backup\model-v80-20261030`

建议先统一版本与溯源、修正打包完整性检查，再补真实录音评估。是否调整固定 router 比例，应由该评估决定，而不是根据段数下降决定。
