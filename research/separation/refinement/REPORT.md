# 音轨细分第一轮交付与验证

日期：2026-11-01。检索和下一阶段路线见[RESEARCH.md](D:/ChordStudio/Dialogue/research/separation/refinement/RESEARCH.md)。本轮优先把“人声集合”推进到“主唱估计＋背景人声余量”，并对参考音频驱动的吉他细分进行本机实验。

## 已实现的人声细分

分轨页新增“细分主唱／和声”。在现有4轨、6轨或人声优先结果上，对父人声WAV做第二级推理；展开时子轨替换父轨，收起恢复父轨，避免重复播放人声。各子轨可独奏、静音及下载无损WAV。

专用模型：MelBand-RoFormer Karaoke，aufr33/viperx公开模型，已通过现有audio-separator注册表下载。输入为44.1kHz立体声，直接读取模型估计并输出浮点WAV，绕开逐轨峰值归一化。主唱是模型估计；第二轨是父人声减去主唱，命名“和声／人声余量”，包含背景和声，也可能包含混响、串音和误差。没有进一步把它虚标成每一条和声。

父人声、旧版本和分析结果均保留。子轨使用不可变生成目录，最后原子写入清单；失败不会覆盖旧的完整清单。复用前核验父轨哈希、推理实现及模型/配置指纹；父轨重算后变化会失效。所有试听和下载URL绑定所选主分离模型。

模型权重SHA256：`1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813`。推理环境audio-separator 0.48.0，torch 2.14.1+cu126，RTX4050 6GB。配置和运行哈希见每次inference.json。

## 《虚拟》实跑

来源为原六轨精细版的人声，未重跑或替换原六轨。

| 项目 | 结果 |
|---|---:|
| 音频样本数/采样率 | 10,501,209 / 44,100 Hz |
| 推理及加载 | 143.21秒 |
| 含转码、标注全流程 | 154.12秒 |
| 峰值已分配显存 | 1754.55 MiB |
| 主唱 RMS / 有声比例 | -21.8 dBFS / 66.2% |
| 和声余量 RMS / 有声比例 | -36.0 dBFS / 31.8% |
| 父轨与两子轨之和最大样本差 | 1.49×10⁻⁸ |

电平及活动比例沿用现有界面的单声道测量口径，不表示分离正确率。最后一项是代数守恒自检，不是质量成绩。本轮未进行人耳盲听评分。

完整清单：[xuni_vocal_result.json](D:/ChordStudio/Dialogue/research/separation/refinement/xuni_vocal_result.json)。

## 有真值的合成压力测试

从一个已提取主唱片段生成升4半音、较小音量的第二声部，保存两份精确波形后再相加；另测只有第一声部的负对照。这是受控重混，不能等同于自然录音双人合唱或同一歌手真实叠录。协议：[controls/protocol.json](D:/ChordStudio/Dialogue/research/separation/refinement/controls/protocol.json)。

| 已知声部 | 输入混音SI-SDR | 对应输出SI-SDR |
|---|---:|---:|
| 主唱 | 10.94 dB | 26.88 dB |
| 合成和声 | -10.85 dB | 15.81 dB |

两个声部在此构造条件下有明显分离，角色无需交换。但在只有主唱的对照里，余量轨仍含输入能量约10.3%（-9.87 dB）。因此此模型有错误分配单一人声的风险；余量不能自动标为真实背景歌手，更不能依据非零余量断言歌曲存在和声。

结果：[vocal_control_results.json](D:/ChordStudio/Dialogue/research/separation/refinement/vocal_control_results.json)。

## 工程检查与一并修复

- 修复鼓组子轨试听/下载缺少model参数，避免切换主模型后读到另一版缓存。
- 修复分轨任务完成后覆盖项目per-model分轨索引的问题；细分任务保持current_run不变。
- 展开/收起重新建播放器前暂停旧audio，清除旧索引；收起处于独奏的子轨时解除独奏，避免无声或后台残留播放。
- 新接口校验模型名和受限文件路径；子轨的缓存失效、资源下载与项目分析保护有测试。

定向测试：20项通过（5项人声/接口/任务测试，15项既有studio controls）。扩大到旧web workbench套件时有8项失败；使用尚未修改的安装版后端复测，同样8项失败，涉及旧音频身份夹具及当前测试解释器的Range响应。没有把它们写成“全部回归通过”。

真实浏览器验证：六行→七行、父轨替换、所选模型URL、WAV下载、独奏静音关系、收起重置独奏、旧播放器移除、无JavaScript异常。截图：[vocal_parts_ui.png](D:/ChordStudio/Dialogue/research/separation/refinement/vocal_parts_ui.png)，记录：[ui_results.json](D:/ChordStudio/Dialogue/research/separation/refinement/ui_results.json)。

## 下一阶段边界

本轮的人声功能是第二级估计分离。它还不是每条和声、每位歌手、每把吉他各自一轨，更不是原工程文件恢复。后续以真实多轨、同音色/同度负对照和跨片段身份一致性决定是否继续拆分，不以轨数增加作为成功标准。

## 吉他细分：已运行，但没有达到可接入产品的效果

取得并校验Banquet官方ev-pre-aug权重，使用官方结构严格加载全部模型参数，包括PaSST；没有随机初始化缺失参数。代码固定提交`79ed5bb75e5c3a40cd319d9d990cee913fc65c26`，权重SHA256为`657295888781e62ef50593002720d2edb3858b9e5bbfabf0c54f715a0da4b9e2`。单独建立运行环境，没有升级现有分离环境的依赖。

构造14秒同音区、同声像双吉他：GM25钢弦木吉他分解和弦与GM29过载电吉他旋律，同时活动。查询采用相同音色但不同音符，避免直接把目标真值波形作为查询。首先使用6秒块/3秒步长的浮点重叠相加实验接口：

| 实验 | 结果 | 判断 |
|---|---:|---|
| 木吉他查询→木吉他目标SI-SDR | -16.25 dB；混音基线+0.03 dB | 明显没有达到分离目标 |
| 电吉他查询，目标输出相对混音能量 | -91.49 dB | 近乎静音，漏分；此时不以SI-SDR数字掩盖失败 |
| 只有木吉他时查询电吉他 | -90.44 dB | 不能单独作为成功证据：有电吉他时也几乎没有输出 |
| 《虚拟》原吉他轨80–110秒，合成木吉他查询 | 输出能量相对父轨-21.54 dB | 无单轨真值，不证明提取到正确吉他 |
| 同一片段，改用该歌父吉他轨4–14秒作查询 | -20.91 dB | 未获得足以支持产品接入的证据，非“参考匹配就解决” |

为排除自写分块接口造成主要失败，另直接调用作者`EndToEndLightningSystem.chunked_inference`，采用官方6秒块/0.5秒步长，仅将batch降到1。作者反射填充要求足够长输入，因此短对照右侧补静音至40秒，评分只取原14秒。木吉他SI-SDR仍为-16.49 dB，目标能量相对混音-33.52 dB，峰值显存约2083 MiB。此对照没有改善主要失败。

以上结果不证明论文整体无效，也尚未定位到训练域、查询表示或依赖实现的具体原因；它证明本机这条候选路径还不能交付可靠的木/电吉他拆分。因此未增加一个会把近静音文件冒充电吉他的正式按钮。保留所有输出、查询、控制材料及推理原型，便于下一轮追查。

实现：[run_banquet.py](D:/ChordStudio/Dialogue/research/separation/refinement/run_banquet.py)，初次快步长脚本快照另存；固定角色评分：[control_scores.json](D:/ChordStudio/Dialogue/research/separation/refinement/control_scores.json)，能量及官方路径对照：[guitar_diagnostics.json](D:/ChordStudio/Dialogue/research/separation/refinement/guitar_diagnostics.json)。

## 如何使用与复现

新增人声功能已同步到安装目录，备份和文件哈希见[deployment.json](D:/ChordStudio/Dialogue/research/separation/refinement/deployment.json)。没有强制关闭现有桌面窗口；重新打开Chord Studio后，在《虚拟》的“六轨精细”页点击“展开主唱／和声”。其他已有主模型点击“细分主唱／和声”会针对那一版父人声独立生成。

本轮《虚拟》试听地址：[本机预览](http://127.0.0.1:61857/stems?project=aa5a41797bb343e590abcf4d8b27ae54&model=bs_roformer_sw)。预览不启动新的分离任务；新任务使用重新打开的桌面程序。

从`D:/ChordStudio/Dialogue`运行：

```powershell
python -m pytest tests/test_vocal_parts.py tests/test_studio_controls.py -q
python research/separation/refinement/score_controls.py
```

模型原型分别运行，输出目录须是新目录：

```powershell
& D:/ChordStudio/runtime/gpu/Scripts/python.exe app/web/vocal_parts_runner.py --input <已有立体声人声WAV> --outdir <新目录>
& D:/ChordStudio/runtime/banquet/Scripts/python.exe research/separation/refinement/run_banquet.py --input <混音WAV> --query <参考声音WAV> --output <新目录> --official
```

下一轮优先：真实多轨上的主唱/和声负对照与模型比较；Banquet真实样本/上游环境复现；再对照MuS3D的非生成式BS-Locoformer路线和音高提示，最后处理跨片段声部身份。不得直接递归当前和声余量生成任意数量的“歌手轨”。
