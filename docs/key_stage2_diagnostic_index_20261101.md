# 句级与录音诊断材料

2026-11-01。这里是研究材料，不是新的歌曲标准答案。人工结构标注来自固定版本的[研究项目](https://github.com/Dsqvival/hierarchical-structure-analysis/tree/3a8d0d096e8dd2f62e38e48500bf0b230eeca585)。两种标注均保留；每张图按同一规则展示标注1中最先两个有旋律音的乐句结尾窗口，未按支持何种调性挑选。

横轴是上游处理后的四分音符位置，纵轴为MIDI音高，红线为结构边界。和弦文字来自上游分析，不能当作独立人工和弦真值。连续音跨边界不表示完成终止。

| 编号 | 用途 | 原MIDI时间映射 | 图 |
|---|---|---|---|
| 001 | 对照 | 通过 | [001](D:/ChordStudio/research/key-stage2-20261101/structure-packet/001.png) |
| 002 | 对照 | 通过 | [002](D:/ChordStudio/research/key-stage2-20261101/structure-packet/002.png) |
| 005 | 对照 | 未通过：内部时移 | [005，仅处理后坐标](D:/ChordStudio/research/key-stage2-20261101/structure-packet/005.png) |
| 006 | 对照 | 通过 | [006](D:/ChordStudio/research/key-stage2-20261101/structure-packet/006.png) |
| 064 | 旧失败；分句方式使末音偏好反转 | 通过 | [064](D:/ChordStudio/research/key-stage2-20261101/structure-packet/064.png) |
| 067 | 旧失败；末音有部分参考支持 | 通过 | [067](D:/ChordStudio/research/key-stage2-20261101/structure-packet/067.png) |
| 070 | 旧失败；不能以低主和弦占比否认旋律证据 | 通过 | [070](D:/ChordStudio/research/key-stage2-20261101/structure-packet/070.png) |
| 143 | 旧失败；末音仍偏向关系小调 | 通过 | [143](D:/ChordStudio/research/key-stage2-20261101/structure-packet/143.png) |

## 两个真实录音退步

- [191347片段](D:/ChordStudio/research/key-stage2-20261101/recording-packet/191347.wav)：原音频0–32秒，包含循环回归。参考D♯m、原型A♯m。需区分循环内部相邻连接与结构性的落点。
- [853834片段](D:/ChordStudio/research/key-stage2-20261101/recording-packet/853834.wav)：原音频47.1–65.1秒，位于模型持续标E大三和弦的区间。参考Em、原型E。需核对三音及其声部来源。

源音频哈希与原和弦缓存一致。本轮没有完成独立听审，亦未生成可靠的旋律/低音转录；没有把这些片段的模型和弦写成标准答案。

## 完整记录

- [结构与音符上下文JSON](D:/ChordStudio/research/key-stage2-20261101/structure-packet/packet.json)：两种标注、末音、前音、休止距离、和弦、钢琴最低音代理及对齐残差。
- [录音JSON](D:/ChordStudio/research/key-stage2-20261101/recording-packet/packet.json)：来源哈希、片段时间、原始标签、CNN候选和评分分解。
- [阶段报告](D:/ChordStudio/Dialogue/docs/key_stage2_report_20261101.md)：结果、限制、后续验收要求。
