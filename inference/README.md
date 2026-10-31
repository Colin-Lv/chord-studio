# Chord Studio：原始音频研究入口

Windows本地音频→冻结retained10和弦JSON/LAB。model目录是已交付v80原样副本，13组件权重不变；这不是新训练或默认模型升级。音频只留在本机。

## 在此电脑使用

把整个目录解压到本地，在PowerShell执行（替换三个用户路径；输出目录必须尚不存在）：

```powershell
& 'C:\Users\33573\miniconda3\python.exe' -I -B '.\audio_predict.py' --audio 'C:\音乐\歌曲.wav' --output-dir 'C:\音乐\结果_001' --cache-dir 'C:\音乐\研究缓存'
```

默认最多300秒、源文件512MiB，支持解码器接受的本地音频；超长拒绝而不会静默截断。可显式设置`--max-seconds`，硬上限600秒，但长曲与所有编码格式没有全面验证。输出目录包含prediction.json、prediction.lab、posterior.npz和执行日志；仅run_status.json中的successful=true确认完整运行。后验未校准。不要将临时缓存或失败日志当完成输出。

同一音频和前端版本再次运行会复用声学缓存，仍须另选新的输出目录。缓存文件损坏或版本身份不符即拒绝，不静默沿用。一个缓存目录同一时间只处理一首；512MiB源文件、256MiB原生PCM临时文件、20分钟执行、内存预算及系统余量限制均生效。默认子进程树1500MiB预算，启动需另留1056MiB，运行时768MiB系统余量为底线。仅结束本命令自己启动的子进程。

## 输入与运行环境

本目录不依赖原仓库或D盘研究模型。runtime.local.json显式绑定此电脑已有FFmpeg、Sonic Annotator、NNLS Chroma插件和madmom0.16.1隔离解释器及三个上游权重。移动此目录可运行，但迁到另一台机器须提供相同版本依赖并重建路径/摘要配置；工具不会自动安装、联网或修改全局设置。系统Python需requirements.txt中的库；model中原样附带decorator许可源文件。

44.1kHz单声道PCM用于CNN、NNLS；DeepChroma沿用原始音频加载路径，三者样本时钟须一致。能量在原始采样率/声道上计算，与既有v51输入一致。NNLS块16384、hop2048、原右侧补齐与149维短窗/音区特征保持；前端纯函数由原实现逐个AST对照。没有参考和弦、MIDI、歌名、拍号或谱面输入。

输出支持maj/min/7/maj7/min7/9/maj9/min9/sus2/sus4和N；不增加5/13/dim7能力，不输出调性/完整调式。没有原和弦时间轴时，显式斜杠低音通常不可获得。N误报仍有明显局限。验证仅说明既定模型在少量旧开发音频的可复现性，不是新准确率证明，旧封账集不重测。

开源前端说明：[MADMOM CNN+CRF](https://madmom.readthedocs.io/en/v0.16.1/modules/features/chords.html)、[DeepChroma](https://madmom.readthedocs.io/en/v0.16.1/modules/audio/chroma.html)、[NNLS Chroma](https://isophonics.net/nnls-chroma)。这些组件用于复现已冻结前端；未改它们的权重或引入新平滑。上游模型和第三方二进制不随此包分发，保留本机已有安装。
