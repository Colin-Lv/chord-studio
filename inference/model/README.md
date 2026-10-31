# Chord Studio retained10：可加载研究模型

此目录可整体复制到其他本地位置，不依赖原D盘模型目录、仓库研究模块或PyTorch。`retained10.npz`包含原149维根音共享骨干、四个DeepChroma性质残差、四个域路由、四个NNLS N头；13份原权重保持不变，不重新拟合。`manifest.json`保留版本和原冻结来源。

环境：Python及requirements.txt列出的NumPy、SciPy、mir_eval。没有音频、专家测试标签或上游CNN权重随包分发。

## 立即运行已暴露的旧开发样例

在本目录运行：

```powershell
python -B predict.py --features example --output my_prediction.json
```

输出JSON和同名LAB；必须使用不存在的输出路径。LAB应与example/expected.lab逐字节相同；此样例仅说明加载与推理一致，不是新的准确率试验。

## Python调用

```python
from predict import Retained10
model = Retained10()
result = model.predict_folder('example')
# 或 model.predict_arrays(features, cells, deep_chroma, energy)
segments = result['segments']
```

输入契约：features.npz的features为n×149 float32；cells.json为n个连续音频格点start/end/chord（chord来自前端估计，非参考标签）；deep.npz的chroma为10Hz的frames×12 float32；energy.npz的energy为n×4 float32。四者必须来自同一音频和时间轴。149维各列、上下文窗口和NNLS时钟详见feature_schema.json；此包是声学特征之后的模型，不把任意矩阵或未经处理的音频当兼容输入。4能量特征为80ms、格点、500ms相对95分位RMS电平和短长差，裁剪/缩放沿用原前端。

推理为四预定种子的后验均值、温度1；p(N)≥0.5输出N；逐格点argmax后只合并相邻相同标签，无新平滑或调参。输入前端显式斜杠低音照原语义保留，不声称新学得低音。域路由使用整段统计，这是离线模型。

支持maj/min/7/maj7/min7/9/maj9/min9/sus2/sus4与N；不输出5/13/dim7，不输出调性或完整调式，词表外声音不自动等同N。概率未校准，不是人工置信度。

已封账19作品114录音对同一十类已解析时长：等作品完整和声69.183%（历史58.141%，加NNLS N基线61.077%）；纯和弦77.734%（18作品，历史68.155%，NNLS N基线67.947%）。已解析时长池化覆盖73.061%；全时长池化准确率边界51.354%–78.292%。这些不是任意歌曲准确率；本包没有重算或再次使用该留出集。N上的误报仍77.474%，不能据此默认部署。

完整实验依据见原项目CHORD_FINAL_HELD19_REVIEW_20261012.md和manifest中的冻结清单摘要。本包可加载性只在一个旧DEV缓存核验，新拟合/更新/录音均0；最终成绩并未因打包产生新验证。应用默认未修改。

随包包含decorator 5.3.1的原样源文件decorator.py及文件头许可证（Michele Simionato），供mir_eval在Python隔离模式下加载。它不改全局Python环境。首次隔离验证曾因该传递依赖不在系统site-packages而失败；神经权重及概率对照未变。
