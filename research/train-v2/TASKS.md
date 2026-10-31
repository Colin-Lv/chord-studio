# 训练工作区（阶段二）

## 现状与瓶颈（2026-10-31 07:5x）

| 环节 | 现状 | 瓶颈 |
|---|---|---|
| 小节网格 | 已修相位（83.05–146.75 s 半小节翻转） | 已解决 |
| 边界判定 | 分类器 + DP：F1 0.93（同曲）/ 0.80（跨域） | 跨域精确率低 |
| 和弦性质 | 模板 0.565（弱先验）/ 生产模型+池化 0.533 | **扩展音与性质判断** |
| 合成数据上限 | 伴奏 stem + 模板 = 99.2%（三和弦为主） | 9 音缺失 |

**结论：瓶颈在“性质/扩展音”这一层，且简单模板与浅层分类器在真实录音上明显掉点。**

## 本轮任务

1. 数据：`boundary-dataset`（144 条，无 9 音）+ `extension-dataset`（48 条，约 45% 和弦扩展为 7/9 音）。
2. 模型：帧级 CNN（CENS chroma → 根音 + 性质），在 GPU 上训练，按歌曲切分验证。
3. 目标：合成集 held-out 完整和弦 ≥95%；再测《虚拟》观察跨域。
4. 若有效，接入生产（阶段目标 3）。

## 目录

```
train-v2/
├── TASKS.md        本文件
├── prepare.py      提取帧级特征（CENS 8 fps，定长 64 帧）→ data/*.npz
├── train.py        帧级 CNN 训练（GPU）
├── eval.py         合成 held-out + 虚拟 评测
└── runs/           模型与日志
```

## 复现

```powershell
$py = "D:\ChordStudio\runtime\gpu\Scripts\python.exe"
& $py train-v2\prepare.py
& $py train-v2\train.py
& $py train-v2\eval.py
```
