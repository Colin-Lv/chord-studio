# 音轨分离（分轨）方案与实现计划

日期：2026-11-01。目标：在 Chord Studio 内新增「音轨分离」能力与一个类 FL Studio 的多轨网格页面，
分离与和声识别作为两个独立按钮。

## 1. 调研结论

### 1.1 可选方案

| 方案 | 代表实现 | 干声质量（MUSDB18-HQ SDR，均值） | 权重/依赖 | 本机可行性 |
|---|---|---|---|---|
| **HTDemucs v4** | `demucs` 4.1.0（MIT） | ≈9.0 dB（drums 10.4 / bass 10.5 / other 7.2 / vocals 8.0） | 80 MB 单模型，torch | ✅ 已装、**权重已缓存** |
| HTDemucs FT | 同仓库 `htdemucs_ft` | ≈9.2 dB（四模型微调，慢 4×） | 4×80 MB | ⚠️ 需下载、CPU 约 6 分钟/首 |
| BS-RoFormer / Mel-RoFormer | MSST / audio-separator | ≈10.3–10.9 dB（当前 SOTA 级） | 数百 MB–GB，显存要求高 | ❌ 6 GB 笔记本 + CPU-only torch 风险高 |
| MDX-Net (UVR) | audio-separator / ONNX | 介于两者之间，人声强 | ONNX 运行时可用 | ⚠️ 需下载模型，CPU 速度未知 |
| Spleeter | Deezer | ≈5.9 dB | 轻 | ❌ 质量明显落后 |

参考：[Demucs v4 论文（Hybrid Transformers for Music Source Separation）](https://arxiv.org/abs/2211.08553)、
[demucs 仓库](https://github.com/facebookresearch/demucs)、
[MSST 预训练模型清单](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)、
[demucs ONNX 导出](https://github.com/sevagh/demucs.onnx)。

### 1.2 本机实测（2026-11-01）

| 项 | 结果 |
|---|---|
| `demucs 4.1.0` | `structure_env` / `harmony_web_env` / `miniconda` 均已装（torch 2.11 CPU） |
| htdemucs 权重 | `~/.cache/torch/hub/checkpoints/955717e8-8726e21a.th`（80 MB）已存在 |
| CPU 速度（60 s 片段，24 线程） | **24.3 s**（RTF 0.40），一首 4 分钟约 90–100 s |
| GPU | RTX 4050 6 GB，但后端环境只有 CPU 版 torch；不为此重装 torch |
| 六轨（`htdemucs_6s`） | 钢琴/吉他分离质量差，且需下载；首版不用 |

### 1.3 选型

**首版用 `htdemucs`（四轨：人声 / 鼓 / 贝斯 / 其他），CPU 推理，后台任务 + 进度。**
理由：质量在可接受区间、权重已在本地、不新增依赖、不占用 GPU 训练环境。
预留升级位：`--model htdemucs_ft`（质量优先）与 ONNX/BS-RoFormer（后续阶段），接口不变。

## 2. 产品设计

### 2.1 入口

项目页新增两个并列按钮：

- **和声识别**（现有分析流程，不改）
- **音轨分离**（新）→ 提交分轨任务，完成后进入「分轨」页面

两件事互不依赖：只做分轨也能听到分轨与网格；只做和声识别完全不受影响。

### 2.2 分轨页面（`/stems?project=<id>`）

类 FL Studio 的多轨网格：

```
┌ 运输条：▶ ⏸ | 00:42 / 04:33 | BPM 130.4 | A 大调 | 小节 23 ─────────┐
├ 轨道行 ×4：颜色块 + 名称 + 乐器 + 电平 + [M] [S] ────────────────────┤
│  人声   ▓▓▓▓░░▓▓▓▓░░  ← 每小节一个块，亮度=该小节能量 ──────────────┤
│  鼓     ░░░░▓▓▓▓▓▓▓▓                                                  │
│  贝斯   ▓▓▓▓▓▓▓▓▓▓▓▓                                                  │
│  其他   ▓▓▓▓▓▓▓▓▓▓▓▓                                                  │
└───────────────────────────────────────────────────────────────────────┘
   小节刻度 1  2  3  4 ...（点击格子 = 跳到该小节并从那里播放）
```

交互：
- 点击轨道头：静音/取消；`S` 独奏（互斥）；音量推子
- 点击任意格子：定位到该小节并播放
- 空格播放/暂停；← → 前后 5 秒
- 当前小节高亮；播放头随小节移动
- 每轨信息面板：名称、电平、活跃占比、首次/最后出现、主要音区、（贝斯）主音

### 2.3 自动标注

| 字段 | 来源 |
|---|---|
| 轨道名/乐器 | 模型固定四轨 + 中文名与颜色 |
| 时长、采样率 | 解码信息 |
| BPM | 优先用项目 `beats.json`（madmom）；没有则 librosa 估计并标注「估计」 |
| 小节网格 | 同上；没有则按 BPM 均分 |
| 每小节能量 | 每轨按小节求 RMS → 归一化 |
| 活跃区间 | 能量阈值 → 首/末时间与占比 |
| 主要音区 | 谱质心 → 低/中/高 |
| 贝斯主音 | 低音区 chroma 的 argmax（仅标注，不参与和声识别） |
| 调性 | 有 `key.json` 时显示，否则空 |

## 3. 后端设计

### 3.1 数据流

```
POST /api/projects/{id}/separate
  → jobs.submit(profile="stems")
  → app.web.worker 分派到 app/web/stems_worker.py
      1. 定位项目音频，算 sha256
      2. 缓存命中？ <data>/stems/<sha>/manifest.json
      3. model_python -m demucs.separate -n htdemucs --out tmp
      4. ffmpeg 转 mp3（192 kbps，44.1 kHz）供浏览器播放
      5. 逐轨分析：RMS/peak/活跃度/谱质心/贝斯主音/每小节能量
      6. BPM 与小节：优先复用 run/beats.json
      7. 写 manifest.json（原子写）
  → 完成：project.stems = {...}，不改 current_run
```

### 3.2 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/projects/{id}/separate` | 提交分轨任务 |
| GET | `/api/projects/{id}/stems` | 读 manifest（没有则 404 + 提示） |
| GET | `/api/projects/{id}/stems/{file}` | 音频（mp3）/ 可选 wav 下载 |
| GET | `/stems` | 分轨页面 HTML |

### 3.3 缓存与磁盘

`<CHORD_DESKTOP_DATA>/stems/<audio_sha>/`：`manifest.json`、`vocals.mp3` … `other.mp3`、
`wav/*.wav`（保留无损，供后续低音/音高分析）。同一音频重复点击直接复用。

## 4. 前端设计

- `frontend/stems.html` + `stems.js` + `stems.css`，与主工作台同一套配色
- 播放：4 个 `<audio>` 元素共享一条运输线；播放/暂停/跳转时统一 `currentTime` 校准
- 网格：DOM 表格（行=轨道，列=小节），格子亮度来自 manifest；当前小节加边框高亮
- 键盘：空格播放/暂停，左右键 ±5 s
- 失败/进度：复用现有 job 轮询，显示阶段与百分比

## 5. 验收（第一版）

1. 点「音轨分离」后 ≤3 分钟出结果（4 分钟歌曲，CPU）；
2. `/stems?project=...` 打开可播放 4 轨，静音/独奏/跳转小节均生效，多轨同步无爆音；
3. 页面显示 BPM、调性、每轨电平与活跃占比；
4. 网格每小节有块，点击可定位；
5. 分离结果在《虚拟》上主观可听：人声/鼓/贝斯/伴奏各自可辨；
6. 原有「和声识别」流程与默认来源不受影响（`verify_all.py` 仍全绿）。

## 6. 里程碑

| 阶段 | 内容 | 产物 |
|---|---|---|
| M1 | 后端 worker + manifest + 接口 | `stems_worker.py`，一次完整分轨 |
| M2 | 前端页面与网格 | `/stems` 可播放 |
| M3 | 入口按钮 + 文案 + 错误处理 | 项目页两个按钮 |
| M4 | 《虚拟》实跑 + 指标 + 截图 | 验收记录 |

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| CPU 分离慢（4 分钟歌 ≈ 100 s） | 后台任务 + 进度；缓存复用；预留 `htdemucs_ft`/GPU 升级位 |
| 浏览器同时播 4 轨不同步 | 统一 `currentTime` 校准 + `seeked` 事件对齐 |
| 4×mp3 带宽/内存 | mp3 192k、按需加载；`preload="auto"` 仅 4 个音频 |
| 分离质量主观 | 首版先给 `htdemucs`，听感不足再换 `htdemucs_ft`/ONNX |
| 项目无 beats（只分轨） | librosa 估计 BPM + 均分小节，标注「估计」 |
