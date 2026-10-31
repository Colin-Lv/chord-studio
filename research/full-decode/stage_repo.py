"""Stage the research side, README and gitignore for the chord-studio repository."""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(r"D:\ChordStudio\Dialogue")
RESEARCH = Path(r"D:\ChordStudio\research")

COPY_FILES = [
    (RESEARCH / "HANDOFF.md", "research/HANDOFF.md"),
    (RESEARCH / "train-v2/PLAN.md", "research/train-v2/PLAN.md"),
    (RESEARCH / "train-v2/REVIEW.md", "research/train-v2/REVIEW.md"),
    (RESEARCH / "train-v2/STATE.md", "research/train-v2/STATE.md"),
    (RESEARCH / "train-v2/TASKS.md", "research/train-v2/TASKS.md"),
    (RESEARCH / "key-v3/chord_seg/reference_golden.lab", "research/golden/reference_golden.lab"),
    (RESEARCH / "key-v3/chord_seg/reference_golden.json", "research/golden/reference_golden.json"),
    (RESEARCH / "boundary-dataset/generate.py", "research/boundary-dataset/generate.py"),
    (RESEARCH / "boundary-dataset/render.py", "research/boundary-dataset/render.py"),
    (RESEARCH / "boundary-dataset/check.py", "research/boundary-dataset/check.py"),
    (RESEARCH / "boundary-dataset/split_bass.py", "research/boundary-dataset/split_bass.py"),
    (RESEARCH / "boundary-dataset/render_all_stems.py", "research/boundary-dataset/render_all_stems.py"),
    (RESEARCH / "boundary-dataset/REPORT.md", "research/boundary-dataset/REPORT.md"),
    (RESEARCH / "boundary-dataset/check_report.md", "research/boundary-dataset/check_report.md"),
]

COPY_DIRS = [
    (RESEARCH / "train-v2", "research/train-v2", {".py"}),
    (RESEARCH / "full-decode", "research/full-decode", {".py", ".js", ".md", ".json"}),
    (RESEARCH / "full-decode/key", "research/full-decode/key", {".py", ".js", ".md", ".json"}),
    (RESEARCH / "full-decode/regress147", "research/full-decode/regress147", {".py", ".json"}),
]

SKIP_NAMES = {"cases", "data", "runs", "__pycache__", "nnls_cache", "audio", "audio_fixed",
              "midi_fixed", "features", "stems", "cache"}
MAX_BYTES = 2 * 1024 * 1024

README = """# Chord Studio

本地和声工作台：从一首歌里读出**和弦时间轴、主调、段落结构**，并把结果交给人校对与导出。
Windows 桌面应用（FastAPI 后端 + 原生 JS 前端 + 冻结的和弦模型）。

## 这一版做了什么

- **完整段解码（默认和弦来源）**：小节相位 DP → NNLS 音符证据的边界分类器 → 半马尔可夫时长先验 →
  低音/音级支持重判每段和弦。相对旧的逐 cell 输出，段数 279 → 62，完整和弦 0.383 → 0.682，
  边界 F1 0.12 → 0.902。
- **连续评分主调（`key_mode=continuous`）**：24 候选连续评分（声学先验 + 和声解释代价 + 主和弦占比 +
  终止/回归计数），严格语义解析，界面显示分项与两条已知退步；API 默认仍是 `madmom`。
- **合成域质量头**：修复 arpeggio 渲染丢音后，CENS + CQT + NNLS 三音区 60 通道、5 折按歌曲切分，
  held-out 完整和弦 **0.9695**（TTA），七和弦 0.915–0.934。
- **147 例主调回归**：把主调证据钉在冻结时间轴上，修复了「主调随显示来源漂移」的退化。

指标全部有脚本复算（见下）。《虚拟》golden 62 事件：

| 输出 | 段数 | 完整和弦 | family | 边界 F1 |
|---|---:|---:|---:|---:|
| 冻结模型逐 cell | 279 | 0.383 | 0.606 | 0.12 |
| 稳定时间轴 | 177 | 0.401 | 0.637 | 0.30 |
| 小节池化（旧默认） | 81 | 0.444 | 0.690 | 0.81 |
| **完整段解码（现默认）** | **62** | **0.682** | **0.774** | **0.902** |

## 结构

```
app/                后端（FastAPI）：识别 worker、时间轴派生、主调裁决、审阅接口
app/web/            桌面端 API 与任务队列（retained_worker / span_pool / full_decode / key_experimental）
frontend/           工作台前端（时间轴、来源切换、主调面板、和弦编辑器）
inference/          冻结模型的推理入口与特征前端（audio_predict / frontend_math / madmom_helper）
research/           训练与解码研究：合成数据生成、60 通道训练、完整段解码、黄金标注、验收脚本
docs/               设计说明与主调审计报告（含错误集合与防复发要求）
tests/              语义与接口测试
```

## 运行（Windows）

应用以打包形式分发（`backend/desktop_backend.py` 启动本地服务，前端由 `/assets` 提供）。
源码方式运行需要一个装了 `fastapi/uvicorn/numpy/scipy/sklearn/librosa` 的 Python 3.13 环境，
再按 `desktop_runtime.json` 指向本机的 ffmpeg / sonic-annotator / NNLS-Chroma 插件 / madmom 环境。

## 复算

```powershell
# 合成域验收（joint >= 0.95、七和弦 >= 0.85）
python research/train-v2/verify_synthetic.py fix60
# 完整段解码器 + 生产接线
python research/full-decode/verify_prod_full.py
# 全部验收
python research/full-decode/verify_all.py
# 和弦音恢复率（各来源对比）
python research/full-decode/tone_recovery.py
```

## 已知限制

- 完整段解码器的参数是在《虚拟》一个 golden 上选的；另外两首只做了稳健性检查，没有逐段真值。
- 连续评分主调的两个已知退步（191347 D#m→A#m、853834 Em→E）未修；分数未校准，147 例是已暴露回归集。
- 真实录音的七/九音仍会丢约 10 个百分点（扩展音比例 0.806 vs 真值 0.903）。
- 合成域模型只用于研究验收，没有进入生产解码。
"""

GITIGNORE_EXTRA = """
# Research datasets, caches and generated artifacts
research/**/data/
research/**/runs/
research/**/cases/
research/**/audio/
research/**/audio_fixed/
research/**/midi_fixed/
research/**/nnls_cache/
research/**/*.npz
research/**/*.joblib
research/**/*.pt
research/**/*.png
research/**/sweep_*.json
research/**/candidate*.json
"""


def main():
    copied = 0
    for source, rel in COPY_FILES:
        if not source.is_file():
            print("missing:", source)
            continue
        target = REPO / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    for source_root, rel_root, suffixes in COPY_DIRS:
        if not source_root.is_dir():
            continue
        for path in source_root.iterdir():
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if path.name in SKIP_NAMES or path.stat().st_size > MAX_BYTES:
                continue
            target = REPO / rel_root / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    (REPO / "README.md").write_text(README, encoding="utf-8")
    ignore = REPO / ".gitignore"
    text = ignore.read_text(encoding="utf-8")
    if "Research datasets, caches" not in text:
        ignore.write_text(text.rstrip() + "\n" + GITIGNORE_EXTRA, encoding="utf-8")
    print(f"copied {copied} research files, wrote README.md, updated .gitignore")


if __name__ == "__main__":
    main()
