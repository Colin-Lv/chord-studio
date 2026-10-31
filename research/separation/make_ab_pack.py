"""Export a 30 s model-comparison pack so the separation can be judged outside the app.

For each model: vocals stem and 'everything else' stem, cut from the same chorus section,
same sample range, natural levels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SHA = "d4867eec9bec73fb4dbf0575cff26e3e6550cdba9553c8f886b3a4adab6ef72f"
STEMS = Path(r"D:\ChordStudio\data\stems") / SHA
OUT = Path(r"D:\ChordStudio\data\exports\虚拟_模型对比")
START, LENGTH = 100.0, 30.0
MODELS = [
    ("htdemucs", "标准（快）"),
    ("htdemucs_ft", "精细（四模型）"),
    ("bs_roformer", "人声优先"),
    ("bs_roformer_sw", "六轨精细"),
]
NON_VOCAL = ("drums", "bass", "other", "guitar", "piano", "instrumental")


def load(path: Path):
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    return data, rate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, label in MODELS:
        wav_dir = STEMS / model / "wav"
        vocals_path = wav_dir / "vocals.wav"
        if not vocals_path.is_file():
            print(f"skip {model}: no vocals")
            continue
        vocals, rate = load(vocals_path)
        backing = None
        for name in NON_VOCAL:
            path = wav_dir / f"{name}.wav"
            if path.is_file():
                data, _ = load(path)
                backing = data if backing is None else backing + data
        if backing is None:
            continue
        a, b = int(START * rate), int((START + LENGTH) * rate)
        vocal_clip = vocals[a:b]
        backing_clip = backing[a:b]
        sf.write(str(OUT / f"{label}_人声.wav"), vocal_clip, rate, subtype="PCM_16")
        sf.write(str(OUT / f"{label}_伴奏.wav"), backing_clip, rate, subtype="PCM_16")
        db = lambda x: 20 * np.log10(float(np.sqrt((x.astype(np.float64) ** 2).mean())) + 1e-12)
        rows.append((model, label, round(db(vocal_clip), 1), round(db(backing_clip), 1),
                     round(float(np.abs(vocal_clip).max()), 2), round(float(np.abs(backing_clip).max()), 2)))
        print(f"{label:12s} vocals {rows[-1][2]:6.1f} dBFS  backing {rows[-1][3]:6.1f} dBFS")

    lines = [
        "# 《虚拟》分离模型对比包",
        "",
        f"片段：{START:.0f}s – {START + LENGTH:.0f}s（副歌，鼓与贝斯在场）",
        "",
        "| 模型 | 人声 RMS | 伴奏 RMS | 人声峰值 | 伴奏峰值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, label, v, b, vp, bp in rows:
        lines.append(f"| {label} | {v} dBFS | {b} dBFS | {vp} | {bp} |")
    lines += [
        "",
        "听法：同一时间点来回切人声与伴奏两组文件，重点听",
        "",
        "1. 人声里的钢琴/混响残留（越少越好）",
        "2. 伴奏里的人声残留（尤其副歌长音）",
        "3. 钢琴有没有被削薄（对比「标准」与「六轨精细」最明显）",
        "",
        "注意：各模型电平略有差异，比较干净度时不要被响度带偏。",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {len(rows) * 2} wav files + README.md to {OUT}")


if __name__ == "__main__":
    main()

