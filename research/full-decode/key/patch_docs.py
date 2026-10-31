"""Refresh REVIEW/HANDOFF numbers after making the full decoder the default source."""
from pathlib import Path

REVIEW = Path(r"D:\ChordStudio\research\train-v2\REVIEW.md")
HANDOFF = Path(r"D:\ChordStudio\research\HANDOFF.md")

EDITS = {
    REVIEW: [
        ("| 3. 完整段解码器（第二档，可切换） | exact ≥ 0.50、边界 F1 ≥ 0.90 | **0.674** / **0.903** | ✅ | `full-decode/verify_prod_full.py` |",
         "| 3. 完整段解码器（现默认，可切换回 span） | exact ≥ 0.50、边界 F1 ≥ 0.90 | **0.682** / **0.902** | ✅ | `full-decode/verify_prod_full.py` |"),
        ("| `retained10_span`（现默认） | 81 | 0.444 | 0.690 | 0.71 | 0.93 | 0.81 |\n"
         "| **`retained10_full`（第二档，新）** | **64** | **0.674** | **0.758** | **0.889** | **0.918** | **0.903** |",
         "| `retained10_span`（旧默认，现可切换） | 81 | 0.444 | 0.690 | 0.71 | 0.93 | 0.81 |\n"
         "| **`retained10_full`（现默认）** | **62** | **0.682** | **0.774** | **0.902** | **0.902** | **0.902** |"),
        ("相对现默认：完整和弦 **+23.0 个点**、family **+6.8 个点**、边界 F1 **0.81 → 0.903**。",
         "相对 span：完整和弦 **+23.8 个点**、family **+8.4 个点**、边界 F1 **0.81 → 0.902**。\n"
         "和弦音恢复率（golden 62 事件逐段比对音级集合）：span 0.827、stable 0.815、冻结原输出 0.796、\n"
         "**full 0.922**；带扩展音（4 音以上）的比例：真值 0.903、span 0.494、**full 0.806**。\n"
         "复算：`full-decode/tone_recovery.py`。这也是把 full 设为默认的直接理由。"),
        ("- 在应用里它就是第四个可选来源（面板「完整段解码（派生，第二档）」），默认仍是 span；",
         "- 在应用里它是默认来源（面板「完整段解码（派生）」），`retained10_span` 仍可切换；"),
        ("生产版（`app/web/full_decode.py`）与上表最后一行是同一套算法，因实现细节\n"
         "（低音按 cell 池化 vs 按段池化）得到 64 段 / exact 0.674 / F1 0.903，仍满足两个阈值。",
         "生产版（`app/web/full_decode.py`）与上表最后一行是同一套算法；实现细节（低音按 cell 池化）\n"
         "与改进后的音级权重（mu 0.6→0.7）使生产版落到 62 段 / exact 0.682 / family 0.774 / F1 0.902，\n"
         "同时把和弦音恢复率从 0.907 提到 0.922、扩展音比例从 0.746 提到 0.806。"),
        ("| 虚拟 | 177 | 81 | **64** | C / C / C（声学 G，和弦覆盖后 C，正确） |\n"
         "| 坏女孩 | 195 | 106 | **88** | C#m / C#m / C#m（声学 E；关系小调偏差是已知问题，未变） |\n"
         "| 沧海一粟 | 154 | 78 | **73** | G# / G# / G#（= Ab，同一音级，正确） |",
         "| 虚拟 | 177 | 81 | **62** | C / C / C（声学 G，和弦覆盖后 C，正确） |\n"
         "| 坏女孩 | 195 | 106 | **90** | C#m / C#m / C#m（声学 E；关系小调偏差是已知问题，未变） |\n"
         "| 沧海一粟 | 154 | 78 | **73** | G# / G# / G#（= Ab，同一音级，正确） |"),
        ("复算：`research/full-decode/check_other_songs.py`。三首歌上 `full` 都没有改变主调，\n段数比 span 再少 6%–19%。",
         "复算：`research/full-decode/check_other_songs.py`。三首歌上 `full` 都没有改变主调，\n段数比 span 再少 6%–23%（虚拟 81→62、坏女孩 106→90、沧海 78→73）。"),
    ],
    HANDOFF: [
        ("| **`retained10_span`** | **现默认**：小节池化 + 逐小节相位 DP | 81 | 0.444 | 0.690 | 0.81 |\n"
         "| **`retained10_full`** | **第二档（新）**：完整段解码 | **64** | **0.674** | **0.758** | **0.903** |",
         "| `retained10_span` | 小节池化 + 逐小节相位 DP（旧默认，现可切换） | 81 | 0.444 | 0.690 | 0.81 |\n"
         "| **`retained10_full`** | **现默认**：完整段解码 | **62** | **0.682** | **0.774** | **0.902** |"),
        ("1. `default_source` 仍是 `retained10_span`；`retained10_full` 只是**可切换的第二档**；",
         "1. `default_source` 是 **`retained10_full`**（和弦音恢复率 0.922 vs span 0.827）；\n"
         "   `retained10_span` 仍可切换，full 跑不出来时自动退回 span；"),
        ("《虚拟》golden：exact **0.674**、family 0.758、边界 P 0.889 / R 0.918 / **F1 0.903**、64 段（真值 62）。",
         "《虚拟》golden：exact **0.682**、family 0.774、边界 P/R **0.902** / **F1 0.902**、62 段（真值 62）；\n"
         "和弦音恢复率 0.922（span 0.827）、扩展音比例 0.806（真值 0.903）。"),
        ("非目标歌曲（无 golden）：虚拟 177→81→64、坏女孩 195→106→88、沧海 154→78→73，\n三首歌的主调都没有被 `full` 改变。",
         "非目标歌曲（无 golden）：虚拟 177→81→62、坏女孩 195→106→90、沧海 154→78→73，\n三首歌的主调都没有被 `full` 改变。"),
    ],
}

for path, edits in EDITS.items():
    text = path.read_text(encoding="utf-8")
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{path.name}: expected 1 occurrence, found {count} for: {old[:60]}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("updated", path.name, f"({len(edits)} edits)")
