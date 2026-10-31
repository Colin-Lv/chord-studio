# PLAN — 和弦识别阶段三（收口）

截止：2026-10-31 14:35。每项都有可执行验收命令；验收不过不标 ✅。

## 阶段 1：合成域完整和弦 ≥ 0.95

- 目标：按歌曲切分的 held-out 完整和弦（根音+性质）≥ 95%，七和弦分项 ≥ 85%
- 数据：`research/train-v2/data/chords_full_fixed.npz`（36 598 patch，60 通道）
- 验收：
  ```powershell
  & "D:\ChordStudio\Dialogue\.tmp\structure_env\Scripts\python.exe" `
      D:\ChordStudio\research\train-v2\verify_synthetic.py fix60
  ```
- 结果：**0.9695（TTA）/ 0.9666（单次）**，七和弦 0.915–0.934 → ✅

## 阶段 2：147 例回归不退化

- 目标：`retained10_span` 作为默认后，147 例的最终主调不退化
- 验收：
  ```powershell
  & $py D:\ChordStudio\research\full-decode\regress147\summarise.py
  & $py D:\ChordStudio\research\full-decode\regress147\verify_key_pinning.py
  ```
- 结果：先测得 −1 例（6 好 7 坏）→ 主调证据钉在 stable → **105 == 105，0 退化** → ✅

## 阶段 3：完整段解码器接入生产（第二档可切换）

- 目标：《虚拟》完整和弦 ≥ 0.50、边界 F1 ≥ 0.90；作为可切换来源，默认不变
- 验收：
  ```powershell
  # 重启应用后重新分析《虚拟》
  & $py D:\ChordStudio\research\full-decode\rerun_xuni.py
  & $py D:\ChordStudio\research\full-decode\verify_prod_full.py
  ```
- 结果：**exact 0.674 / 边界 F1 0.903**；`analysis.json.sources` 含 `retained10_full`（64 行），
  `default_source` 仍是 `retained10_span`，`provenance.full_decode.applied = true` → ✅

## 阶段 4：证据落地

- 目标：每项结论都能用脚本复算，写进 `REVIEW.md`
- 验收：`& $py D:\ChordStudio\research\full-decode\verify_all.py` 打印全 PASS 且退出码 0
- 结果：✅（REVIEW.md §1–§3 与历史记录都在）

## 反早停条款

只有 `verify_all.py` 全绿、且 REVIEW.md 里每项都有脚本与数字，才允许 `update_goal complete`。
禁止用「基本完成」「剩余很小」收尾。
