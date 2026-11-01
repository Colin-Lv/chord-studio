"""Where do the substituted inputs differ from the original? Count fixes vs breaks per golden segment."""
from __future__ import annotations

import json
from pathlib import Path

WORK = Path(r"D:\ChordStudio\research\separation\stem-input")
GOLDEN = Path(r"D:\ChordStudio\Dialogue\research\golden\reference_golden.lab")
STEP = 0.25


def read_lab(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            rows.append((float(parts[0]), float(parts[1]), parts[2]))
    return rows


def label_at(rows, t):
    for start, end, label in rows:
        if start <= t < end:
            return label
    return None


def majority(rows, start, end):
    counts = {}
    t = start
    while t < end:
        guess = label_at(rows, t)
        if guess is not None:
            counts[guess] = counts.get(guess, 0) + 1
        t += STEP
    return max(counts, key=counts.get) if counts else None


def main():
    golden = read_lab(GOLDEN)
    base = read_lab(WORK / "A_original.lab")
    results = {}
    for name in ("B_accomp_plus_bass", "C_accomp_only", "D_no_vocals"):
        pred = read_lab(WORK / f"{name}.lab")
        fixed = broken = same_right = same_wrong = 0
        examples = []
        for start, end, truth in golden:
            if end - start < 0.1:
                continue
            a = majority(base, start, end)
            b = majority(pred, start, end)
            if a == truth and b == truth:
                same_right += 1
            elif a != truth and b == truth:
                fixed += 1
                examples.append((round(start, 1), truth, a, b))
            elif a == truth and b != truth:
                broken += 1
            else:
                same_wrong += 1
        results[name] = dict(fixed=fixed, broken=broken, same_right=same_right,
                             same_wrong=same_wrong, examples=examples[:8])
        print(f"{name:>22s}  fixed {fixed:3d}  broken {broken:3d}  "
              f"both-right {same_right:3d}  both-wrong {same_wrong:3d}")
        for example in examples[:5]:
            print(f"    at {example[0]:6.1f}s truth {example[1]:8s} base {example[2]:8s} -> {example[3]}")
    (WORK / "diff-analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
