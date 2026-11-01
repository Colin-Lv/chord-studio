"""Score the three input variants: frame-level within the golden intervals, then whole segmentation."""
from __future__ import annotations

import json
from pathlib import Path

GOLDEN = Path(r"D:\ChordStudio\Dialogue\research\golden\reference_golden.lab")
WORK = Path(r"D:\ChordStudio\research\separation\stem-input")
STEP = 0.25
TOL = 0.5


def read_lab(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            rows.append((float(parts[0]), float(parts[1]), parts[2]))
    return rows


def label_at(rows, t):
    lo, hi = 0, len(rows) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, label = rows[mid]
        if t < start:
            hi = mid - 1
        elif t >= end:
            lo = mid + 1
        else:
            return label
    return None


def frame_scores(pred, golden):
    start, end = golden[0][0], golden[-1][1]
    total = hit = root = 0
    t = start
    while t < end:
        truth = label_at(golden, t)
        guess = label_at(pred, t)
        if truth is not None:
            total += 1
            hit += guess == truth
            root += (guess or "N").split(":")[0] == truth.split(":")[0]
        t += STEP
    return hit / max(1, total), root / max(1, total), total


def segment_scores(pred, golden):
    seg_total = seg_hit = 0
    for start, end, truth in golden:
        if end - start < 0.1:
            continue
        counts = {}
        t = start
        while t < end:
            guess = label_at(pred, t)
            if guess is not None:
                counts[guess] = counts.get(guess, 0) + 1
            t += STEP
        if not counts:
            continue
        seg_total += 1
        seg_hit += max(counts, key=counts.get) == truth
    return seg_hit / max(1, seg_total), seg_total


def boundary_f1(pred, golden, tol=TOL):
    def starts(rows):
        return [row[0] for row in rows if row[0] > 0.05]

    p, g = starts(pred), starts(golden)
    used = set()
    matched = 0
    for value in g:
        best = None
        for index, candidate in enumerate(p):
            if index in used:
                continue
            distance = abs(candidate - value)
            if distance <= tol and (best is None or distance < best[0]):
                best = (distance, index)
        if best is not None:
            used.add(best[1])
            matched += 1
    precision = matched / max(1, len(p))
    recall = matched / max(1, len(g))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return precision, recall, f1, len(p), len(g)


def main():
    golden = read_lab(GOLDEN)
    print(f"golden: {len(golden)} segments, {golden[0][0]:.2f}-{golden[-1][1]:.2f}s")
    rows = []
    for path in sorted(WORK.glob("*.lab")):
        pred = read_lab(path)
        exact, root, frames = frame_scores(pred, golden)
        seg_exact, seg_total = segment_scores(pred, golden)
        precision, recall, f1, count, golden_count = boundary_f1(pred, golden)
        rows.append(dict(condition=path.stem, frame=exact, root=root, segment=seg_exact,
                         segments=len(pred), boundary_f1=f1, boundary_p=precision,
                         boundary_r=recall, frames=frames, golden_segments=seg_total))
        print(f"{path.stem:>22s}  frame {exact * 100:5.2f}  root {root * 100:5.2f} | "
              f"segment {seg_exact * 100:5.2f} | segs {len(pred):4d} | "
              f"boundary P {precision * 100:5.2f} R {recall * 100:5.2f} F1 {f1 * 100:5.2f}")
    (WORK / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
