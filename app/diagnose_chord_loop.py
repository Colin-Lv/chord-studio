from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def normalize_label(label: str, *, strip_bass: bool) -> str:
    label = (label or "N").strip()
    if strip_bass and "/" in label:
        label = label.split("/", 1)[0]
    return label or "N"


def read_units(path: Path, field: str, strip_bass: bool) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    units: list[dict[str, str]] = []
    for row in rows:
        label = normalize_label(row.get(field, "") or row.get("roman", "") or row.get("chord", ""), strip_bass=strip_bass)
        units.append(
            {
                "bar": row.get("bar", ""),
                "beat_start": row.get("beat_start", ""),
                "beat_end": row.get("beat_end", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
                "label": label,
                "chord": normalize_label(row.get("chord_display", "") or row.get("chord", ""), strip_bass=strip_bass),
                "roman": normalize_label(row.get("roman_display", "") or row.get("roman", ""), strip_bass=strip_bass),
            }
        )
    return units


def cycle_score(labels: list[str], pattern: tuple[str, ...]) -> float:
    if not labels:
        return 0.0
    matches = 0
    for idx, label in enumerate(labels):
        if label == pattern[idx % len(pattern)]:
            matches += 1
    return matches / len(labels)


def best_cycles(labels: list[str], cycle_len: int, max_offset: int) -> list[tuple[float, int, tuple[str, ...], int]]:
    candidates: list[tuple[float, int, tuple[str, ...], int]] = []
    for offset in range(min(max_offset, max(cycle_len, len(labels)))):
        clipped = labels[offset:]
        if len(clipped) < cycle_len * 2:
            continue
        buckets: list[list[str]] = [[] for _ in range(cycle_len)]
        for idx, label in enumerate(clipped):
            if label != "N":
                buckets[idx % cycle_len].append(label)
        pattern = tuple((Counter(bucket).most_common(1)[0][0] if bucket else "N") for bucket in buckets)
        score = cycle_score(clipped, pattern)
        candidates.append((score, offset, pattern, len(clipped)))
    candidates.sort(reverse=True, key=lambda item: (item[0], item[3]))
    return candidates


def print_window(units: list[dict[str, str]], offset: int, count: int, field: str) -> None:
    print(f"\nwindow from unit offset {offset}:")
    for idx, unit in enumerate(units[offset : offset + count], start=offset + 1):
        print(
            f"{idx:03d}  bar {unit['bar']:>3} beat {unit['beat_start']}-{unit['beat_end']:<2} "
            f"{unit['start']}-{unit['end']:<5}  {unit[field]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Find repeated chord/progression loops in chord_studio CSV output.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--field", choices=["roman", "chord", "label"], default="roman")
    parser.add_argument("--cycle-len", type=int, default=4)
    parser.add_argument("--max-offset", type=int, default=16)
    parser.add_argument("--strip-bass", action="store_true", help="Treat slash chords as their upper chord/root function.")
    parser.add_argument("--show", type=int, default=24)
    args = parser.parse_args()

    units = read_units(args.csv_path, args.field, args.strip_bass)
    labels = [unit[args.field] for unit in units]
    cycles = best_cycles(labels, args.cycle_len, args.max_offset)

    print(f"file: {args.csv_path}")
    print(f"units: {len(units)}")
    print(f"field: {args.field}")
    print(f"cycle_len: {args.cycle_len}")
    print("\nbest cycle candidates:")
    for score, offset, pattern, n_units in cycles[:8]:
        print(f"  score={score:.3f}  offset={offset:02d}  units={n_units:03d}  pattern={' | '.join(pattern)}")

    if cycles:
        _, offset, _, _ = cycles[0]
        print_window(units, offset, args.show, args.field)


if __name__ == "__main__":
    main()
