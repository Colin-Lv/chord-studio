#!/usr/bin/env python
"""Collect Roman-numeral progression statistics from benchmark chord references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from chord_studio import chord_to_roman_display
from evaluate_chords import TimedChord, convert_annotation_chord, read_reference


def roman_function_label(label: str) -> str:
    text = label.split("/", 1)[0]
    if text == "N":
        return "N"
    degrees = ["bVII", "#VII", "VII", "bVI", "#VI", "VI", "#IV", "IV", "bIII", "#III", "III", "bII", "II", "V", "I"]
    for degree in degrees:
        if not text.startswith(degree):
            continue
        suffix = text[len(degree) :]
        return f"{degree}m" if suffix.startswith("m") else degree
    return text


def merge_repeats(chords: list[TimedChord], keep_stays: bool = False) -> list[str]:
    labels = []
    previous = None
    for item in chords:
        label = convert_annotation_chord(item.chord)
        if not keep_stays and label == previous:
            continue
        labels.append(label)
        previous = label
    return labels


def roman_sequence(chords: list[str], key: str) -> list[str]:
    return [
        roman_function_label(chord_to_roman_display(chord, key, "N", 0.0, 1.0))
        for chord in chords
        if chord != "N"
    ]


def resolve_path(base: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base / path


def row_allowed(row: dict[str, str], splits: set[str] | None, datasets: set[str] | None) -> bool:
    if splits is not None and row.get("split", "") not in splits:
        return False
    if datasets is not None and row.get("dataset", "") not in datasets:
        return False
    return True


def pair_scores(pair_counts: Counter[tuple[str, str]], alpha: float = 0.25) -> dict[str, float]:
    labels = sorted({label for pair in pair_counts for label in pair})
    outgoing: defaultdict[str, int] = defaultdict(int)
    for (left, _), count in pair_counts.items():
        outgoing[left] += count
    label_count = max(len(labels), 1)
    scores: dict[str, float] = {}
    for left in labels:
        total = outgoing[left]
        if total <= 0:
            continue
        uniform = 1.0 / label_count
        for right in labels:
            count = pair_counts.get((left, right), 0)
            probability = (count + alpha) / (total + alpha * label_count)
            score = math.log(probability / uniform)
            scores[f"{left}->{right}"] = float(max(-1.5, min(1.5, score)))
    return scores


def collect(
    manifest: Path,
    splits: set[str] | None = None,
    datasets: set[str] | None = None,
    keep_stays: bool = False,
    max_ngram: int = 4,
) -> dict[str, object]:
    base = manifest.parent
    pair_counts: Counter[tuple[str, str]] = Counter()
    gram3_counts: Counter[tuple[str, str, str]] = Counter()
    ngram_counts: dict[int, Counter[tuple[str, ...]]] = {size: Counter() for size in range(2, max(2, max_ngram) + 1)}
    item_count = 0
    skipped = 0
    filtered = 0
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row_allowed(row, splits, datasets):
                filtered += 1
                continue
            key = str(row.get("key") or "").strip()
            reference = resolve_path(base, row.get("chord_reference", ""))
            if not key or reference is None or not reference.exists():
                skipped += 1
                continue
            labels = roman_sequence(merge_repeats(read_reference(reference), keep_stays=keep_stays), key)
            if len(labels) < 2:
                skipped += 1
                continue
            item_count += 1
            pair_counts.update(zip(labels, labels[1:]))
            gram3_counts.update(zip(labels, labels[1:], labels[2:]))
            for size, counter in ngram_counts.items():
                if len(labels) < size:
                    continue
                counter.update(tuple(labels[index : index + size]) for index in range(len(labels) - size + 1))
    return {
        "source_manifest": str(manifest),
        "items": item_count,
        "skipped": skipped,
        "filtered": filtered,
        "alpha": 0.25,
        "label_style": "roman_function",
        "keep_stays": keep_stays,
        "max_ngram": max_ngram,
        "pair_scores": pair_scores(pair_counts),
        "pairs": [
            {"from": left, "to": right, "count": count}
            for (left, right), count in pair_counts.most_common()
        ],
        "trigrams": [
            {"a": a, "b": b, "c": c, "count": count}
            for (a, b, c), count in gram3_counts.most_common()
        ],
        "ngrams": {
            str(size): [
                {"pattern": list(pattern), "count": count}
                for pattern, count in counter.most_common(200)
            ]
            for size, counter in ngram_counts.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect progression statistics from benchmark manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmarks/reports/progression_stats.json"))
    parser.add_argument("--split", action="append", help="Only include rows whose split matches this value.")
    parser.add_argument("--dataset", action="append", help="Only include rows whose dataset matches this value.")
    parser.add_argument("--keep-stays", action="store_true", help="Keep adjacent repeated labels, useful for bar-grid phrase priors.")
    parser.add_argument("--max-ngram", type=int, default=4, help="Maximum Roman n-gram length to export.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = collect(
        args.manifest,
        set(args.split) if args.split else None,
        set(args.dataset) if args.dataset else None,
        keep_stays=args.keep_stays,
        max_ngram=args.max_ngram,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"items: {stats['items']}")
    print(f"skipped: {stats['skipped']}")
    print(f"filtered: {stats['filtered']}")
    for item in stats["pairs"][:12]:
        print(f"{item['from']} -> {item['to']}: {item['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
