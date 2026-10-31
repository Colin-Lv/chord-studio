#!/usr/bin/env python
"""Build a Roman-numeral progression prior from user-corrected analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from progression_stats import pair_scores, roman_function_label


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def iter_analysis_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.rglob("*_analysis.json")))
        elif path.exists() and path.suffix.lower() == ".json":
            result.append(path)
    return result


def normalize_roman(value: object, keep_extensions: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return "N"
    if not keep_extensions:
        return roman_function_label(text)
    body = text.split("/", 1)[0]
    if body == "N":
        return "N"
    degrees = ["bVII", "#VII", "VII", "bVI", "#VI", "VI", "#IV", "IV", "bIII", "#III", "III", "bII", "II", "V", "I"]
    for degree in degrees:
        if not body.startswith(degree):
            continue
        suffix = body[len(degree) :]
        minor = "m" if suffix.startswith("m") else ""
        extension_suffix = suffix[1:] if minor else suffix
        extension = ""
        for marker in ("maj7", "m7b5", "m7", "7", "dim", "aug", "sus4", "sus2", "add9", "9", "6"):
            if marker in extension_suffix:
                extension = marker
                break
        return f"{degree}{minor}{extension}"
    return body


def sequence_from_analysis(path: Path, edited_only: bool, keep_extensions: bool) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    spans: list[dict[str, Any]] = []
    for bar in payload.get("bars", []):
        for span in bar.get("spans", []):
            if not isinstance(span, dict):
                continue
            if edited_only and not span.get("edited"):
                continue
            spans.append(span)
    spans.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    labels = [normalize_roman(item.get("roman_display") or item.get("roman"), keep_extensions) for item in spans]
    return [label for label in labels if label != "N"]


def label_at_time(spans: list[dict[str, Any]], time: float, keep_extensions: bool) -> str:
    if not spans:
        return "N"
    for span in spans:
        start = float(span.get("start", 0.0) or 0.0)
        end = float(span.get("end", start) or start)
        if start <= time < end:
            return normalize_roman(span.get("roman_display") or span.get("roman"), keep_extensions)
    nearest = min(
        spans,
        key=lambda span: abs(
            ((float(span.get("start", 0.0) or 0.0) + float(span.get("end", 0.0) or 0.0)) / 2.0) - time
        ),
    )
    return normalize_roman(nearest.get("roman_display") or nearest.get("roman"), keep_extensions)


def bar_slot_sequences_from_analysis(
    path: Path,
    edited_only: bool,
    keep_extensions: bool,
    slots_per_bar_values: list[int],
) -> dict[int, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {slots: [] for slots in slots_per_bar_values}
    for bar in sorted(payload.get("bars", []), key=lambda item: int(item.get("bar", 0) or 0)):
        spans = [span for span in bar.get("spans", []) if isinstance(span, dict)]
        if edited_only:
            edited_spans = [span for span in spans if span.get("edited")]
            if edited_spans:
                spans = edited_spans
        start = float(bar.get("start", 0.0) or 0.0)
        end = float(bar.get("end", start) or start)
        duration = max(end - start, 0.001)
        for slots in slots_per_bar_values:
            for index in range(slots):
                center = start + duration * ((index + 0.5) / slots)
                result[slots].append(label_at_time(spans, center, keep_extensions))
    return result


def sequences_from_corrections_csv(path: Path, keep_extensions: bool) -> list[list[str]]:
    grouped: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = normalize_roman(row.get("roman"), keep_extensions)
            if label == "N":
                continue
            analysis = str(row.get("analysis") or path.name)
            try:
                start = float(row.get("start") or 0.0)
                end = float(row.get("end") or start)
            except ValueError:
                start = 0.0
                end = start
            grouped[analysis].append((start, end, label))
    sequences = []
    for rows in grouped.values():
        rows.sort(key=lambda item: (item[0], item[1]))
        sequences.append([label for _, _, label in rows])
    return sequences


def compressed(labels: list[str], keep_stays: bool) -> list[str]:
    if keep_stays:
        return labels
    result = []
    previous = None
    for label in labels:
        if label == previous:
            continue
        result.append(label)
        previous = label
    return result


def collect_sequences(
    analysis_paths: list[Path],
    correction_csvs: list[Path],
    edited_only: bool,
    keep_stays: bool,
    keep_extensions: bool,
) -> tuple[list[list[str]], dict[int, list[list[str]]]]:
    sequences = []
    bar_sequences_by_slots: dict[int, list[list[str]]] = {1: [], 2: [], 4: []}
    for path in analysis_paths:
        labels = compressed(sequence_from_analysis(path, edited_only, keep_extensions), keep_stays)
        if len(labels) >= 2:
            sequences.append(labels)
        for slots, slot_labels in bar_slot_sequences_from_analysis(path, edited_only, keep_extensions, [1, 2, 4]).items():
            slot_labels = [label for label in slot_labels if label]
            if len(slot_labels) >= slots * 2:
                bar_sequences_by_slots[slots].append(slot_labels)
    for path in correction_csvs:
        for labels in sequences_from_corrections_csv(path, keep_extensions):
            labels = compressed(labels, keep_stays)
            if len(labels) >= 2:
                sequences.append(labels)
    return sequences, bar_sequences_by_slots


def top_cycles(sequences: list[list[str]], lengths: list[int]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for labels in sequences:
        for length in lengths:
            if length <= 1 or len(labels) < length:
                continue
            for index in range(0, len(labels) - length + 1):
                window = tuple(labels[index : index + length])
                if "N" not in window:
                    counts[window] += 1
    return [
        {"pattern": list(pattern), "count": count}
        for pattern, count in counts.most_common(40)
    ]


def roman_degree_code(label: str, drop_minor: bool = False) -> str:
    text = str(label or "N").split("/", 1)[0]
    if text == "N":
        return "N"
    degree_map = {
        "bVII": "b7",
        "#VII": "#7",
        "VII": "7",
        "bVI": "b6",
        "#VI": "#6",
        "VI": "6",
        "#IV": "#4",
        "IV": "4",
        "bIII": "b3",
        "#III": "#3",
        "III": "3",
        "bII": "b2",
        "II": "2",
        "V": "5",
        "I": "1",
    }
    for degree, code in degree_map.items():
        if not text.startswith(degree):
            continue
        suffix = text[len(degree) :]
        minor = suffix.startswith("m")
        extension_suffix = suffix[1:] if minor else suffix
        if drop_minor:
            minor_text = ""
        else:
            minor_text = "m" if minor else ""
        extension = ""
        if "9sus4" in extension_suffix:
            extension = "9sus4"
        elif "sus4" in extension_suffix:
            extension = "sus4"
        elif "maj7" in extension_suffix:
            extension = "maj7"
        elif "7" in extension_suffix:
            extension = "7"
        elif "6" in extension_suffix:
            extension = "6"
        return f"{code}{minor_text}{extension}"
    return text


def phrase_pattern_rows(
    sequences_by_slots: dict[int, list[list[str]]],
    bar_lengths: list[int],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, int, tuple[str, ...]]] = Counter()
    for slots_per_bar, sequences in sequences_by_slots.items():
        for labels in sequences:
            for bars in bar_lengths:
                length = slots_per_bar * bars
                if length <= 0 or len(labels) < length:
                    continue
                for index in range(0, len(labels) - length + 1):
                    window = tuple(labels[index : index + length])
                    if "N" in window:
                        continue
                    counts[(slots_per_bar, bars, window)] += 1
    rows = []
    ranked = counts.most_common(limit)
    for (slots_per_bar, bars, pattern), count in ranked:
        rows.append(
            {
                "slots_per_bar": slots_per_bar,
                "bars": bars,
                "pattern": list(pattern),
                "degree_pattern": "-".join(roman_degree_code(label) for label in pattern),
                "pop_code": "".join(roman_degree_code(label, drop_minor=True).replace("sus4", "s").replace("maj7", "M7") for label in pattern),
                "count": count,
            }
        )
    return rows


def build_prior(
    sequences: list[list[str]],
    bar_sequences_by_slots: dict[int, list[list[str]]],
    alpha: float,
    keep_stays: bool,
    keep_extensions: bool,
) -> dict[str, Any]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    trigram_counts: Counter[tuple[str, str, str]] = Counter()
    label_counts: Counter[str] = Counter()
    for labels in sequences:
        label_counts.update(labels)
        pair_counts.update(zip(labels, labels[1:]))
        trigram_counts.update(zip(labels, labels[1:], labels[2:]))
    return {
        "items": len(sequences),
        "alpha": alpha,
        "label_style": "roman_extended" if keep_extensions else "roman_function",
        "source": "user_corrections",
        "keep_stays": keep_stays,
        "keep_extensions": keep_extensions,
        "pair_scores": pair_scores(pair_counts, alpha=alpha),
        "labels": [
            {"label": label, "count": count}
            for label, count in label_counts.most_common()
        ],
        "pairs": [
            {"from": left, "to": right, "count": count}
            for (left, right), count in pair_counts.most_common()
        ],
        "trigrams": [
            {"a": a, "b": b, "c": c, "count": count}
            for (a, b, c), count in trigram_counts.most_common()
        ],
        "cycles": top_cycles(sequences, [4, 6, 8]),
        "bar_phrase_ngrams": phrase_pattern_rows(bar_sequences_by_slots, [1, 2, 4]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Analysis JSON files/directories or correction CSV files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "user_progression_prior.json",
    )
    parser.add_argument(
        "--regression-cases",
        type=Path,
        help="Use final analyses from benchmarks/user_regression_cases.json instead of scanning ad hoc outputs.",
    )
    parser.add_argument("--force-regression", action="store_true", help="Regenerate command-based regression analyses.")
    parser.add_argument("--edited-only", action="store_true", help="Use only spans marked edited=true in analysis JSON files.")
    parser.add_argument("--drop-stays", action="store_true", help="Compress repeated adjacent labels before counting.")
    parser.add_argument("--keep-extensions", action="store_true", help="Preserve extensions such as III7/V7/m7 instead of reducing to functional triads.")
    parser.add_argument("--alpha", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.alpha <= 0 or not math.isfinite(args.alpha):
        raise SystemExit("--alpha must be positive")
    paths = [resolve_path(path) for path in args.paths]
    if args.regression_cases:
        from run_user_regression import analysis_path_for_case, load_json

        config = load_json(resolve_path(args.regression_cases))
        for case in config.get("cases", []):
            paths.append(analysis_path_for_case(case, args.force_regression))
    if not paths:
        raise SystemExit("Provide at least one path or --regression-cases.")
    analysis_paths = iter_analysis_paths(paths)
    correction_csvs = [path for path in paths if path.exists() and path.suffix.lower() == ".csv"]
    sequences, bar_sequences_by_slots = collect_sequences(
        analysis_paths,
        correction_csvs,
        edited_only=args.edited_only,
        keep_stays=not args.drop_stays,
        keep_extensions=args.keep_extensions,
    )
    prior = build_prior(
        sequences,
        bar_sequences_by_slots,
        args.alpha,
        keep_stays=not args.drop_stays,
        keep_extensions=args.keep_extensions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"sequences: {len(sequences)}")
    print(
        "bar sequences: "
        + ", ".join(f"{slots}slot={len(values)}" for slots, values in sorted(bar_sequences_by_slots.items()))
    )
    for item in prior["pairs"][:12]:
        print(f"{item['from']} -> {item['to']}: {item['count']}")
    print("top bar phrases:")
    for item in prior.get("bar_phrase_ngrams", [])[:8]:
        print(f"{item['bars']}bar/{item['slots_per_bar']}slot {item['degree_pattern']}: {item['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
