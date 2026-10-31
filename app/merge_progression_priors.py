#!/usr/bin/env python
"""Merge multiple Roman-numeral progression priors with configurable weights."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from progression_stats import pair_scores


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_source(value: str) -> tuple[Path, float]:
    if "::" in value:
        path_text, weight_text = value.rsplit("::", 1)
        weight = float(weight_text)
    else:
        path_text = value
        weight = 1.0
    if weight < 0 or not math.isfinite(weight):
        raise ValueError("source weight must be a finite non-negative number")
    return resolve_path(path_text), weight


def add_pairs(counter: Counter[tuple[str, str]], prior: dict[str, Any], weight: float) -> None:
    for item in prior.get("pairs", []):
        left = str(item.get("from") or "").strip()
        right = str(item.get("to") or "").strip()
        if not left or not right or left == "N" or right == "N":
            continue
        count = float(item.get("count") or 0.0)
        if count > 0:
            counter[(left, right)] += count * weight


def add_ngrams(counters: dict[int, Counter[tuple[str, ...]]], prior: dict[str, Any], weight: float) -> None:
    for size_text, rows in (prior.get("ngrams", {}) or {}).items():
        try:
            size = int(size_text)
        except (TypeError, ValueError):
            continue
        counter = counters.setdefault(size, Counter())
        for item in rows or []:
            pattern = tuple(str(value).strip() for value in item.get("pattern", []) if str(value).strip())
            if len(pattern) != size or "N" in pattern:
                continue
            count = float(item.get("count") or 0.0)
            if count > 0:
                counter[pattern] += count * weight


def merge(sources: list[tuple[Path, float]], alpha: float) -> dict[str, Any]:
    pair_counter: Counter[tuple[str, str]] = Counter()
    ngram_counters: dict[int, Counter[tuple[str, ...]]] = {}
    source_summaries = []
    for path, weight in sources:
        prior = json.loads(path.read_text(encoding="utf-8"))
        add_pairs(pair_counter, prior, weight)
        add_ngrams(ngram_counters, prior, weight)
        source_summaries.append(
            {
                "path": str(path),
                "weight": weight,
                "items": prior.get("items"),
                "source": prior.get("source", path.stem),
            }
        )
    return {
        "items": sum(float(item.get("items") or 0.0) * item["weight"] for item in source_summaries),
        "alpha": alpha,
        "label_style": "roman_function",
        "source": "weighted_merge",
        "sources": source_summaries,
        "pair_scores": pair_scores(pair_counter, alpha=alpha),
        "pairs": [
            {"from": left, "to": right, "count": count}
            for (left, right), count in pair_counter.most_common()
        ],
        "ngrams": {
            str(size): [
                {"pattern": list(pattern), "count": count}
                for pattern, count in counter.most_common(200)
            ]
            for size, counter in sorted(ngram_counters.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Prior JSON path, optionally suffixed with ::weight, e.g. prior.json::0.35",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "merged_progression_prior.json",
    )
    parser.add_argument("--alpha", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.alpha <= 0 or not math.isfinite(args.alpha):
        raise SystemExit("--alpha must be positive")
    sources = [parse_source(item) for item in args.source]
    for path, _ in sources:
        if not path.exists():
            raise SystemExit(f"Missing prior: {path}")
    prior = merge(sources, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    for item in prior["pairs"][:12]:
        print(f"{item['from']} -> {item['to']}: {item['count']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
