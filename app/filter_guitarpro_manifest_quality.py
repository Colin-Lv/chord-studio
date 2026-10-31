#!/usr/bin/env python
"""Split imported Guitar Pro manifest rows by explicit chord-label quality."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def lab_stats(path: Path) -> dict[str, object]:
    labels: list[str] = []
    durations: list[float] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            labels.append(parts[2])
            durations.append(max(0.0, end - start))
    unique = sorted(set(labels))
    return {
        "chord_events": len(labels),
        "unique_chords": len(unique),
        "total_labeled": sum(durations),
        "max_duration": max(durations) if durations else 0.0,
        "unique_list": ", ".join(unique[:80]),
    }


def classify(stats: dict[str, object], min_events: int, min_unique: int, max_duration: float) -> tuple[str, str]:
    events = int(stats["chord_events"])
    unique = int(stats["unique_chords"])
    longest = float(stats["max_duration"])
    if events <= 0:
        return "needs_review", "no_explicit_chord_symbols"
    if unique <= 1:
        return "needs_review", "single_chord_or_placeholder"
    if events < min_events:
        return "needs_review", "too_few_chord_events"
    if unique < min_unique:
        return "needs_review", "too_few_unique_chords"
    if longest >= max_duration:
        return "needs_review", "overlong_chord_span"
    return "accepted", "ok"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_quality(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "id",
        "title",
        "artist",
        "key",
        "meter",
        "chord_events",
        "unique_chords",
        "total_labeled",
        "max_duration",
        "status",
        "reason",
        "unique_list",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--needs-review", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=8)
    parser.add_argument("--min-unique", type=int, default=3)
    parser.add_argument("--max-duration", type=float, default=80.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = resolve_path(args.manifest)
    rows = read_manifest(manifest)
    accepted: list[dict[str, str]] = []
    needs_review: list[dict[str, str]] = []
    quality_rows: list[dict[str, object]] = []
    for row in rows:
        ref = row.get("chord_reference", "")
        lab_path = resolve_path(Path("benchmarks") / ref) if ref else Path("__missing__")
        stats = lab_stats(lab_path)
        status, reason = classify(stats, args.min_events, args.min_unique, args.max_duration)
        (accepted if status == "accepted" else needs_review).append(row)
        quality_rows.append(
            {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "key": row.get("key", ""),
                "meter": row.get("meter", ""),
                "chord_events": stats["chord_events"],
                "unique_chords": stats["unique_chords"],
                "total_labeled": f"{float(stats['total_labeled']):.3f}",
                "max_duration": f"{float(stats['max_duration']):.3f}",
                "status": status,
                "reason": reason,
                "unique_list": stats["unique_list"],
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    write_manifest(resolve_path(args.accepted), accepted, fieldnames)
    write_manifest(resolve_path(args.needs_review), needs_review, fieldnames)
    write_quality(resolve_path(args.quality_report), quality_rows)
    print(
        {
            "rows": len(rows),
            "accepted": len(accepted),
            "needs_review": len(needs_review),
            "quality_report": str(resolve_path(args.quality_report)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
