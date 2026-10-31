#!/usr/bin/env python
"""Build a lightweight harmonic knowledge base from chord references."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from chord_studio import chord_to_roman, parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent
FUNCTIONAL_DOM7_RESOLUTION = {
    "I7": "IV",
    "II7": "V",
    "III7": "VIm",
    "IV7": "bVII",
    "V7": "I",
    "VI7": "IIm",
    "VII7": "IIIm",
    "bVII7": "bIII",
}
BORROWED_MAJOR = {"bIII", "IVm", "bVI", "bVII"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_lab(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            label = parts[2]
            if label in {"", "N", "X"}:
                continue
            rows.append({"start": start, "end": end, "chord": normalize_chord_label(label)})
    return rows


def normalize_chord_label(label: str) -> str:
    text = str(label or "N").strip()
    if ":" in text:
        root, rest = text.split(":", 1)
        rest = rest.replace("maj", "").replace("min", "m")
        text = root + rest
    text = text.replace("(9)", "9").replace("(", "").replace(")", "")
    text = text.replace("/3", "").replace("/5", "").replace("/b3", "")
    return text


def roman_root(roman: str) -> str:
    text = str(roman or "").split("/", 1)[0]
    if text == "N":
        return "N"
    degrees = ["bVII", "#VII", "VII", "bVI", "#VI", "VI", "#IV", "IV", "bIII", "#III", "III", "bII", "II", "V", "I"]
    for degree in degrees:
        if not text.startswith(degree):
            continue
        suffix = text[len(degree) :]
        return f"{degree}m" if suffix.startswith("m") else degree
    return text


def chord_family(label: str) -> str:
    _, suffix = parse_chord_name(str(label or "N").split("/", 1)[0])
    low = (suffix or "").lower()
    if "maj7" in low:
        return "maj7"
    if "m7" in low or (low.startswith("m") and "7" in low):
        return "m7"
    if "7" in low:
        return "7"
    if "9" in low or "add9" in low:
        return "9"
    if "sus" in low:
        return "sus"
    if low.startswith("m"):
        return "m"
    if "dim" in low:
        return "dim"
    return "maj"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def update_counter(counter: Counter[str], key: str) -> None:
    if key:
        counter[key] += 1


def scan_song(item: dict[str, str]) -> dict[str, Any]:
    ref = item.get("chord_reference", "")
    path = resolve_path(Path("benchmarks") / ref) if not ref.startswith("benchmarks") else resolve_path(Path(ref))
    key = item.get("key", "")
    chords = read_lab(path)
    roman_events = []
    for row in chords:
        roman = chord_to_roman(row["chord"], key)
        roman_events.append({**row, "roman": roman, "roman_root": roman_root(roman), "family": chord_family(row["chord"])})
    return {"manifest": item, "events": roman_events}


def build(manifests: list[Path]) -> dict[str, Any]:
    songs = []
    for manifest in manifests:
        for item in read_manifest(manifest):
            scanned = scan_song(item)
            if scanned["events"]:
                songs.append(scanned)

    functional_counts: Counter[str] = Counter()
    functional_resolution: dict[str, Counter[str]] = defaultdict(Counter)
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    borrowed_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    progression_ngrams: Counter[str] = Counter()

    for song in songs:
        events = song["events"]
        roots = [event["roman_root"] for event in events]
        for index, event in enumerate(events):
            roman = event["roman"]
            root = event["roman_root"]
            family = event["family"]
            update_counter(family_counts, family)
            if index + 1 < len(events):
                next_root = events[index + 1]["roman_root"]
                transition_counts[root][next_root] += 1
            if roman in FUNCTIONAL_DOM7_RESOLUTION:
                functional_counts[roman] += 1
                next_root = events[index + 1]["roman_root"] if index + 1 < len(events) else ""
                functional_resolution[roman][next_root] += 1
            if root in BORROWED_MAJOR or roman in BORROWED_MAJOR:
                borrowed_counts[roman] += 1
        for size in (2, 3, 4):
            for index in range(0, max(0, len(roots) - size + 1)):
                progression_ngrams[" ".join(roots[index : index + size])] += 1

    def counter_to_probs(counter: Counter[str]) -> list[dict[str, Any]]:
        total = sum(counter.values()) or 1
        return [
            {"item": item, "count": count, "prob": round(count / total, 6)}
            for item, count in counter.most_common()
        ]

    return {
        "songs": len(songs),
        "functional_dom7_counts": counter_to_probs(functional_counts),
        "functional_dom7_resolution": {
            roman: counter_to_probs(counter) for roman, counter in sorted(functional_resolution.items())
        },
        "borrowed_counts": counter_to_probs(borrowed_counts),
        "chord_family_counts": counter_to_probs(family_counts),
        "top_transitions": {root: counter_to_probs(counter)[:12] for root, counter in sorted(transition_counts.items())},
        "top_progression_ngrams": counter_to_probs(progression_ngrams)[:100],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.manifest)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"songs": payload["songs"], "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
