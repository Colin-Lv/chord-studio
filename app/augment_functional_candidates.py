#!/usr/bin/env python
"""Add functional harmony candidates such as III7 and secondary dominants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_candidate_training_set import chord_family
from chord_studio import chord_to_roman, chord_to_roman_display, note_name_for_pc, parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent
MAJOR_ROMAN_TO_PC = {
    "I": 0,
    "bII": 1,
    "II": 2,
    "bIII": 3,
    "III": 4,
    "IV": 5,
    "#IV": 6,
    "V": 7,
    "bVI": 8,
    "VI": 9,
    "bVII": 10,
    "VII": 11,
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("song_id", ""),
        row.get("bar", ""),
        row.get("reference_start") or row.get("start", ""),
        row.get("reference_end") or row.get("end", ""),
    )


def chord_root_name(key: str, roman: str, suffix: str) -> str | None:
    key_root, _ = parse_chord_name(key or "C")
    if key_root is None:
        return None
    offset = MAJOR_ROMAN_TO_PC.get(roman)
    if offset is None:
        return None
    pc = (key_root + offset) % 12
    return note_name_for_pc(pc, key) + suffix


def split_progression(value: str) -> list[str]:
    items = []
    for raw in value.split(","):
        text = raw.strip()
        if text:
            items.append(text)
    return items


def load_knowledge_romans(path: Path | None, min_count: int, min_prob: float) -> list[str]:
    if path is None:
        return []
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    output = []
    for item in payload.get("functional_dom7_counts", []) or []:
        roman = str(item.get("item", ""))
        count = int(item.get("count", 0) or 0)
        prob = float(item.get("prob", 0.0) or 0.0)
        if roman and count >= min_count and prob >= min_prob:
            output.append(roman)
    return output


def load_resolution_probs(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    output: dict[str, dict[str, float]] = {}
    for roman, rows in (payload.get("functional_dom7_resolution", {}) or {}).items():
        output[str(roman)] = {str(item.get("item", "")): float(item.get("prob", 0.0) or 0.0) for item in rows}
    return output


def roman_root(value: str) -> str:
    text = str(value or "")
    for suffix in ("maj7", "m7", "dim7", "7", "m", "dim", "aug", "sus"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def acoustic_rank1_roman_root(group: list[dict[str, str]]) -> str:
    topn = [
        row
        for row in group
        if str(row.get("candidate_source", "")).endswith("rank1")
        or str(row.get("candidate_rank", "")) == "1"
    ]
    row = topn[0] if topn else group[0]
    return roman_root(row.get("candidate_roman") or row.get("candidate_roman_root") or "")


def ordered_groups(grouped: dict[tuple[str, str, str, str], list[dict[str, str]]]) -> list[tuple[tuple[str, str, str, str], list[dict[str, str]]]]:
    def start_of(item: tuple[tuple[str, str, str, str], list[dict[str, str]]]) -> tuple[str, float]:
        group = item[1]
        row = group[0]
        try:
            start = float(row.get("reference_start") or row.get("start") or 0.0)
        except ValueError:
            start = 0.0
        return row.get("song_id", ""), start

    return sorted(grouped.items(), key=start_of)


def next_acoustic_lookup(grouped: dict[tuple[str, str, str, str], list[dict[str, str]]]) -> dict[tuple[str, str, str, str], str]:
    ordered = ordered_groups(grouped)
    output: dict[tuple[str, str, str, str], str] = {}
    for index, (key, group) in enumerate(ordered):
        row = group[0]
        next_root = ""
        if index + 1 < len(ordered):
            _, next_group = ordered[index + 1]
            if next_group[0].get("song_id", "") == row.get("song_id", ""):
                next_root = acoustic_rank1_roman_root(next_group)
        output[key] = next_root
    return output


def reference_fields(candidate: dict[str, str], reference_chord: str, reference_roman: str) -> None:
    candidate["target_exact_chord"] = "1" if candidate.get("candidate_chord") == reference_chord else "0"
    candidate["target_same_root"] = "1" if candidate.get("candidate_chord_root") and candidate.get("candidate_chord_root") == candidate.get("reference_chord_root") else "0"
    candidate["target_same_roman_root"] = "1" if candidate.get("candidate_roman_root") and candidate.get("candidate_roman_root") == candidate.get("reference_roman_root") else "0"
    candidate["target_same_chord_family"] = "1" if candidate.get("candidate_chord_family") == candidate.get("reference_chord_family") else "0"


def group_local_key(group: list[dict[str, str]]) -> str:
    for row in group:
        key = str(row.get("local_key") or "").strip()
        if key:
            return key
    return ""


def add_candidate(
    base: dict[str, str],
    source_name: str,
    chord: str,
    roman_hint: str,
    rank: int,
    score: float = 0.0,
    confidence: float = 0.35,
    local_key: str | None = None,
) -> dict[str, str]:
    row = dict(base)
    key = local_key if local_key is not None else row.get("local_key") or ""
    row["local_key"] = key
    root, suffix = parse_chord_name(chord)
    row["candidate_source"] = f"{source_name}_rank{rank}"
    row["candidate_chord"] = chord
    row["candidate_roman"] = chord_to_roman(chord, key)
    row["candidate_roman_root"] = roman_hint[:-1] if roman_hint.endswith("7") else roman_hint
    row["candidate_chord_root"] = "" if root is None else note_name_for_pc(root, key)
    row["candidate_chord_family"] = chord_family(chord)
    row["candidate_confidence"] = f"{confidence:.6f}"
    row["candidate_rank"] = str(rank)
    row["candidate_score"] = f"{score:.6f}"
    reference_fields(row, row.get("reference_chord", ""), row.get("reference_roman", ""))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--romans", default="III7,VI7,II7,V7,I7,IV7,bVII7")
    parser.add_argument("--knowledge-base", type=Path, help="Optional harmonic knowledge base JSON from build_harmonic_knowledge_base.py.")
    parser.add_argument("--kb-min-count", type=int, default=5)
    parser.add_argument("--kb-min-prob", type=float, default=0.0)
    parser.add_argument("--resolution-aware", action="store_true", help="Only add a KB candidate when its empirical resolution matches the next acoustic top-1 roman root.")
    parser.add_argument("--min-resolution-prob", type=float, default=0.05)
    parser.add_argument("--source-name", default="functional_dom7")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = resolve_path(args.candidate_csv)
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)
    added = []
    kb_romans = load_knowledge_romans(args.knowledge_base, args.kb_min_count, args.kb_min_prob)
    resolution_probs = load_resolution_probs(args.knowledge_base)
    roman_specs = list(dict.fromkeys(kb_romans + split_progression(args.romans)))
    next_lookup = next_acoustic_lookup(grouped)
    for key, group in grouped.items():
        base = group[0]
        local_key = group_local_key(group)
        if not local_key:
            continue
        existing = {row.get("candidate_chord") for row in group}
        rank = 1
        for spec in roman_specs:
            resolution_score = 0.0
            if args.resolution_aware:
                next_root = next_lookup.get(key, "")
                resolution_score = resolution_probs.get(spec, {}).get(next_root, 0.0)
                if resolution_score < args.min_resolution_prob:
                    continue
            roman = spec[:-1] if spec.endswith("7") else spec
            suffix = "7" if spec.endswith("7") else ""
            chord = chord_root_name(local_key, roman, suffix)
            if not chord or chord in existing:
                continue
            score = resolution_score if args.resolution_aware else 0.0
            confidence = 0.30 + min(0.35, resolution_score * 0.7)
            added.append(add_candidate(base, args.source_name, chord, spec, rank, score, confidence, local_key))
            existing.add(chord)
            rank += 1
    output_rows = rows + added
    for row in output_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for row in output_rows:
        for field in fieldnames:
            row.setdefault(field, "")
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} row(s), added {len(added)} functional candidate row(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
