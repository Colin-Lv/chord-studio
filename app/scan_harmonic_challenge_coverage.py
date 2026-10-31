#!/usr/bin/env python
"""Scan existing references for harmonic challenge coverage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
ROOT_RE = re.compile(r"^([A-G](?:#|b)?)")
SECONDARY_DOMINANT_MAJOR = {
    2: "II7/V",
    4: "III7/VIm",
    9: "VI7/IIm",
    11: "VII7/IIIm",
}
FUNCTIONAL_DOM7_MAJOR = {
    0: "I7/IV",
    2: "II7/V",
    4: "III7/VIm",
    5: "IV7/bVII",
    7: "V7/I",
    9: "VI7/IIm",
    10: "bVII7/bIII",
    11: "VII7/IIIm",
}
BORROWED_MAJOR_ROOTS = {
    3: "bIII",
    8: "bVI",
    10: "bVII",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_root(label: str) -> tuple[int | None, str]:
    text = str(label or "N").strip()
    if text in {"", "N", "X"}:
        return None, ""
    text = text.split("/", 1)[0]
    text = text.split(":", 1)[0] if ":" in text else text
    match = ROOT_RE.match(text)
    if not match:
        return None, text
    root = match.group(1)
    return ROOT_TO_PC.get(root), text[len(root) :]


def key_root(key: str) -> tuple[int | None, bool]:
    pc, suffix = parse_root(key)
    return pc, str(key or "").strip().lower().endswith("m")


def chord_flags(label: str, key: str) -> dict[str, Any]:
    text = str(label or "N").strip()
    low = text.lower()
    root_pc, suffix = parse_root(text)
    key_pc, key_minor = key_root(key)
    relative = None if root_pc is None or key_pc is None else (root_pc - key_pc) % 12
    has_maj7 = "maj7" in low or "maj(7" in low or ":maj7" in low
    has_min7 = "min7" in low or ":min7" in low or "m7" in low
    has_7 = "7" in low
    dominant_7 = has_7 and not has_maj7 and not has_min7
    has_9 = "9" in low or "add9" in low or "maj(9" in low
    slash = "/" in text and not text.endswith("/1")
    borrowed = False
    borrowed_kind = ""
    secondary = ""
    functional_dom7 = ""
    if relative is not None and not key_minor:
        if relative in BORROWED_MAJOR_ROOTS:
            borrowed = True
            borrowed_kind = BORROWED_MAJOR_ROOTS[relative]
        if dominant_7:
            secondary = SECONDARY_DOMINANT_MAJOR.get(relative, "")
            functional_dom7 = FUNCTIONAL_DOM7_MAJOR.get(relative, "")
    return {
        "slash": slash,
        "seventh": has_7,
        "dominant7": dominant_7,
        "maj7": has_maj7,
        "min7": has_min7,
        "ninth": has_9,
        "sus": "sus" in low,
        "dim": "dim" in low or "o" in low,
        "aug": "aug" in low or "+" in low,
        "borrowed": borrowed,
        "borrowed_kind": borrowed_kind,
        "secondary": secondary,
        "functional_dom7": functional_dom7,
    }


def read_lab(path: Path) -> list[str]:
    labels = []
    if not path.exists():
        return labels
    with path.open(encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.append(parts[2])
    return labels


def iter_json_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans = []
    for bar in payload.get("bars", []):
        for span in bar.get("spans") or []:
            item = dict(span)
            item["bar"] = bar.get("bar", item.get("bar"))
            spans.append(item)
    return spans


def scan_labels(labels: list[str], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    unique = set()
    for label in labels:
        if str(label).strip() in {"", "N", "X"}:
            continue
        unique.add(label)
        counts["chord_events"] += 1
        flags = chord_flags(label, key)
        for name in ("slash", "seventh", "dominant7", "maj7", "min7", "ninth", "sus", "dim", "aug", "borrowed"):
            counts[name] += int(bool(flags[name]))
        if flags["secondary"]:
            counts["secondary_dominant"] += 1
            counts[f"secondary::{flags['secondary']}"] += 1
        if flags["functional_dom7"]:
            counts["functional_dom7"] += 1
            counts[f"functional::{flags['functional_dom7']}"] += 1
        if flags["borrowed_kind"]:
            counts[f"borrowed::{flags['borrowed_kind']}"] += 1
    counts["unique_chords"] = len(unique)
    return counts


def row_score(counts: Counter[str]) -> int:
    return (
        counts["secondary_dominant"] * 5
        + counts["functional_dom7"] * 4
        + counts["borrowed"] * 3
        + counts["slash"] * 2
        + counts["maj7"] * 2
        + counts["ninth"] * 2
        + counts["seventh"]
    )


def tags_for(counts: Counter[str]) -> str:
    tags = []
    if counts["secondary_dominant"]:
        tags.append("secondary_dominant")
    if counts["functional_dom7"]:
        tags.append("functional_dom7")
    if counts["borrowed"]:
        tags.append("borrowed")
    if counts["slash"]:
        tags.append("slash")
    if counts["maj7"] or counts["ninth"] or counts["sus"]:
        tags.append("extensions")
    if counts["dominant7"]:
        tags.append("dom7")
    return ",".join(tags)


def scan_manifest(path: Path, dataset_name: str) -> list[dict[str, Any]]:
    rows = []
    for item in read_csv(path):
        ref = item.get("chord_reference") or ""
        labels = read_lab(resolve_path(Path("benchmarks") / ref) if not ref.startswith("benchmarks") else resolve_path(ref))
        if not labels:
            continue
        counts = scan_labels(labels, item.get("key", ""))
        rows.append(
            {
                "source": dataset_name,
                "id": item.get("id", ""),
                "artist": item.get("artist", ""),
                "title": item.get("title", ""),
                "key": item.get("key", ""),
                "events": counts["chord_events"],
                "unique_chords": counts["unique_chords"],
                "secondary_dominant": counts["secondary_dominant"],
                "functional_dom7": counts["functional_dom7"],
                "borrowed": counts["borrowed"],
                "slash": counts["slash"],
                "seventh": counts["seventh"],
                "dominant7": counts["dominant7"],
                "maj7": counts["maj7"],
                "min7": counts["min7"],
                "ninth": counts["ninth"],
                "sus": counts["sus"],
                "score": row_score(counts),
                "tags": tags_for(counts),
            }
        )
    return rows


def scan_user_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    rows = []
    for case in payload.get("cases", []):
        analysis = case.get("analysis")
        if not analysis:
            continue
        analysis_path = resolve_path(analysis)
        if not analysis_path.exists():
            continue
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
        labels = []
        keys = set()
        modulation = 0
        for span in iter_json_spans(data):
            labels.append(str(span.get("chord_display") or span.get("chord") or "N"))
            if span.get("local_key"):
                keys.add(str(span.get("local_key")))
        sections = data.get("sections") or []
        if len({str(section.get("key", "")) for section in sections if section.get("key")}) > 1 or len(keys) > 1:
            modulation = 1
        counts = scan_labels(labels, data.get("key", ""))
        rows.append(
            {
                "source": "user_regression",
                "id": case.get("id", ""),
                "artist": case.get("artist", ""),
                "title": case.get("title", ""),
                "key": data.get("key", ""),
                "events": counts["chord_events"],
                "unique_chords": counts["unique_chords"],
                "secondary_dominant": counts["secondary_dominant"],
                "functional_dom7": counts["functional_dom7"],
                "borrowed": counts["borrowed"],
                "slash": counts["slash"],
                "seventh": counts["seventh"],
                "dominant7": counts["dominant7"],
                "maj7": counts["maj7"],
                "min7": counts["min7"],
                "ninth": counts["ninth"],
                "sus": counts["sus"],
                "modulation": modulation,
                "score": row_score(counts) + modulation * 20,
                "tags": ",".join(filter(None, [tags_for(counts), "modulation" if modulation else ""])),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--user-cases", type=Path, default=REPO_ROOT / "benchmarks" / "user_regression_cases.json")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for manifest in args.manifest or []:
        rows.extend(scan_manifest(manifest, manifest.stem.replace("manifest.", "")))
    rows.extend(scan_user_cases(args.user_cases))
    rows = sorted(rows, key=lambda item: (-int(item.get("score", 0)), item.get("source", ""), item.get("id", "")))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "id",
        "artist",
        "title",
        "key",
        "events",
        "unique_chords",
        "secondary_dominant",
        "functional_dom7",
        "borrowed",
        "slash",
        "seventh",
        "dominant7",
        "maj7",
        "min7",
        "ninth",
        "sus",
        "modulation",
        "score",
        "tags",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "output": str(output),
        "top": rows[:20],
    }
    if args.summary:
        resolve_path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
