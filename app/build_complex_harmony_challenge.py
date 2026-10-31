#!/usr/bin/env python
"""Build a phrase-level complex harmony challenge set from existing references."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chord_studio import chord_to_roman, parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_RE = re.compile(r"^([A-G](?:#|b)?)")
FUNCTIONAL_DOM7 = {"I7", "II7", "III7", "IV7", "V7", "VI7", "VII7", "bVII7"}
SECONDARY_DOM7 = {"I7", "II7", "III7", "VI7", "VII7"}
BORROWED_ROOTS = {"bIII", "bVI", "bVII", "IVm", "Vm", "bII", "bVII7", "IVm7"}
EXTENSION_MARKERS = ("maj7", "m7", "min7", "dim7", "7", "9", "add9", "sus", "6", "dim", "aug")
COMMON_PHRASES = {
    ("VIm", "IV", "I", "V"): "6415",
    ("I", "V", "VIm", "IV"): "1564",
    ("I", "V", "VIm", "IIIm", "IV", "I", "IV", "V"): "15634145",
    ("IV", "V", "IIIm", "VIm", "IIm", "V", "I"): "4536251",
    ("IV", "V", "IIIm", "VIm", "IIm7", "V9sus4", "I"): "4536251_color",
    ("IIm", "V", "I"): "251",
    ("IIm7", "V7", "I"): "2517",
    ("I", "IVm", "I"): "borrowed_iv",
}


@dataclass
class Event:
    source: str
    song_id: str
    artist: str
    title: str
    key: str
    index: int
    bar: int
    start: float
    end: float
    chord: str
    roman: str
    local_key: str
    tags: set[str]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalize_chord(label: object) -> str:
    text = str(label or "N").strip()
    if not text or text in {"X", "N"}:
        return "N"
    if ":" in text:
        root, suffix = text.split(":", 1)
        suffix = suffix.replace("maj", "").replace("min", "m")
        text = root + suffix
    text = text.replace(":maj", "").replace(":min", "m")
    text = text.replace("majmaj7", "maj7")
    text = text.replace("(9)", "9").replace("(", "").replace(")", "")
    text = text.replace(":7", "7")
    return text


def roman_root(value: object) -> str:
    text = str(value or "").strip()
    if "/" in text and text.startswith("V"):
        return text
    for suffix in ("maj7", "m7", "dim7", "7", "m", "dim", "aug", "sus2", "sus4", "sus"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def read_beats(path: Path | None) -> list[tuple[float, int]]:
    if path is None or not path.exists():
        return []
    beats = []
    with path.open(encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                beats.append((float(parts[0]), int(float(parts[1]))))
            except ValueError:
                continue
    return beats


def bar_for_time(beats: list[tuple[float, int]], start: float, fallback: int) -> int:
    if not beats:
        return fallback
    bar = 0
    for time, beat in beats:
        if time > start + 1e-6:
            break
        if beat == 1:
            bar += 1
    return max(1, bar)


def read_lab_events(item: dict[str, str], source_name: str) -> list[Event]:
    ref = item.get("chord_reference") or ""
    if not ref:
        return []
    ref_path = resolve_path(Path("benchmarks") / ref if not ref.startswith("benchmarks") else ref)
    beat_ref = item.get("beat_reference") or ""
    beat_path = resolve_path(Path("benchmarks") / beat_ref if beat_ref and not beat_ref.startswith("benchmarks") else beat_ref) if beat_ref else None
    beats = read_beats(beat_path)
    events = []
    if not ref_path.exists():
        return events
    with ref_path.open(encoding="utf-8-sig", errors="ignore") as handle:
        for index, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            chord = normalize_chord(parts[2])
            if chord == "N":
                continue
            key = item.get("key") or ""
            roman = chord_to_roman(chord, key) if key else chord
            bar = bar_for_time(beats, start, index)
            events.append(
                Event(
                    source=source_name,
                    song_id=item.get("id", ""),
                    artist=item.get("artist", ""),
                    title=item.get("title", ""),
                    key=key,
                    index=index,
                    bar=bar,
                    start=start,
                    end=end,
                    chord=chord,
                    roman=roman,
                    local_key=key,
                    tags=set(),
                )
            )
    return events


def iter_user_analysis_cases(path: Path) -> list[dict[str, str]]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("cases", []):
        analysis = item.get("analysis")
        if not analysis:
            continue
        rows.append(
            {
                "id": item.get("id", ""),
                "artist": item.get("artist", ""),
                "title": item.get("title", ""),
                "analysis": analysis,
            }
        )
    return rows


def read_json_events(item: dict[str, str], source_name: str) -> list[Event]:
    path = resolve_path(item["analysis"])
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = str(payload.get("key") or payload.get("detected_key") or "")
    events = []
    index = 0
    previous_key = ""
    for bar in payload.get("bars", []):
        spans = bar.get("spans") or []
        bar_number = int(bar.get("bar") or bar.get("number") or len(events) + 1)
        for span_index, span in enumerate(spans, start=1):
            chord = normalize_chord(span.get("chord") or span.get("label") or span.get("display") or "N")
            roman = str(span.get("roman_display") or span.get("roman") or "")
            local_key = str(span.get("local_key") or bar.get("local_key") or key)
            if not roman and chord != "N":
                roman = chord_to_roman(chord, local_key or key)
            if chord == "N" and roman in {"", "N"}:
                continue
            index += 1
            start = float(span.get("start", bar.get("start", 0.0)) or 0.0)
            end = float(span.get("end", bar.get("end", start)) or start)
            tags: set[str] = set()
            if len(spans) > 1 or span_index > 1:
                tags.add("offbeat_or_within_bar")
            if previous_key and local_key and local_key != previous_key:
                tags.add("modulation")
            previous_key = local_key or previous_key
            events.append(
                Event(
                    source=source_name,
                    song_id=item.get("id", path.stem),
                    artist=item.get("artist", ""),
                    title=item.get("title", payload.get("title", "")),
                    key=key,
                    index=index,
                    bar=bar_number,
                    start=start,
                    end=end,
                    chord=chord,
                    roman=roman or "N",
                    local_key=local_key,
                    tags=tags,
                )
            )
    return events


def classify_event(event: Event) -> None:
    chord = event.chord
    roman = event.roman
    root = roman_root(roman)
    low = chord.lower()
    if "/" in chord:
        event.tags.add("slash")
    if any(marker in low for marker in EXTENSION_MARKERS):
        event.tags.add("extension")
    dom7_token = roman if roman in FUNCTIONAL_DOM7 else root + "7"
    if roman.endswith("7") and dom7_token in FUNCTIONAL_DOM7:
        event.tags.add("functional_dom7")
    if roman.endswith("7") and dom7_token in SECONDARY_DOM7:
        event.tags.add("secondary_dominant")
    if "/" in roman and roman.startswith("V"):
        event.tags.add("secondary_dominant")
        event.tags.add("functional_dom7")
    if root in BORROWED_ROOTS or roman in BORROWED_ROOTS or "IVm" in roman:
        event.tags.add("borrowed")
    if re.search(r"b(?:II|III|VI|VII)", roman):
        event.tags.add("borrowed")
    parsed_root, suffix = parse_chord_name(chord.split("/", 1)[0])
    if parsed_root is not None and suffix.lower().startswith("m") and root == "IV":
        event.tags.add("borrowed")


def phrase_tags(pattern: tuple[str, ...]) -> set[str]:
    roots = tuple(roman_root(item) for item in pattern if item and item != "N")
    tags = set()
    for known, name in COMMON_PHRASES.items():
        if len(roots) >= len(known):
            for offset in range(0, len(roots) - len(known) + 1):
                if roots[offset : offset + len(known)] == known:
                    tags.add(f"phrase:{name}")
    if any(root in FUNCTIONAL_DOM7 for root in roots):
        tags.add("phrase:functional_dom7")
    if any(root in BORROWED_ROOTS for root in roots):
        tags.add("phrase:borrowed")
    return tags


def mine_phrases(events: list[Event], radius: int) -> list[dict[str, Any]]:
    rows = []
    for index, event in enumerate(events):
        if not event.tags:
            continue
        start = max(0, index - radius)
        end = min(len(events), index + radius + 1)
        window = events[start:end]
        pattern = tuple(item.roman for item in window)
        tags = set(event.tags) | phrase_tags(pattern)
        rows.append(
            {
                "source": event.source,
                "song_id": event.song_id,
                "artist": event.artist,
                "title": event.title,
                "key": event.key,
                "local_key": event.local_key,
                "focus_index": event.index,
                "focus_bar": event.bar,
                "focus_start": round(event.start, 3),
                "focus_end": round(event.end, 3),
                "focus_chord": event.chord,
                "focus_roman": event.roman,
                "tags": sorted(tags),
                "window_start_bar": window[0].bar,
                "window_end_bar": window[-1].bar,
                "chord_pattern": [item.chord for item in window],
                "roman_pattern": [item.roman for item in window],
            }
        )
    return rows


def event_row(event: Event) -> dict[str, Any]:
    return {
        "source": event.source,
        "song_id": event.song_id,
        "artist": event.artist,
        "title": event.title,
        "key": event.key,
        "local_key": event.local_key,
        "index": event.index,
        "bar": event.bar,
        "start": f"{event.start:.3f}",
        "end": f"{event.end:.3f}",
        "chord": event.chord,
        "roman": event.roman,
        "tags": "|".join(sorted(event.tags)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(events: list[Event], phrases: list[dict[str, Any]]) -> str:
    by_tag: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    for event in events:
        if event.tags:
            by_source[event.source] += 1
            by_tag.update(event.tags)
    phrase_tag_counts: Counter[str] = Counter()
    for phrase in phrases:
        phrase_tag_counts.update(phrase["tags"])
    lines = [
        "# Complex Harmony Challenge Set",
        "",
        "This benchmark is phrase-level: one song can contribute many focused hard examples.",
        "",
        f"- tagged events: `{sum(by_source.values())}`",
        f"- phrase windows: `{len(phrases)}`",
        "",
        "## Event Tags",
        "",
    ]
    for tag, count in by_tag.most_common():
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## Sources", ""])
    for source, count in by_source.most_common():
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Phrase Tags", ""])
    for tag, count in phrase_tag_counts.most_common():
        lines.append(f"- `{tag}`: {count}")
    lines.extend(
        [
            "",
            "## Recommended Use",
            "",
            "- Use this set to evaluate candidate coverage before reranking.",
            "- Train generic priors from large sources, but keep user-corrected cases as high-confidence regression tests.",
            "- Treat POP909 and Guitar Pro symbolic cases as medium-confidence unless matched to the exact audio version.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--user-regression", type=Path, default=REPO_ROOT / "benchmarks" / "user_regression_cases.json")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "benchmarks" / "complex_harmony_challenge")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources: dict[str, list[Event]] = defaultdict(list)
    for manifest in args.manifest:
        source_name = resolve_path(manifest).stem.replace("manifest.", "")
        for item in read_csv(manifest):
            sources[source_name].extend(read_lab_events(item, source_name))
    if args.user_regression and resolve_path(args.user_regression).exists():
        for item in iter_user_analysis_cases(args.user_regression):
            sources["user_regression"].extend(read_json_events(item, "user_regression"))

    all_events: list[Event] = []
    all_phrases: list[dict[str, Any]] = []
    for events in sources.values():
        for event in events:
            classify_event(event)
        all_events.extend(events)
        grouped: dict[str, list[Event]] = defaultdict(list)
        for event in events:
            grouped[event.song_id].append(event)
        for song_events in grouped.values():
            all_phrases.extend(mine_phrases(song_events, args.radius))

    tagged_events = [event_row(event) for event in all_events if event.tags]
    output_dir = resolve_path(args.output_dir)
    write_csv(output_dir / "events.csv", tagged_events)
    (output_dir / "phrases.json").write_text(json.dumps(all_phrases, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(build_summary(all_events, all_phrases), encoding="utf-8")
    print(
        json.dumps(
            {
                "events": len(tagged_events),
                "phrases": len(all_phrases),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
