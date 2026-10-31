#!/usr/bin/env python
"""Apply a fine-grained scale-degree chord chain to one bar."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

from chord_studio import chord_to_roman, chord_to_roman_display, estimate_mode_from_key, note_name_for_pc, parse_chord_name, write_html


MAJOR_DEGREES = {
    "1": (0, ""),
    "2": (2, "m"),
    "3": (4, "m"),
    "4": (5, ""),
    "5": (7, ""),
    "6": (9, "m"),
    "7": (11, "dim"),
}
MINOR_DEGREES = {
    "1": (0, "m"),
    "2": (2, "dim"),
    "3": (3, ""),
    "4": (5, "m"),
    "5": (7, ""),
    "6": (8, ""),
    "7": (10, ""),
}


def parse_chain(value: str) -> list[str]:
    return [token for token in re.split(r"[\s,\-–—]+", value.strip()) if token]


def degree_to_chord(token: str, key: str) -> str:
    key_root, key_suffix = parse_chord_name(key)
    if key_root is None:
        raise ValueError(f"Cannot parse key: {key}")
    match = re.fullmatch(r"([b#]?)([1-7])([A-Za-z0-9+#b°ø]*)", token)
    if not match:
        return token
    accidental, degree, suffix = match.groups()
    table = MINOR_DEGREES if key_suffix == "m" else MAJOR_DEGREES
    offset, default_quality = table[degree]
    if accidental == "b":
        offset -= 1
    elif accidental == "#":
        offset += 1
    quality = suffix
    if quality in {"", None}:
        quality = default_quality
    quality = quality.replace("°", "dim")
    if quality == "7" and default_quality == "m":
        quality = "m7"
    root = note_name_for_pc((key_root + offset) % 12, key)
    return f"{root}{quality}"


def allocation(count: int, subdivisions: int) -> list[int]:
    base = subdivisions // count
    remainder = subdivisions % count
    return [base + (1 if index < remainder else 0) for index in range(count)]


def relabel_span(span: dict, chord: str, key: str) -> dict:
    item = deepcopy(span)
    item["chord"] = chord
    item["chord_display"] = chord
    item["roman"] = chord_to_roman(chord, key)
    item["roman_display"] = chord_to_roman_display(chord, key)
    item["local_key"] = key
    item["bass_root"] = "N"
    item["bass_confidence"] = 0.0
    item["confidence"] = 1.0
    item["edited"] = True
    return item


def apply_chain(payload: dict, bar_number: int, key: str, chain: list[str], subdivisions: int) -> dict:
    output = deepcopy(payload)
    bars = output.get("bars", [])
    target = next((bar for bar in bars if int(bar.get("bar", 0)) == bar_number), None)
    if target is None:
        raise ValueError(f"Bar not found: {bar_number}")
    if not chain:
        raise ValueError("Chain is empty")
    chords = [degree_to_chord(token, key) for token in chain]
    counts = allocation(len(chords), subdivisions)
    bar_start = float(target["start"])
    bar_end = float(target["end"])
    step = (bar_end - bar_start) / subdivisions
    spans = []
    cursor = 0
    for chord, amount in zip(chords, counts):
        start = bar_start + cursor * step
        end = bar_start + (cursor + amount) * step
        beat_position = 1 + (cursor / subdivisions) * int(output.get("beats_per_bar") or 4)
        span = {
            "bar": bar_number,
            "unit": "sixteenth",
            "beat_start": round(beat_position, 2),
            "beat_end": round(1 + ((cursor + amount) / subdivisions) * int(output.get("beats_per_bar") or 4), 2),
            "start": start,
            "end": end,
            "position": f"{int(beat_position)}.{int((beat_position % 1) * 4) + 1}/16",
        }
        spans.append(relabel_span(span, chord, key))
        cursor += amount
    target["spans"] = spans
    for beat in target.get("beats", []):
        center = (float(beat["start"]) + float(beat["end"])) / 2.0
        span = next((item for item in spans if float(item["start"]) <= center < float(item["end"])), spans[-1])
        beat.update(relabel_span(beat, str(span["chord"]), key))
    output["events"] = sorted(
        [
            {**deepcopy(span), "bar": int(bar.get("bar", span.get("bar", 0)))}
            for bar in output.get("bars", [])
            for span in bar.get("spans", [])
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    if "modulations" not in output:
        output["modulations"] = [
            {
                "start_bar": 1,
                "end_bar": None,
                "key": key,
                "mode": estimate_mode_from_key(key),
                "progression": "",
                "unit": "bar",
                "phase": 0,
            }
        ]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--bar", type=int, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--chain", required=True, help="Scale-degree chain, e.g. 2-1-7-6-5-4.")
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    output = apply_chain(payload, args.bar, args.key, parse_chain(args.chain), args.subdivisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    if args.html:
        html_path = args.output.with_name(args.output.stem.replace("_analysis", "") + "_studio.html")
        write_html(html_path, output)
        print(f"Wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
