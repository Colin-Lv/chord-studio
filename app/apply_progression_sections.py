from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from chord_studio import (
    chord_display_name,
    chord_to_roman,
    chord_to_roman_display,
    estimate_mode_from_key,
    note_name_for_pc,
    parse_progression_hints,
    write_html,
)


def parse_section(value: str) -> tuple[int, int | None, str, str, str, int]:
    parts = value.split(":")
    if len(parts) not in {4, 5, 6}:
        raise ValueError("section must be start:end:key:progression[:unit[:phase]]")
    start = int(parts[0])
    end = int(parts[1]) if parts[1] else None
    key = parts[2]
    progression = parts[3]
    unit = parts[4] if len(parts) >= 5 and parts[4] else "bar"
    phase = int(parts[5]) if len(parts) >= 6 and parts[5] else 0
    if unit not in {"bar", "half-bar"}:
        raise ValueError("unit must be bar or half-bar")
    return start, end, key, progression, unit, phase


def hint_to_chord_name(hint, key: str) -> tuple[str, str]:
    root = note_name_for_pc(hint.root_pc, key)
    chord = f"{root}{hint.family}"
    bass = note_name_for_pc(hint.bass_pc, key) if hint.bass_pc is not None else root
    return chord, bass


def label_item(item: dict, chord: str, bass: str, key: str) -> dict:
    item = deepcopy(item)
    item["chord"] = chord
    item["chord_display"] = chord_display_name(chord, bass, 1.0, 1.0)
    item["roman"] = chord_to_roman(chord, key)
    item["roman_display"] = chord_to_roman_display(chord, key, bass, 1.0, 1.0)
    item["local_key"] = key
    item["bass_root"] = bass
    item["bass_confidence"] = 1.0
    item["confidence"] = 1.0
    item["edited"] = True
    return item


def apply_section(payload: dict, start_bar: int, end_bar: int | None, key: str, progression: str, unit: str, phase: int) -> None:
    hints = parse_progression_hints(progression, key)
    if not hints:
        raise ValueError(f"Cannot parse progression {progression!r} for key {key!r}")
    cycle = [hint_to_chord_name(hint, key) for hint in hints]
    for bar in payload.get("bars", []):
        bar_number = int(bar.get("bar", 0))
        if bar_number < start_bar or (end_bar is not None and bar_number > end_bar):
            continue
        relative_bar = bar_number - start_bar
        beats = sorted(bar.get("beats", []), key=lambda item: int(item.get("beat", 0)))
        for beat in beats:
            beat_number = int(beat.get("beat", 1))
            cycle_index = (
                relative_bar + phase
                if unit == "bar"
                else relative_bar * 2 + (1 if beat_number > 2 else 0) + phase
            ) % len(cycle)
            chord, bass = cycle[cycle_index]
            updated = label_item(beat, chord, bass, key)
            beat.clear()
            beat.update(updated)

        groups = (
            [beats]
            if unit == "bar"
            else [
                [beat for beat in beats if int(beat.get("beat", 0)) <= 2],
                [beat for beat in beats if int(beat.get("beat", 0)) > 2],
            ]
        )
        spans = []
        for index, group in enumerate(groups):
            if not group:
                continue
            cycle_index = (relative_bar + phase if unit == "bar" else relative_bar * 2 + index + phase) % len(cycle)
            chord, bass = cycle[cycle_index]
            span = {
                "bar": bar_number,
                "unit": "bar" if unit == "bar" else "beat",
                "beat_start": min(int(beat.get("beat", 0)) for beat in group),
                "beat_end": max(int(beat.get("beat", 0)) for beat in group),
                "start": min(float(beat.get("start", 0.0)) for beat in group),
                "end": max(float(beat.get("end", 0.0)) for beat in group),
            }
            spans.append(label_item(span, chord, bass, key))
        bar["spans"] = spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply multiple local-key progression sections to a chord_studio analysis JSON.")
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--section",
        action="append",
        required=True,
        help="start:end:key:progression[:unit[:phase]], e.g. 3:74:C:6545:bar",
    )
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["modulations"] = []
    for spec in args.section:
        start, end, key, progression, unit, phase = parse_section(spec)
        apply_section(payload, start, end, key, progression, unit, phase)
        payload["modulations"].append(
            {
                "start_bar": start,
                "end_bar": end,
                "key": key,
                "mode": estimate_mode_from_key(key),
                "progression": progression,
                "unit": unit,
                "phase": phase,
            }
        )

    labels = [f"{item['key']} {item['mode']}" for item in payload["modulations"]]
    payload["key"] = " -> ".join(dict.fromkeys(labels))
    payload["mode"] = "modulating" if len(labels) > 1 else payload["modulations"][0]["mode"]
    payload["events"] = sorted(
        [
            {**deepcopy(span), "bar": int(bar.get("bar", span.get("bar", 0)))}
            for bar in payload.get("bars", [])
            for span in bar.get("spans", [])
        ],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    if args.html:
        html_path = args.output.with_name(args.output.stem.replace("_analysis", "") + "_studio.html")
        write_html(html_path, payload)
        print(f"Wrote: {html_path}")


if __name__ == "__main__":
    main()
