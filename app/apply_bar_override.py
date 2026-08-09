from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from chord_studio import (
    chord_display_name,
    chord_to_roman,
    chord_to_roman_display,
    note_name_for_pc,
    parse_chord_name,
    write_html,
)


def label_item(
    item: dict,
    chord: str,
    key: str,
    edited: bool,
    roman: str | None = None,
    roman_display: str | None = None,
    local_key: str | None = None,
) -> dict:
    root, _ = parse_chord_name(chord)
    bass = note_name_for_pc(root, key) if root is not None else "N"
    item = deepcopy(item)
    item["chord"] = chord
    item["chord_display"] = chord_display_name(chord, bass, 1.0, 1.0)
    item["roman"] = roman or chord_to_roman(chord, key)
    item["roman_display"] = roman_display or chord_to_roman_display(chord, key, bass, 1.0, 1.0)
    item["local_key"] = local_key or key
    item["bass_root"] = bass
    item["bass_confidence"] = 1.0
    item["confidence"] = 1.0
    item["edited"] = edited
    return item


def apply_bar_override(
    payload: dict,
    start_bar: int,
    end_bar: int,
    chord: str,
    key: str,
    roman: str | None = None,
    roman_display: str | None = None,
) -> dict:
    payload = deepcopy(payload)
    payload.setdefault("bar_overrides", []).append(
        {
            "start_bar": start_bar,
            "end_bar": end_bar,
            "chord": chord,
            "reason": "User-verified section correction",
        }
    )

    events: list[dict] = []
    for bar in payload.get("bars", []):
        bar_number = int(bar.get("bar", 0))
        if start_bar <= bar_number <= end_bar:
            beats = sorted(bar.get("beats", []), key=lambda item: int(item.get("beat", 0)))
            for beat in beats:
                updated = label_item(beat, chord, key, edited=True, roman=roman, roman_display=roman_display)
                beat.clear()
                beat.update(updated)
            if beats:
                span = {
                    "bar": bar_number,
                    "unit": "bar",
                    "beat_start": min(int(beat.get("beat", 0)) for beat in beats),
                    "beat_end": max(int(beat.get("beat", 0)) for beat in beats),
                    "start": min(float(beat.get("start", 0.0)) for beat in beats),
                    "end": max(float(beat.get("end", 0.0)) for beat in beats),
                }
                span = label_item(span, chord, key, edited=True, roman=roman, roman_display=roman_display)
                bar["spans"] = [span]
        events.extend(deepcopy(span) for span in bar.get("spans", []))
    payload["events"] = sorted(events, key=lambda item: (float(item.get("start", 0.0)), int(item.get("bar", 0))))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a chord override to a bar range in chord_studio analysis JSON.")
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--start-bar", type=int, required=True)
    parser.add_argument("--end-bar", type=int, required=True)
    parser.add_argument("--chord", required=True)
    parser.add_argument("--key", default=None)
    parser.add_argument("--roman", default=None)
    parser.add_argument("--roman-display", default=None)
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    key = args.key or payload.get("key") or "C"
    corrected = apply_bar_override(payload, args.start_bar, args.end_bar, args.chord, key, args.roman, args.roman_display)
    args.output.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    if args.html:
        html_path = args.output.with_name(args.output.stem.replace("_analysis", "") + "_studio.html")
        write_html(html_path, corrected)
        print(f"Wrote: {html_path}")


if __name__ == "__main__":
    main()
