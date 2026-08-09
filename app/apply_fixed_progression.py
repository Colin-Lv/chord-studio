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
    parse_progression_hints,
    parse_chord_name,
    write_html,
)


def hint_to_chord_name(hint) -> tuple[str, str, float]:
    root = note_name_for_pc(hint.root_pc)
    suffix = hint.family
    if suffix == "":
        suffix = ""
    chord = f"{root}{suffix}"
    bass = note_name_for_pc(hint.bass_pc) if hint.bass_pc is not None else root
    return chord, bass, 1.0


def label_item(item: dict, chord: str, bass: str, confidence: float, key: str, edited: bool) -> dict:
    item = deepcopy(item)
    item["chord"] = chord
    item["chord_display"] = chord_display_name(chord, bass, confidence, confidence)
    item["roman"] = chord_to_roman(chord, key)
    item["roman_display"] = chord_to_roman_display(chord, key, bass, confidence, confidence)
    item["bass_root"] = bass
    item["bass_confidence"] = confidence
    item["confidence"] = confidence
    item["edited"] = edited
    return item


def fixed_progression_payload(
    payload: dict,
    progression: str,
    key: str,
    unit: str,
    start_bar: int = 1,
    end_bar: int | None = None,
    phase: int = 0,
) -> dict:
    hints = parse_progression_hints(progression, key)
    if not hints:
        raise ValueError(f"Cannot parse progression {progression!r} for key {key!r}")
    chord_cycle = [hint_to_chord_name(hint) for hint in hints]
    if unit not in {"half-bar", "bar"}:
        raise ValueError("Only --unit half-bar and --unit bar are currently supported.")

    payload = deepcopy(payload)
    payload["key"] = key
    key_root, key_suffix = parse_chord_name(key)
    payload["mode"] = "minor" if key_suffix == "m" else "major"
    payload["progression_override"] = {
        "progression": progression,
        "unit": unit,
        "start_bar": start_bar,
        "end_bar": end_bar,
        "phase": phase,
        "reason": "User-verified fixed progression correction",
    }

    events: list[dict] = []
    for bar_index, bar in enumerate(payload.get("bars", [])):
        bar_number = int(bar.get("bar", bar_index + 1))
        in_range = bar_number >= start_bar and (end_bar is None or bar_number <= end_bar)
        beats = sorted(bar.get("beats", []), key=lambda item: int(item.get("beat", 0)))
        if in_range:
            relative_bar = bar_number - start_bar
            for beat in beats:
                beat_number = int(beat.get("beat", 1))
                cycle_index = (
                    relative_bar + phase
                    if unit == "bar"
                    else relative_bar * 2 + (1 if beat_number > 2 else 0) + phase
                ) % len(chord_cycle)
                chord, bass, confidence = chord_cycle[cycle_index]
                updated = label_item(beat, chord, bass, confidence, key, edited=True)
                beat.clear()
                beat.update(updated)

        spans: list[dict] = []
        beat_groups = (
            [beats]
            if unit == "bar"
            else [
                [beat for beat in beats if int(beat.get("beat", 0)) <= 2],
                [beat for beat in beats if int(beat.get("beat", 0)) > 2],
            ]
        )
        if in_range:
            relative_bar = bar_number - start_bar
            for half_index, group in enumerate(beat_groups):
                if not group:
                    continue
                cycle_index = (
                    relative_bar + phase
                    if unit == "bar"
                    else relative_bar * 2 + half_index + phase
                ) % len(chord_cycle)
                chord, bass, confidence = chord_cycle[cycle_index]
                span = {
                    "bar": bar.get("bar"),
                    "unit": "bar" if unit == "bar" else "beat",
                    "beat_start": min(int(beat.get("beat", 0)) for beat in group),
                    "beat_end": max(int(beat.get("beat", 0)) for beat in group),
                    "start": min(float(beat.get("start", 0.0)) for beat in group),
                    "end": max(float(beat.get("end", 0.0)) for beat in group),
                }
                span = label_item(span, chord, bass, confidence, key, edited=True)
                spans.append(span)
                events.append(deepcopy(span))
            bar["spans"] = spans
        else:
            events.extend(deepcopy(span) for span in bar.get("spans", []))
    payload["events"] = events
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a user-verified fixed progression to a chord_studio analysis JSON.")
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--progression", default="6415")
    parser.add_argument("--key", default=None)
    parser.add_argument("--unit", default="half-bar", choices=["half-bar", "bar"])
    parser.add_argument("--start-bar", type=int, default=1)
    parser.add_argument("--end-bar", type=int, default=None)
    parser.add_argument("--phase", type=int, default=0, help="Rotate the progression by this many harmonic units.")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    key = args.key or payload.get("key") or "C"
    corrected = fixed_progression_payload(
        payload,
        args.progression,
        key,
        args.unit,
        start_bar=args.start_bar,
        end_bar=args.end_bar,
        phase=args.phase,
    )
    args.output.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    if args.html:
        html_path = args.output.with_name(args.output.stem.replace("_analysis", "") + "_studio.html")
        write_html(html_path, corrected)
        print(f"Wrote: {html_path}")


if __name__ == "__main__":
    main()
