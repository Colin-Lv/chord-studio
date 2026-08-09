#!/usr/bin/env python
"""Retiming helper: keep chord labels, replace beat/bar grid with downbeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chord_studio import (
    build_beat_intervals,
    meter_from_downbeat_rows,
    parse_meter,
    write_html,
)


def read_downbeats(path: Path) -> list[tuple[float, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("beats", raw) if isinstance(raw, dict) else raw
    return [(float(row["time"]), int(row["beat"])) for row in rows]


def active_event_at(events: list[dict[str, object]], time: float) -> dict[str, object] | None:
    if not events:
        return None
    lo = 0
    hi = len(events) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        item = events[mid]
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if time < start:
            hi = mid - 1
        elif time >= end:
            lo = mid + 1
        else:
            return item
    return events[max(0, min(len(events) - 1, hi))]


def chord_fields(item: dict[str, object] | None) -> dict[str, object]:
    if not item:
        return {
            "chord": "N",
            "chord_display": "N",
            "roman": "N",
            "roman_display": "N",
            "bass_root": "N",
            "bass_confidence": 0.0,
            "confidence": 0.0,
        }
    chord = str(item.get("chord") or "N")
    chord_display = str(item.get("chord_display") or chord)
    roman = str(item.get("roman") or "N")
    roman_display = str(item.get("roman_display") or roman)
    return {
        "chord": chord,
        "chord_display": chord_display,
        "roman": roman,
        "roman_display": roman_display,
        "bass_root": str(item.get("bass_root") or "N"),
        "bass_confidence": float(item.get("bass_confidence") or 0.0),
        "confidence": float(item.get("confidence") or 0.0),
    }


def same_chord(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        left.get("chord_display") == right.get("chord_display")
        and left.get("roman_display") == right.get("roman_display")
        and left.get("bass_root") == right.get("bass_root")
    )


def span_from_beats(bar: dict[str, object], group: list[dict[str, object]]) -> dict[str, object]:
    first = group[0]
    last = group[-1]
    beat_start = int(first["beat"])
    beat_end = int(last["beat"])
    beats = bar["beats"]
    unit = "bar" if beat_start == int(beats[0]["beat"]) and beat_end == int(beats[-1]["beat"]) else "beat"
    confidence = sum(float(beat.get("confidence") or 0.0) for beat in group) / max(1, len(group))
    return {
        "bar": int(bar["bar"]),
        "unit": unit,
        "beat_start": beat_start,
        "beat_end": beat_end,
        "start": float(first["start"]),
        "end": float(last["end"]),
        "chord": first["chord"],
        "chord_display": first["chord_display"],
        "roman": first["roman"],
        "roman_display": first["roman_display"],
        "bass_root": first["bass_root"],
        "bass_confidence": first["bass_confidence"],
        "confidence": confidence,
    }


def spans_for_bar(bar: dict[str, object]) -> list[dict[str, object]]:
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for beat in bar["beats"]:
        if not current or same_chord(current[-1], beat):
            current.append(beat)
        else:
            groups.append(current)
            current = [beat]
    if current:
        groups.append(current)
    return [span_from_beats(bar, group) for group in groups]


def retime_payload(
    payload: dict[str, object],
    downbeat_rows: list[tuple[float, int]],
    meter_arg: str,
) -> dict[str, object]:
    duration = float(payload.get("duration") or 0.0)
    requested = parse_meter(meter_arg)
    meter = meter_from_downbeat_rows(downbeat_rows, duration, requested)
    if meter is None:
        raise ValueError("Downbeat file did not contain a usable beat grid")

    source_events = sorted(
        [dict(event) for event in payload.get("events", [])],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    intervals = [item for item in build_beat_intervals(meter, duration) if item.bar >= 1]
    bars: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    for bar_number in sorted({item.bar for item in intervals}):
        bar_intervals = [item for item in intervals if item.bar == bar_number]
        beats = []
        for interval in bar_intervals:
            center = (interval.start + interval.end) / 2.0
            fields = chord_fields(active_event_at(source_events, center))
            beats.append(
                {
                    "beat": interval.beat,
                    "start": interval.start,
                    "end": interval.end,
                    **fields,
                }
            )
        bar = {
            "bar": bar_number,
            "start": beats[0]["start"],
            "end": beats[-1]["end"],
            "beats": beats,
            "spans": [],
        }
        bar["spans"] = spans_for_bar(bar)
        bars.append(bar)
        events.extend(bar["spans"])

    output = json.loads(json.dumps(payload, ensure_ascii=False))
    output.update(
        {
            "bpm": float(payload.get("bpm") or meter.bpm),
            "subdivision_bpm": meter.bpm,
            "meter": meter.meter_label,
            "meter_confidence": meter.confidence,
            "beat_period": meter.beat_period,
            "beats_per_bar": meter.beats_per_bar,
            "beat_grid": "madmom_downbeat_retimed",
            "beat_source": "madmom",
            "tempo_variant": "madmom-downbeat-retime",
            "bars": bars,
            "events": events,
        }
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retiming an existing analysis JSON to a downbeat grid.")
    parser.add_argument("analysis", type=Path)
    parser.add_argument("downbeats", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output base path without suffix.")
    parser.add_argument("--meter", default="6/8")
    parser.add_argument("--no-html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    downbeat_rows = read_downbeats(args.downbeats)
    output = retime_payload(payload, downbeat_rows, args.meter)
    json_path = args.output.with_name(args.output.name + "_analysis.json")
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {json_path}")
    if not args.no_html:
        html_path = args.output.with_name(args.output.name + "_studio.html")
        write_html(html_path, output)
        print(f"Wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
