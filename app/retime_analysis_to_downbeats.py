#!/usr/bin/env python
"""Attach a supplied beat grid without quantizing the original chord timeline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import median

GRID_FIELDS = ("bar", "beat_start", "beat_end", "position")


def read_downbeats(path: Path) -> list[tuple[float, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("beats", raw) if isinstance(raw, dict) else raw
    return [(float(row["time"]), float(row["beat"])) for row in rows]


def active_event_at(events: list[dict], time: float) -> dict | None:
    """Return only an actually covering event; gaps never extend a neighbour."""
    return next((event for event in events if float(event["start"]) <= time < float(event["end"])), None)


def chord_fields(item: dict | None) -> dict:
    if item is None:
        return {"chord": "X", "chord_display": "X", "roman": "?", "roman_display": "?",
                "bass_root": "N", "bass_confidence": 0.0, "confidence": 0.0,
                "annotation_status": "unlabeled"}
    chord = str(item.get("chord") or "X")
    roman = str(item.get("roman") or "?")
    return {"chord": chord, "chord_display": str(item.get("chord_display") or chord),
            "roman": roman, "roman_display": str(item.get("roman_display") or roman),
            "bass_root": str(item.get("bass_root") or "N"),
            "bass_confidence": float(item.get("bass_confidence") or 0),
            "confidence": float(item.get("confidence") or 0)}


def grid_regions(rows: list[tuple[float, int]], duration: float, meter_arg: str) -> tuple[list[dict], dict]:
    """Use observed adjacent markers only; no extrapolated head or final beat."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Analysis duration must be positive and finite")
    if len(rows) < 2:
        raise ValueError("At least two observed beat markers are required")
    if any(not math.isfinite(float(beat)) or not float(beat).is_integer() for _, beat in rows):
        raise ValueError("Beat numbers must be finite integers")
    ordered = sorted((float(time), int(beat)) for time, beat in rows)
    if any(not math.isfinite(time) or not 0 <= time <= duration or beat < 1 for time, beat in ordered):
        raise ValueError("Beat markers must be finite, inside the audio and have positive beat numbers")
    if any(right[0] <= left[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Beat marker timestamps must be distinct")
    if meter_arg.lower() == "auto":
        numerator = max(beat for _, beat in ordered)
        denominator = 8 if numerator == 6 else 4
        if numerator < 2:
            raise ValueError("Downbeat-only markers do not identify a meter; supply --meter")
    else:
        from chord_studio import parse_meter
        numerator, denominator = parse_meter(meter_arg)
    if numerator > 12 or any(beat > numerator for _, beat in ordered):
        raise ValueError("A beat number exceeds the requested meter")
    periods = [right[0] - left[0] for left, right in zip(ordered, ordered[1:])
               if right[1] == left[1] % numerator + 1]
    period = median(periods) if periods else None
    regions: list[dict] = []
    if ordered[0][0] > 0:
        regions.append({"bar": 0, "start": 0.0, "end": ordered[0][0], "beats": [],
                        "grid_status": "uncovered", "grid_reason": "before_first_observed_beat"})
    current_bar = 1
    previous_beat = None
    sequence_gaps = []
    for index, ((start, beat), (end, next_beat)) in enumerate(zip(ordered, ordered[1:])):
        if previous_beat is not None and beat <= previous_beat:
            current_bar += 1
        complete = next_beat == beat % numerator + 1
        status = "observed" if complete else "incomplete"
        if not complete:
            sequence_gaps.append({"start": start, "end": end, "beat": beat, "next_beat": next_beat})
        if not regions or regions[-1]["bar"] != current_bar:
            regions.append({"bar": current_bar, "start": start, "end": end, "beats": [],
                            "grid_status": status, "partial_start": beat != 1})
        region = regions[-1]
        region["end"] = end
        if not complete:
            region["grid_status"] = "incomplete"
        region["beats"].append({"beat": beat, "start": start, "end": end,
                                "grid_status": status, "start_observed": True,
                                "end_observed": True, "source_marker_index": index})
        previous_beat = beat
    if ordered[-1][0] < duration:
        regions.append({"bar": current_bar + 1, "start": ordered[-1][0], "end": duration,
                        "beats": [], "grid_status": "uncovered", "grid_reason": "after_last_observed_beat"})
    return regions, {"meter": f"{numerator}/{denominator}", "beats_per_bar": numerator,
                     "beat_period": period, "subdivision_bpm": 60 / period if period else None,
                     "sequence_gaps": sequence_gaps, "observed_markers": ordered}


def beat_position(region: dict, time: float) -> float | None:
    beats = region["beats"]
    if not beats:
        return None
    for beat in beats:
        if beat["start"] <= time < beat["end"] or (beat is beats[-1] and time == beat["end"]):
            if beat["grid_status"] == "incomplete":
                return float(beat["beat"]) if time == beat["start"] else None
            return round(beat["beat"] + (time - beat["start"]) / (beat["end"] - beat["start"]), 9)
    return None


def interval_spans(events: list[dict], start: float, end: float, region: dict) -> list[dict]:
    spans = []
    for index, event in enumerate(events):
        left, right = max(start, float(event["start"])), min(end, float(event["end"]))
        if right <= left:
            continue
        span = deepcopy(event)
        old_grid = {field: deepcopy(span[field]) for field in GRID_FIELDS if field in span}
        if old_grid:
            span["retime_original_grid"] = old_grid
        span.update({"bar": region["bar"], "start": left, "end": right,
                     "beat_start": beat_position(region, left), "beat_end": beat_position(region, right),
                     "grid_status": region["grid_status"], "source_event_index": index,
                     "source_event_start": event["start"], "source_event_end": event["end"]})
        if "position" in span:
            span["position"] = "unplaced" if span["beat_start"] is None else f"beat {span['beat_start']:g}"
        spans.append(span)
    return sorted(spans, key=lambda item: (item["start"], item["end"], item["source_event_index"]))


def retime_payload(payload: dict, downbeat_rows: list[tuple[float, int]], meter_arg: str,
                   *, beat_source: str = "external", meter_confidence: float | None = None) -> dict:
    duration = float(payload.get("duration") or 0)
    bars, grid = grid_regions(downbeat_rows, duration, meter_arg)
    return attach_observed_grid(payload, bars, grid, beat_source=beat_source,
                                meter_confidence=meter_confidence)


def attach_observed_grid(payload: dict, bars: list[dict], grid: dict, *,
                         beat_source: str = "external", meter_confidence: float | None = None) -> dict:
    """Attach validated adjacent-marker regions, preserving the source timeline."""
    duration = float(payload.get("duration") or 0)
    bars = deepcopy(bars)
    if meter_confidence is not None and (not math.isfinite(meter_confidence) or not 0 <= meter_confidence <= 1):
        raise ValueError("Supplied meter confidence must be in [0, 1]")
    source_events = deepcopy(payload.get("events", []))
    for index, event in enumerate(source_events):
        start, end = float(event["start"]), float(event["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 1e-6):
            raise ValueError(f"Invalid source event {index}: {start}--{end}")
    for bar in bars:
        bar["beat_source"] = beat_source if bar["beats"] else "uncovered"
        bar["spans"] = interval_spans(source_events, bar["start"], bar["end"], bar)
        for beat in bar["beats"]:
            spans = interval_spans(source_events, beat["start"], beat["end"], bar)
            dominant = max(spans, key=lambda span: span["end"] - span["start"], default=None)
            beat.update(chord_fields(dominant))
            beat.update({"spans": spans, "display_aggregation": "maximum_temporal_overlap",
                         "annotation_covered_seconds": sum(span["end"] - span["start"] for span in spans)})
    events = deepcopy(source_events)
    for index, event in enumerate(events):
        placements = [{key: span[key] for key in ("bar", "start", "end", "beat_start", "beat_end", "grid_status")}
                      for bar in bars for span in bar["spans"] if span["source_event_index"] == index]
        event["retime_original_grid"] = {field: deepcopy(event[field]) for field in GRID_FIELDS if field in event}
        event["grid_placement"] = placements
        if placements:
            event.update({key: placements[0][key] for key in ("bar", "beat_start", "beat_end", "grid_status")})
            if "position" in event:
                event["position"] = "unplaced" if placements[0]["beat_start"] is None else f"beat {placements[0]['beat_start']:g}"
    # Keep real overlaps visible for annotation repair; do not choose or trim labels.
    overlap_seconds, last_end = 0.0, 0.0
    for event in sorted(source_events, key=lambda item: float(item["start"])):
        start, end = float(event["start"]), float(event["end"])
        overlap_seconds += max(0.0, min(last_end, end) - start)
        last_end = max(last_end, end)
    output = deepcopy(payload)
    output.update({"bars": bars, "events": events, "meter": grid["meter"],
                   "beats_per_bar": grid["beats_per_bar"], "beat_period": grid["beat_period"],
                   "subdivision_bpm": grid["subdivision_bpm"],
                   "bpm": payload.get("bpm") or grid["subdivision_bpm"],
                   "meter_confidence": meter_confidence if meter_confidence is not None else 0.0,
                   "meter_confidence_status": "supplied" if meter_confidence is not None else "not_supplied",
                   "beat_grid": "external_observed_retimed", "beat_source": beat_source,
                   "tempo_variant": "external-grid-retime"})
    output["retime_provenance"] = {"beat_source": beat_source,
        "source_beat_source": payload.get("beat_source"), "source_beat_grid": payload.get("beat_grid"),
        "event_boundaries_preserved": True, "event_metadata_preserved": True,
        "quantized": False, "observed_markers": [deepcopy(row) if isinstance(row, dict) else {"time": row[0], "beat": row[1]}
                                                 for row in grid["observed_markers"]],
        "uncovered_regions": [{"start": bar["start"], "end": bar["end"], "reason": bar["grid_reason"]}
                              for bar in bars if bar["grid_status"] == "uncovered"],
        "beat_sequence_gaps": grid["sequence_gaps"], "source_event_overlap_seconds": overlap_seconds,
        "source_event_count": len(source_events)}
    # Preserve structure times and manual metadata, updating only its bar index.
    for collection in ("sections", "loops", "manual_overrides"):
        for item in output.get("structure", {}).get(collection, []):
            if not isinstance(item, dict) or "start" not in item or "end" not in item:
                continue
            touched = [bar for bar in bars if min(float(item["end"]), bar["end"]) > max(float(item["start"]), bar["start"])]
            if touched:
                item["retime_original_grid"] = {key: deepcopy(item[key]) for key in ("start_bar", "end_bar") if key in item}
                item.update({"start_bar": touched[0]["bar"], "end_bar": touched[-1]["bar"]})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("downbeats", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output base path without suffix.")
    parser.add_argument("--meter", default="auto")
    parser.add_argument("--beat-source", help="Explicit grid source; otherwise read metadata, falling back to external.")
    parser.add_argument("--meter-confidence", type=float, help="Use only a confidence actually provided by the source.")
    parser.add_argument("--no-html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    raw = json.loads(args.downbeats.read_text(encoding="utf-8"))
    metadata = raw if isinstance(raw, dict) else {}
    source = args.beat_source or metadata.get("beat_source") or metadata.get("source") or "external"
    confidence = args.meter_confidence if args.meter_confidence is not None else metadata.get("meter_confidence")
    output = retime_payload(payload, read_downbeats(args.downbeats), args.meter,
                            beat_source=str(source), meter_confidence=confidence)
    output["retime_provenance"].update({"source_analysis": str(args.analysis.resolve()),
                                       "beat_grid_file": str(args.downbeats.resolve())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_name(args.output.name + "_analysis.json")
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {json_path}")
    if not args.no_html:
        from chord_studio import write_html
        html_path = args.output.with_name(args.output.name + "_studio.html")
        write_html(html_path, output)
        print(f"Wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
