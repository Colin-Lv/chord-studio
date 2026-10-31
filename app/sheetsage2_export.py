"""Export completed SheetSage2 predictions to the offline review workbench."""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

from evaluate_chords import convert_annotation_chord, read_reference, validate_timeline
from retime_analysis_to_downbeats import attach_observed_grid


def model_beat_regions(beats: list[dict], duration: float) -> tuple[list[dict], dict]:
    """Keep local meters and irregular beat IDs; infer no unobserved beat."""
    if len(beats) < 2:
        raise ValueError("At least two observed beat markers are required")
    rows = deepcopy(beats)
    for row in rows:
        time = float(row["time"])
        values = [float(row[key]) for key in ("beat", "meter_numerator", "meter_denominator")]
        if not math.isfinite(time) or not 0 <= time <= duration or any(not math.isfinite(v) or not v.is_integer() or v < 1 for v in values):
            raise ValueError("Invalid predicted beat or meter marker")
        beat, numerator, denominator = map(int, values)
        if beat > numerator:
            raise ValueError("Predicted beat number exceeds its local meter")
        row.update(time=time, beat=beat, meter_numerator=numerator, meter_denominator=denominator)
    if any(right["time"] <= left["time"] for left, right in zip(rows, rows[1:])):
        raise ValueError("Predicted beat timestamps must increase strictly")
    bars, sequence_gaps, periods, timeline = [], [], [], []
    meter_seconds = defaultdict(float)
    if rows[0]["time"] > 0:
        bars.append(dict(bar=0, start=0., end=rows[0]["time"], beats=[], grid_status="uncovered",
                         grid_reason="before_first_observed_beat"))
    number = 0
    previous = None
    for index, (row, following) in enumerate(zip(rows, rows[1:])):
        numerator, denominator = row["meter_numerator"], row["meter_denominator"]
        meter = f"{numerator}/{denominator}"
        start, end, beat = row["time"], following["time"], row["beat"]
        meter_changed = previous is not None and (numerator, denominator) != (previous["meter_numerator"], previous["meter_denominator"])
        if previous is None or meter_changed or beat < previous["beat"] or (numerator == 1 and beat == 1):
            number += 1
            bars.append(dict(bar=number, start=start, end=end, beats=[], meter=meter,
                             beats_per_bar=numerator, grid_status="observed", partial_start=beat != 1))
        same_next_meter = (numerator, denominator) == (following["meter_numerator"], following["meter_denominator"])
        complete = following["beat"] == beat % numerator + 1 and (same_next_meter or beat == numerator)
        status = "observed" if complete else "incomplete"
        if complete:
            periods.append(end - start)
        else:
            sequence_gaps.append(dict(start=start, end=end, beat=beat, next_beat=following["beat"],
                                      meter=meter, next_meter=f"{following['meter_numerator']}/{following['meter_denominator']}"))
        bar = bars[-1]
        bar["end"] = end
        if not complete:
            bar["grid_status"] = "incomplete"
        bar["beats"].append(dict(beat=beat, start=start, end=end, grid_status=status,
                                  start_observed=True, end_observed=True, source_marker_index=index))
        meter_seconds[meter] += end - start
        if timeline and timeline[-1]["meter"] == meter:
            timeline[-1]["end"] = end
        else:
            timeline.append(dict(start=start, end=end, meter=meter, beats_per_bar=numerator))
        previous = row
    if rows[-1]["time"] < duration:
        bars.append(dict(bar=number + 1, start=rows[-1]["time"], end=duration, beats=[],
                         grid_status="uncovered", grid_reason="after_last_observed_beat"))
    predominant = max(meter_seconds, key=meter_seconds.get)
    period = median(periods) if periods else None
    return bars, dict(meter=predominant, beats_per_bar=int(predominant.split("/")[0]),
                      beat_period=period, subdivision_bpm=60 / period if period else None,
                      sequence_gaps=sequence_gaps, observed_markers=rows, meter_timeline=timeline,
                      meter_mode="variable" if len(meter_seconds) > 1 else "constant")


def export_studio(directory: Path) -> Path:
    import chord_studio as studio
    directory = directory.resolve()
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "ok":
        raise ValueError("Only a completed transcription may be exported as a studio analysis")
    duration = float(run["clip_duration_seconds"])
    keys = []
    for line in (directory / "key.lab").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3:
            key = studio.normalize_key_label(fields[2])
            keys.append({"start": float(fields[0]), "end": float(fields[1]), "key": key,
                         "raw_key": fields[2], "source": "sheetsage2"})
    events = []
    for row in validate_timeline(read_reference(directory / "chord.lab"), "SheetSage2"):
        key = next((item["key"] for item in keys if item["start"] <= row.start < item["end"]), None)
        chord = convert_annotation_chord(row.chord)
        events.append({"start": row.start, "end": row.end, "chord": chord, "chord_display": chord,
                       "raw_chord": row.chord, "local_key": key, "unit": "external",
                       "roman": studio.chord_to_roman(chord, key) if key else "?",
                       "roman_display": studio.chord_to_roman_display(chord, key) if key else "?",
                       "confidence": 0.0, "confidence_status": "not_exported_by_model",
                       "source": "sheetsage2", "bass_root": chord.split("/", 1)[1] if "/" in chord else "N",
                       "bass_confidence": 0.0})
    beats = json.loads((directory / "beats.json").read_text(encoding="utf-8"))["beats"]
    key = keys[0]["key"] if keys else "Unknown"
    payload = {"source": str(directory / "audio.wav"), "audio_file": "audio.wav", "audio_offset": 0.,
               "original_source": run["source_audio"], "original_offset_seconds": run["original_offset_seconds"],
               "duration": duration, "key": key, "mode": "minor" if key.endswith("m") else "major",
               "key_timeline": {"segments": keys, "source": "sheetsage2"},
               "key_estimation": {"key": key, "source": "sheetsage2", "score_kind": "not_exported"},
               "analysis_model": "sheetsage2_experimental", "display": "name", "events": events,
               "structure": {"sections": [], "loops": [], "manual_overrides": []},
               "model_provenance": run, "beat_source": "sheetsage2", "export_warnings": []}
    try:
        bars, grid = model_beat_regions(beats, duration)
        payload = attach_observed_grid(payload, bars, grid, beat_source="sheetsage2")
        payload.update(meter_mode=grid["meter_mode"], meter_timeline=grid["meter_timeline"])
        if grid["meter_mode"] == "variable":
            payload["export_warnings"].append("Predicted local meter changes are preserved and require review; top-level meter is the duration-dominant value")
        if grid["sequence_gaps"]:
            payload["export_warnings"].append(f"{len(grid['sequence_gaps'])} irregular beat transitions retained without inventing missing markers")
    except ValueError as exc:
        payload["export_warnings"].append(str(exc))
    if "bars" not in payload:
        spans = [dict(deepcopy(event), bar=0, beat_start=None, beat_end=None, grid_status="uncovered") for event in events]
        payload.update(meter="unknown", beats_per_bar=0, bpm=0, meter_confidence=0,
                       bars=[dict(bar=0, start=0., end=duration, beats=[], spans=spans,
                                  grid_status="uncovered", beat_source="uncovered")])
    from validate_analysis import audit_analysis
    audit = audit_analysis(payload)
    (directory / "studio_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["valid"]:
        raise ValueError("Studio export has conflicting intervals; see studio_audit.json")
    path = directory / "sheetsage2_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    studio.write_html(directory / "sheetsage2_studio.html", payload)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(export_studio(args.directory))
