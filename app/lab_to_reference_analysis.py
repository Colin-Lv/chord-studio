#!/usr/bin/env python
"""Convert a start/end/chord .lab annotation into a Chord Studio reference analysis JSON."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from chord_studio import chord_to_roman, chord_to_roman_display, write_html
from evaluate_chords import convert_annotation_chord, read_reference


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def active_spans_for_bar(chords: list, bar: dict, key: str) -> list[dict]:
    start = float(bar.get("start", 0.0))
    end = float(bar.get("end", 0.0))
    spans = []
    for chord in chords:
        left = max(start, float(chord.start))
        right = min(end, float(chord.end))
        if right - left <= 0.025:
            continue
        label = convert_annotation_chord(str(chord.chord))
        span = {
            "bar": int(bar.get("bar", 0)),
            "unit": "reference",
            "beat_start": "",
            "beat_end": "",
            "start": left,
            "end": right,
            "chord": label,
            "chord_display": label,
            "roman": chord_to_roman(label, key),
            "roman_display": chord_to_roman_display(label, key),
            "local_key": key,
            "confidence": 1.0,
            "bass_root": "N",
            "bass_confidence": 0.0,
        }
        spans.append(span)
    return spans


def convert(prediction_payload: dict, chords: list, key: str) -> dict:
    output = deepcopy(prediction_payload)
    output["analysis_model"] = "reference-lab"
    output["reference_source"] = "lab"
    output["key"] = key
    output["mode"] = "minor" if key.endswith("m") else "major"
    events = []
    for bar in output.get("bars", []):
        spans = active_spans_for_bar(chords, bar, key)
        bar["spans"] = spans
        events.extend(deepcopy(span) for span in spans)
        for beat in bar.get("beats", []):
            center = (float(beat.get("start", 0.0)) + float(beat.get("end", 0.0))) / 2.0
            active = next((span for span in spans if float(span["start"]) <= center < float(span["end"])), None)
            if active is not None:
                for field in ("chord", "chord_display", "roman", "roman_display", "local_key", "confidence"):
                    beat[field] = active[field]
    output["events"] = sorted(events, key=lambda item: (float(item["start"]), float(item["end"])))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chord_lab", type=Path)
    parser.add_argument("prediction_analysis", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--html", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lab_path = resolve_path(args.chord_lab)
    prediction_path = resolve_path(args.prediction_analysis)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    chords = read_reference(lab_path)
    output = convert(payload, chords, args.key)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")
    if args.html:
        html_path = output_path.with_name(output_path.stem.replace("_analysis", "") + "_studio.html")
        write_html(html_path, output)
        print(f"Wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
