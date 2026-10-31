"""Controlled acoustic diagnostic using reference key and beat grid.

This intentionally measures the acoustic/decoder layer, not end-to-end beat or
key estimation. Predictions and a reproducible per-song report are kept apart
from the dataset's existing prediction paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from evaluate_chords import TimedChord, evaluate, read_reference


def score(reference, events, studio=None):
    """Compatibility helper routed through the same standard full-time scorer."""
    rows = [event if isinstance(event, TimedChord) else TimedChord(*event) for event in reference]
    prediction = [TimedChord(event.start, event.end, event.chord) for event in events]
    metrics = evaluate(rows, prediction)
    return dict(seconds=metrics["non_no_chord_seconds"], root_accuracy=metrics["root_accuracy"],
                triad_accuracy=metrics["triad_accuracy"], metric_version="mir_eval_full_reference_v2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--feature-mode", default="fast", choices=["fast", "cqt"])
    parser.add_argument("--analysis-model", default="functional", choices=["functional", "template", "acoustic"])
    parser.add_argument("--acoustic-model", type=Path)
    parser.add_argument("--mode", default="triads", choices=["triads", "seventh", "rich"])
    parser.add_argument("--harmonic-unit", default="half-bar", choices=["bar", "half-bar", "beat"])
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    import chord_studio as studio
    artifact = None
    if args.analysis_model == "acoustic":
        if args.acoustic_model is None:
            parser.error("--acoustic-model is required for learned acoustic inference.")
        import joblib
        artifact = joblib.load(args.acoustic_model)
        if artifact["feature_mode"] != args.feature_mode or artifact["harmonic_unit"] != args.harmonic_unit:
            parser.error("Feature mode and harmonic unit must match the trained model.")

    manifest = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    if args.limit:
        rows = rows[:args.limit]
    report = {"protocol": "controlled_reference_key_and_beats_mir_eval_v2",
              "warning": "Not end-to-end accuracy. *_accuracy excludes N/X; *_wcsr includes N. Missing predictions are N under the standard protocol; coverage is reported separately.",
              "config": {key: str(value) for key, value in vars(args).items()}, "songs": []}
    for row in rows:
        if artifact is not None and row["id"] in artifact["training_song_ids"]:
            raise ValueError("Refusing to report held-out accuracy on a model's training song.")
        started = time.perf_counter()
        paths = {field: (manifest.parent / row[field]).resolve()
                 for field in ("audio", "beat_reference", "chord_reference")}
        if artifact is not None and str(paths["audio"]).casefold() in artifact.get("training_audio_paths", []):
            raise ValueError("Refusing to score a training audio path under a different song ID.")
        y = studio.decode_audio(paths["audio"])
        duration = len(y) / studio.SAMPLE_RATE
        meter = studio.meter_from_downbeat_rows(studio.read_beat_grid_rows(paths["beat_reference"]), duration, None)
        if meter is None:
            raise ValueError(f"Missing reference beat grid: {row['id']}")
        intervals = studio.build_beat_intervals(meter, duration)
        chroma = studio.extract_chroma(y, studio.SAMPLE_RATE, 2048, mode=args.feature_mode)
        bass = studio.extract_bass_chroma(y, studio.SAMPLE_RATE, 2048, mode=args.feature_mode)
        key = studio.normalize_key_label(row["key"])
        if args.analysis_model == "functional":
            spans = studio.recognize_functional_spans(y, chroma, bass, intervals, key, "sharps", 0.2, 0.025, 0.18,
                                                      args.harmonic_unit, meter.beats_per_bar)
        elif args.analysis_model == "acoustic":
            from acoustic_chord_model import recognize_acoustic
            spans, _ = recognize_acoustic(y, chroma, bass, intervals, artifact=artifact, key=key, notation="sharps",
                                           harmonic_unit=args.harmonic_unit, beats_per_bar=meter.beats_per_bar,
                                           transition_strength=artifact["transition_strength"])
        else:
            cells = studio.harmonic_intervals_from_beats(intervals, args.harmonic_unit, meter.beats_per_bar)
            vectors = studio.chroma_for_intervals(chroma, y, studio.SAMPLE_RATE, 2048, cells)
            rms = studio.rms_for_intervals(y, studio.SAMPLE_RATE, cells)
            bv, br, bc = studio.bass_for_intervals(bass, studio.SAMPLE_RATE, 2048, cells, "sharps")
            matched = studio.match_interval_chords(vectors, rms, args.mode, "sharps", key, 0.025, 0.18,
                                                   bv, br, bc, bass_weight=0.15, transition_strength=0.2)
            spans = [studio.ChordSpan("beat", cell.bar, cell.beat_start, cell.beat_end, cell.start, cell.end,
                                      matched.chords[i], float(matched.confidence[i]), br[i], float(bc[i]))
                     for i, cell in enumerate(cells)]
        reference = read_reference(paths["chord_reference"])
        metrics = evaluate(reference, [TimedChord(span.start, span.end, span.chord) for span in spans], .1)
        record = {"id": row["id"], **metrics, "elapsed": time.perf_counter() - started}
        report["songs"].append(record)
        payload = studio.make_payload(paths["audio"], duration, 0, key, studio.estimate_mode_from_key(key), meter,
                                      studio.beat_chords_from_spans(intervals, spans), spans, "roman",
                                      args.analysis_model, "reference", args.harmonic_unit, True, True, [])
        payload["evaluation_context"] = {"reference_key": True, "reference_beat_grid": True}
        payload["beat_source"] = "reference_grid"
        payload["feature_mode"] = args.feature_mode
        if artifact is not None:
            payload["acoustic_model"] = {"path": str(args.acoustic_model.resolve()),
                                         "feature_schema": artifact["feature_schema"],
                                         "transition_strength": artifact["transition_strength"]}
        studio.write_json(output / f"{row['id']}_analysis.json", payload)
        print(json.dumps(record), flush=True)
        for metric in ("root_accuracy", "triad_accuracy", "root_wcsr", "triad_wcsr"):
            report[f"macro_{metric}"] = float(np.mean([item[metric] for item in report["songs"]]))
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
