"""Replay final fusion from frozen automatic audio evidence, with honest provenance.

Use an evaluate_end_to_end report plus its manifest, or a four-song workbench
report. Reference labels are opened only after the prediction is complete.
This never claims to rerun the key, beat or chord networks.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import time

import downbeat_external_fusion as fusion
from audio_waveform import attach_waveform
from chord_studio import write_html, write_json
from evaluate_chords import evaluate, read_prediction
from evaluate_end_to_end import clean_json, file_signature, score_track, summarize, write_score_table
from song_structure import analyze_song_structure, detect_harmonic_loops, valid_bars
from validate_analysis import audit_analysis


def non_n_events(payload):
    return [(float(row["start"]), float(row["end"]), row.get("chord"))
            for row in payload["events"] if row.get("chord") != "N"]


def beat_coordinates(payload):
    return [(bar["bar"], beat["beat"], beat["start"], beat["end"])
            for bar in payload["bars"] for beat in bar.get("beats", [])]


def refresh_structure(payload, previous):
    existing = previous.get("structure", {})
    if existing.get("method") == "bar_multifeature_novelty":
        payload["structure"] = copy.deepcopy(existing)
        payload["structure"] = analyze_song_structure(payload)
        return
    payload["structure"] = copy.deepcopy(existing)
    bars = valid_bars(payload)
    for name in ("sections", "automatic_sections", "manual_overrides", "loops"):
        for section in payload["structure"].get(name, []):
            if not isinstance(section, dict) or "start" not in section or "end" not in section:
                continue
            touched = [bar for bar in bars if min(float(section["end"]), bar["end"]) > max(float(section["start"]), bar["start"])]
            if touched:
                section.update(start_bar=touched[0]["bar"], end_bar=touched[-1]["bar"])
    payload["structure"]["loops"] = detect_harmonic_loops(bars)


def validate_frozen_cnn_evidence(previous, base):
    for payload in (previous, base):
        if payload.get("key_source") != "madmom_cnn_key" or payload.get("beat_source") != "madmom":
            raise ValueError("This final replay requires automatic madmom CNN key and madmom beat evidence")
        if payload.get("fusion_provenance", {}).get("key_source") == "provided":
            raise ValueError("An explicitly provided key cannot be described as automatic CNN evidence")
    if previous.get("key") != base.get("key") or Path(previous["source"]).resolve() != Path(base["source"]).resolve():
        raise ValueError("Frozen base and final prediction disagree on source/key")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Required for an end-to-end report; omitted for user-workbench reports")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("reference_hints_used") is not False:
        parser.error("Only explicitly unhinted automatic reports may be replayed")
    records = baseline.get("tracks", baseline.get("songs", []))
    is_benchmark = "tracks" in baseline
    if is_benchmark != bool(args.manifest):
        parser.error("An end-to-end report requires --manifest; a user report supplies its own scoring-only references")
    manifest = {row["id"]: row for row in csv.DictReader(args.manifest.open(encoding="utf-8-sig"))} if args.manifest else {}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == args.baseline.resolve().parent:
        parser.error("Replay output must be separate from the frozen baseline")
    completed = []
    implementation = [file_signature(Path(__file__).resolve()), file_signature(Path(fusion.__file__).resolve())]
    target_report = output / ("evaluation.json" if is_benchmark else "report.json")
    for old_record in records:
        if old_record.get("status") != "complete":
            raise ValueError("Cannot silently omit incomplete original tracks")
        began = time.perf_counter()
        previous_path = Path(old_record["prediction"])
        if file_signature(previous_path)["sha256"] != old_record.get("prediction_signature", {}).get("sha256"):
            raise ValueError("Frozen prediction content changed or lacks a recorded signature")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        base = previous_path.with_name(previous_path.name.removesuffix("_analysis.json") + "_base_grid_analysis.json")
        validate_frozen_cnn_evidence(previous, json.loads(base.read_text(encoding="utf-8")))
        lab = Path(old_record["cached_lab"]["path"])
        if file_signature(lab)["sha256"] != old_record["cached_lab"]["sha256"]:
            raise ValueError("Frozen external LAB content changed")
        target = output / previous_path.name.removesuffix("_analysis.json")
        prediction = target.with_name(target.name + "_analysis.json")
        command = ["fusion", str(base), str(base), str(lab), "-o", str(target),
                   "--beat-grid-from-base", "--meter", "auto", "--harmonic-unit", "auto", "--no-html",
                   "--dominant-seventh-threshold", "1.01", "--bass-root-weight", "0", "--edge-fill",
                   "--edge-fill-mode", "segments", "--edge-fill-min-duration", "0.001", "--edge-fill-min-score", "0",
                   "--edge-fill-include-no-chord", "--no-infer-slash-bass"]
        previous_argv = sys.argv
        try:
            sys.argv = command
            with contextlib.redirect_stdout(io.StringIO()):
                status = fusion.main()
        finally:
            sys.argv = previous_argv
        if status:
            raise RuntimeError("Fusion replay failed")
        payload = json.loads(prediction.read_text(encoding="utf-8"))
        if non_n_events(payload) != non_n_events(previous):
            raise ValueError("Replay unexpectedly changed non-N chord labels or boundaries")
        if beat_coordinates(payload) != beat_coordinates(previous):
            raise ValueError("Replay unexpectedly changed observed beat coordinates")
        for key in ("display_title", "external_chord_cache", "key_source", "key_estimation", "key_candidates", "key_scope"):
            if key in previous:
                payload[key] = copy.deepcopy(previous[key])
        payload["fusion_replay_provenance"] = {"previous_prediction": file_signature(previous_path),
            "base_evidence": file_signature(base), "external_lab": file_signature(lab),
            "command": command, "audio_networks_reused": True, "reference_hints_used": False,
            "non_no_chord_events_unchanged": True, "beat_coordinates_unchanged": True,
            "changes": ["preserve_explicit_source_N_at_edges", "suppress_unvalidated_inferred_slash_display"]}
        refresh_structure(payload, previous)
        attach_waveform(payload)
        audit = audit_analysis(payload)
        if not audit["valid"]:
            raise ValueError(f"Replay failed interval integrity: {audit}")
        write_json(prediction, payload)
        html = target.with_name(target.name + "_studio.html")
        write_html(html, payload)
        record = copy.deepcopy(old_record)
        record.update(prediction=str(prediction), html=str(html), prediction_signature=file_signature(prediction),
                      replay_command=command, original_inference_command=old_record.get("command"),
                      audit=audit, event_count=len(payload["events"]), status="complete",
                      replay_elapsed_seconds=time.perf_counter()-began, reused_audio_evidence=True)
        if is_benchmark:
            record.update(score_track(manifest[record["id"]], prediction, args.manifest.parent, external_lab=lab))
        else:
            reference = Path(record["reference_for_scoring_only"]["path"])
            if file_signature(reference)["sha256"] != record["reference_for_scoring_only"]["sha256"]:
                raise ValueError("Scoring reference changed")
            record["metrics"] = evaluate(read_prediction(reference), read_prediction(prediction))
            record["configuration"] = payload["fusion_provenance"]
        completed.append(record)
        if is_benchmark:
            report = summarize(completed, len(records), args.manifest, "madmom", 1.01, 0., True, True, False)
            write_score_table(completed, output / "per_song.csv")
        else:
            report = {key: copy.deepcopy(value) for key, value in baseline.items() if key not in {"songs", "macro_metrics"}}
            report.update(songs=completed, completed_count=len(completed), remaining_count=len(records)-len(completed),
                          macro_metrics={metric: sum(item["metrics"][metric] for item in completed)/len(completed)
                          for metric in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "prediction_coverage")})
        report.update(protocol="frozen audio-only CNN key/beat/base/LAB evidence + final fusion replay",
                      replay_only=True, baseline_report=file_signature(args.baseline), infer_slash_bass=False,
                      replay_implementation=implementation,
                      change_note="Only explicit source N is restored at edges, and unvalidated display inversions are suppressed. Non-N labels/times and beat coordinates are verified identical.")
        target_report.write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"{len(completed)}/{len(records)} {record['id']}: complete", flush=True)
    print(f"Report: {target_report}")


if __name__ == "__main__":
    main()
