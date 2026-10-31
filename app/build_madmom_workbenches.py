"""Build unhinted recommended-workflow examples from same-source cached LABs.

Only audio and pretrained expert predictions enter inference. User annotations
are opened after inference solely for the disclosed four-song comparison.
Existing All-In-One audio caches add structure without rerunning separation or
replacing the automatically inferred madmom beat grid.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from chord_studio import write_html, write_json
from evaluate_chords import evaluate, read_prediction
from evaluate_end_to_end import clean_json, file_signature, run_pipeline
from structure_expert import import_expert_structure, shift_expert_timebase
from validate_analysis import audit_analysis


CASES = {"gurenge_corrected": ("gurenge", "紅蓮華"), "qimeidi_corrected": ("qimeidi", "凄美地"),
         "pingfan_zhilu": ("pingfan", "平凡之路"),
         "yorushika_dakara_boku_bar97_chain": ("yorushika", "だから僕は音楽を辞めた")}


def attach_cached_structure(payload, cached_analysis):
    cached = json.loads(cached_analysis.read_text(encoding="utf-8"))
    if Path(payload["source"]).resolve() != Path(cached["original_source"]).resolve():
        raise ValueError("Cached structure belongs to another original audio")
    if abs(float(payload["duration"]) - float(cached["duration"])) > .001 or float(payload.get("audio_offset", 0)) != float(cached.get("original_offset_seconds", 0)):
        raise ValueError("Structure cache and new workflow used different audio windows")
    provenance = cached["structure"]["provenance"]
    result_path = Path(provenance["result"])
    external = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(result_path.with_suffix(".activ.npz"), allow_pickle=False) as data:
        activations = {key: data[key] for key in data.files}
    transform = provenance.get("timebase_transform")
    if transform:
        external = shift_expert_timebase(external, float(transform["offset_seconds"]), float(payload["duration"]))
    events, bars = deepcopy(payload["events"]), deepcopy(payload["bars"])
    payload["structure"] = import_expert_structure(payload, external, activations, "harmonix-fold0")
    payload["structure"]["provenance"].update(deepcopy(provenance))
    payload["structure"]["provenance"].update(cache_reused=True, mapped_to="automatic_madmom_grid",
        structure_cache_analysis=file_signature(cached_analysis), audio_offset=payload.get("audio_offset", 0),
        duration=payload["duration"], chord_or_beat_annotations_used_as_model_input=False)
    if payload["events"] != events or payload["bars"] != bars:
        raise ValueError("Attaching structure changed the original harmony or beat grid")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_report", type=Path)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--sheetsage-report", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    raw = json.loads(args.raw_report.read_text(encoding="utf-8"))
    sage = json.loads(args.sheetsage_report.read_text(encoding="utf-8"))
    sage_by_id = {item["id"]: item for item in sage["songs"]}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[name] = str(max(1, args.threads))
    files = [Path(__file__).with_name(name) for name in ("analyze_with_madmom.py", "downbeat_external_fusion.py", "chord_studio.py")]
    implementation = [file_signature(path) for path in files]
    report_path = output / "report.json"
    previous = {}
    if report_path.exists():
        previous = {item["id"]: item for item in json.loads(report_path.read_text(encoding="utf-8")).get("songs", [])}
    report = {"protocol": "audio_only_recommended_madmom_workflow_plus_cached_audio_only_structure",
        "reference_hints_used": False, "pretrained_training_overlap": "unknown", "threads": args.threads,
        "raw_report": file_signature(args.raw_report), "implementation_at_start": implementation,
        "comparison_reference_kind": "Existing user-edited examples; not independent external annotations or broad generalization evidence",
        "structure_does_not_change_chords_or_grid": True, "songs": []}
    for raw_record in raw["songs"]:
        case_id = raw_record["id"]
        name, title = CASES[case_id]
        base = output / name
        prediction = output / (name + "_analysis.json")
        started = time.monotonic()
        record = {"id": name, "case_id": case_id, "title": title, "prediction": str(prediction),
                  "html": str(output / (name + "_studio.html")), "reference_hints_used": False}
        try:
            audio = Path(raw_record["audio"])
            lab = args.raw_report.parent / case_id / "chord.lab"
            if file_signature(audio)["sha256"] != raw_record["audio_sha256"] or file_signature(lab)["sha256"] != raw_record["lab_sha256"]:
                raise ValueError("Audio or cached LAB differs from the frozen raw expert run")
            command = [sys.executable, str(Path(__file__).with_name("analyze_with_madmom.py").resolve()),
                str(audio), "-o", str(base), "--beat-source", "madmom", "--madmom-lab", str(lab.resolve()), "--structure", "none"]
            record.update(command=command, audio=file_signature(audio), cached_lab=file_signature(lab))
            if not prediction.exists():
                if args.score_only:
                    raise FileNotFoundError("Missing completed main workflow analysis")
                record["reused_prediction"] = False
                print("Running " + name, flush=True)
                log_path = output / (name + ".log")
                with log_path.open("w", encoding="utf-8") as log:
                    code = run_pipeline(command, environment, log, args.timeout)
                if code:
                    raise RuntimeError(f"Workflow failed with code {code}; see {log_path}")
            else:
                record["reused_prediction"] = True
            payload = json.loads(prediction.read_text(encoding="utf-8"))
            if Path(payload["source"]).resolve() != audio.resolve() or payload.get("beat_source") != "madmom" or payload.get("key_source") != "madmom_cnn_key":
                raise ValueError("Unexpected source or automatic model provenance")
            configuration = payload.get("fusion_provenance", {})
            if configuration.get("template_seventh_inheritance_enabled", True) or configuration.get("bass_root_weight", .075) != 0:
                raise ValueError("Workflow does not use the frozen no-template-seventh/no-bass-bonus defaults")
            payload = attach_cached_structure(payload, args.structure_dir / (name + "_structure_analysis.json"))
            payload["display_title"] = title + " · 和声工作台"
            payload["workflow_provenance"] = {"reference_hints_used": False, "main_default_configuration": configuration,
                "source_audio_sha256": raw_record["audio_sha256"], "raw_lab_sha256": raw_record["lab_sha256"],
                "structure_cache_did_not_change_events_or_grid": True}
            audit = audit_analysis(payload)
            if not audit["valid"]:
                raise ValueError("Main workflow failed interval integrity audit: " + str(audit["issues"]))
            write_json(prediction, payload)
            write_html(output / (name + "_studio.html"), payload)
            # Only now consult existing user examples for this separate comparison.
            reference = Path(raw_record["reference"])
            if file_signature(reference)["sha256"] != raw_record["reference_sha256"]:
                raise ValueError("Comparison annotations changed during inference")
            metrics = evaluate(read_prediction(reference), read_prediction(prediction))
            record.update(status="complete", metrics=metrics, raw_madmom_metrics=raw_record["metrics"],
                sheetsage_metrics=sage_by_id[case_id]["metrics"], audit=audit, inferred_key=payload["key"],
                inferred_meter=payload["meter"], duration=payload["duration"], event_count=len(payload["events"]),
                role_section_count=len(payload["structure"]["sections"]), configuration=configuration,
                reference_for_scoring_only=file_signature(reference), prediction_signature=file_signature(prediction),
                structure_chords_and_grid_unchanged=True)
            print(f"{name}: key={payload['key']}, meter={payload['meter']}, triad={metrics['triad_accuracy']:.6f}, tetrad={metrics['tetrad_accuracy']:.6f}", flush=True)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"FAILED {name}: {record['error']}", flush=True)
        record["elapsed_seconds"] = round(time.monotonic()-started, 3)
        if record.get("reused_prediction") and name in previous:
            record["first_run_elapsed_seconds"] = previous[name].get("first_run_elapsed_seconds", previous[name].get("elapsed_seconds"))
        report["songs"].append(record)
        successful = [item for item in report["songs"] if item["status"] == "complete"]
        report.update(completed_count=len(successful), failed_count=len(report["songs"])-len(successful), remaining_count=len(raw["songs"])-len(report["songs"]))
        if successful:
            report["macro_metrics"] = {source: {key: sum(item[source][key] for item in successful)/len(successful)
                for key in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "triad_wcsr", "prediction_coverage")}
                for source in ("metrics", "raw_madmom_metrics", "sheetsage_metrics")}
        report_path.write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
