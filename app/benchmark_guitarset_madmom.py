"""Frozen GuitarSet diagnostic: raw audio-only CNN+CRF, two label protocols."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from benchmark_madmom_cache_reproducibility import model_metadata
from evaluate_chords import evaluate, harte_chord, read_reference, validate_timeline
from madmom_bridge import DEFAULT_MADMOM_PYTHON, file_identity, provenance_path
from sheetsage2_bridge import REPO_ROOT, run_worker_monitored

SELECTION_FIELDS = ("id", "audio", "chord_reference", "instructed_reference",
                    "annotation_metadata", "reference_kind", "performer", "composition_id",
                    "progression_family", "style", "split")
LABEL_PROTOCOL = {
    "performed": "Sheet-informed symbolic chord transcription; separate-string notes with segmentation/root from sheet music; semi-automatic with manual verification, not independent manual chord truth.",
    "instructed": "Intended lead-sheet chords, not a claim that every intended pitch was sounded.",
}


def selection_rows(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [{key: row[key] for key in SELECTION_FIELDS} for row in csv.DictReader(handle)]
    if len(rows) != 10 or len({row["id"] for row in rows}) != 10:
        raise ValueError("This frozen diagnostic requires exactly 10 unique excerpts")
    if Counter(row["style"] for row in rows) != Counter({style: 2 for style in ("BN", "Funk", "Jazz", "Rock", "SS")}):
        raise ValueError("Require two excerpts in each of the five predeclared styles")
    if any(not row["id"].endswith("_comp") or row["performer"] not in {"00", "05"}
           or row["split"] != "diagnostic_no_fitting" for row in rows):
        raise ValueError("Only the predeclared 00/05 comp diagnostic is permitted")
    return rows


def checked_reference(path: Path):
    rows = validate_timeline(read_reference(path), "Reference")
    if path.suffix.lower() in {".lab", ".txt"}:
        source_rows = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
        if len(rows) != len(source_rows):
            raise ValueError(f"Reference rows cannot be silently skipped: {path}")
    if not rows:
        raise ValueError(f"Empty reference: {path}")
    unsupported = [row.chord for row in rows if row.chord.upper() not in {"N", "X"} and harte_chord(row.chord) == "X"]
    if unsupported:
        raise ValueError(f"Unsupported labels must not silently disappear: {unsupported}")
    return rows


def prepare_protocol(manifest: Path, data_root: Path) -> dict:
    rows = selection_rows(manifest)
    frozen = []
    for row in rows:
        metadata = json.loads((data_root / row["annotation_metadata"]).read_text(encoding="utf-8"))
        references = {name: data_root / row[column] for name, column in
                      (("performed", "chord_reference"), ("instructed", "instructed_reference"))}
        for path in references.values():
            checked_reference(path)
        source_jams = file_identity(Path(metadata["source_jams"]))
        if source_jams["sha256"] != metadata["source_jams_sha256"]:
            raise ValueError("Imported labels no longer match their original JAMS identity")
        frozen.append(dict(selection=row, audio_path=str((data_root / row["audio"]).resolve()),
                           references={name: file_identity(path) for name, path in references.items()},
                           source_jams=source_jams,
                           annotation_metadata_snapshot=metadata))
    return dict(protocol="predeclared_10_comp_5_styles_2_performers_no_fitting_no_selection_by_scores",
                frozen_at=datetime.now().astimezone().isoformat(),
                manifest_at_selection=file_identity(manifest), data_root=str(data_root.resolve()),
                selection_rule="styles BN/Funk/Jazz/Rock/SS x performers 00/05; family=(style_index+performer_index)%3+1; first sorted lead; known alignment issues excluded before any predictions",
                label_protocol=LABEL_PROTOCOL, references_are_independent_manual_truth=False,
                inference_inputs="Original complete microphone waveform only; no labels/key/beats/prompt", songs=frozen)


def summarize(songs: list[dict]) -> dict:
    good = [row for row in songs if row.get("status") == "ok"]
    result = dict(completed=len(good), failed=len(songs) - len(good))
    for label in LABEL_PROTOCOL:
        result[label] = {}
        for metric in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy",
                       "root_wcsr", "triad_wcsr", "tetrad_wcsr", "tetrad_inv_wcsr", "prediction_coverage"):
            if not good:
                continue
            values = [row["metrics"][label] for row in good]
            weight = "non_no_chord_seconds" if metric.endswith("accuracy") else "evaluated_seconds"
            denominator = sum(value[weight] for value in values)
            result[label]["macro_" + metric] = sum(value[metric] for value in values) / len(values)
            result[label]["duration_weighted_" + metric] = sum(value[metric] * value[weight] for value in values) / denominator if denominator else 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "benchmarks")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-starting-at", default="2026-09-26T19:47:00+08:00")
    parser.add_argument("--finish-at", default="2026-09-26T19:49:00+08:00")
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    args = parser.parse_args()
    start_limit, finish_limit = datetime.fromisoformat(args.stop_starting_at), datetime.fromisoformat(args.finish_at)
    if start_limit.tzinfo is None or finish_limit.tzinfo is None or finish_limit <= start_limit:
        parser.error("Require timezone-aware ordered start/finish deadlines")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(prepare_protocol(args.manifest, args.data_root), ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if selection_rows(args.manifest) != [row["selection"] for row in protocol["songs"]]:
        raise ValueError("Manifest selection changed after freezing; refusing to replace excerpts")
    if args.prepare_only:
        print(f"Frozen 10 excerpts before inference: {protocol_path}")
        return 0
    if (args.output / "report.json").exists():
        raise FileExistsError("Fresh diagnostic needs a new report; existing results are preserved")
    # All audio identities are frozen before the first model invocation.
    for item in protocol["songs"]:
        item["audio"] = file_identity(Path(item["audio_path"]))
        for identity in [*item["references"].values(), item["source_jams"]]:
            if file_identity(Path(identity["path"]))["sha256"] != identity["sha256"]:
                raise ValueError("Frozen reference or original JAMS changed")
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    metadata = model_metadata(args.madmom_python, environment)
    report = dict(protocol=protocol, protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  model=metadata, started_at=datetime.now().astimezone().isoformat(),
                  sample_rate=44100, threads=2, offset=0, duration=None, status="running", songs=[])
    def save():
        report["summary"] = summarize(report["songs"])
        pending = args.output / "report.pending.json"
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(args.output / "report.json")
    save()
    for item in protocol["songs"]:
        now = datetime.now(start_limit.tzinfo)
        if now >= start_limit:
            report["status"] = "stopped_at_deadline"
            save()
            return 2
        directory = args.output / item["selection"]["id"]
        directory.mkdir(exist_ok=True)
        output = directory / "chord.lab"
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, str(REPO_ROOT / "app/madmom_bridge.py"), item["audio"]["path"],
                   "-o", str(output.resolve()), "--sample-rate", "44100", "--madmom-python", str(args.madmom_python)]
        print(f"Fresh {len(report['songs']) + 1}/10 {item['selection']['id']}", flush=True)
        remaining = max(1, (finish_limit - now).total_seconds())
        code = run_worker_monitored(command, environment, directory,
                                    active_timeout=min(90, remaining), wall_timeout=min(120, remaining))
        record = dict(id=item["selection"]["id"], style=item["selection"]["style"],
                      command=command, return_code=code, status="failed" if code else "ok",
                      resources=json.loads((directory / "resource_monitor.json").read_text(encoding="utf-8")))
        if code:
            report["songs"].append(record)
            report["status"] = "failed"
            save()
            return code
        for identity in [item["audio"], *item["references"].values(), item["source_jams"]]:
            if file_identity(Path(identity["path"]))["sha256"] != identity["sha256"]:
                raise ValueError("Frozen input changed during inference")
        prediction = validate_timeline(read_reference(output), "Prediction")
        record.update(lab=file_identity(output), cache_provenance=json.loads(provenance_path(output).read_text(encoding="utf-8")),
                      metrics={name: evaluate(checked_reference(Path(identity["path"])), prediction)
                               for name, identity in item["references"].items()})
        report["songs"].append(record)
        save()
        print(json.dumps(dict(id=record["id"], triad={name: value["triad_accuracy"] for name, value in record["metrics"].items()})), flush=True)
    report["model_unchanged_after_run"] = model_metadata(args.madmom_python, environment) == metadata
    report["status"] = "ok" if report["model_unchanged_after_run"] else "model_changed"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
