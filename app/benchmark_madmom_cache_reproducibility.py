"""Regenerate a complete frozen manifest and audit legacy raw madmom caches."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import csv

from evaluate_chords import evaluate, read_reference, validate_timeline
from madmom_bridge import DEFAULT_MADMOM_PYTHON, file_identity, provenance_path
from sheetsage2_bridge import REPO_ROOT, run_worker_monitored

MODEL_METADATA = r'''
import hashlib,inspect,json,platform,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import madmom,numpy,scipy
import madmom.features.chords as chords
from madmom.models import CHORDS_CNN_FEAT,CHORDS_CFCRF
def identity(p):
 p=Path(p)
 return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'runtime_python':platform.python_version(),'madmom_version':madmom.__version__,
 'numpy_version':numpy.__version__,'scipy_version':scipy.__version__,
 'model_weights':{'cnn':identity(CHORDS_CNN_FEAT[0]),'crf':identity(CHORDS_CFCRF[0])},
 'processor_source':identity(inspect.getfile(chords)),
 'constructors':{'cnn':inspect.getsource(chords.CNNChordFeatureProcessor.__init__),
                 'crf':inspect.getsource(chords.CRFChordRecognitionProcessor.__init__)}}))
'''


def model_metadata(runtime: Path, environment: dict) -> dict:
    completed = subprocess.run([str(runtime), "-c", MODEL_METADATA], env=environment,
                               capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
    return json.loads(completed.stdout)


def compare_labs(old_path: Path, fresh_path: Path) -> dict:
    old_text = old_path.read_text(encoding="utf-8-sig")
    fresh_text = fresh_path.read_text(encoding="utf-8-sig")
    old, fresh = read_reference(old_path), read_reference(fresh_path)
    differences = []
    for index in range(max(len(old), len(fresh))):
        left = old[index] if index < len(old) else None
        right = fresh[index] if index < len(fresh) else None
        if left != right:
            differences.append(dict(index=index, old=asdict(left) if left else None,
                                    fresh=asdict(right) if right else None))
    return dict(byte_identical=old_path.read_bytes() == fresh_path.read_bytes(),
                text_identical=old_text == fresh_text,
                old_text_sha256=hashlib.sha256(old_text.encode("utf-8")).hexdigest(),
                fresh_text_sha256=hashlib.sha256(fresh_text.encode("utf-8")).hexdigest(),
                parsed_events_identical=not differences,
                old_event_count=len(old), fresh_event_count=len(fresh),
                differing_event_positions=len(differences), first_differences=differences[:20])


def summarize(songs: list[dict]) -> dict:
    good = [row for row in songs if row.get("status") == "ok"]
    result = dict(completed=len(good), failures=len(songs) - len(good),
                  byte_identical_songs=sum(row["comparison"]["byte_identical"] for row in good),
                  text_identical_songs=sum(row["comparison"]["text_identical"] for row in good),
                  parsed_events_identical_songs=sum(row["comparison"]["parsed_events_identical"] for row in good))
    for policy in ("old", "fresh"):
        result[policy] = {}
        for metric in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy",
                       "root_wcsr", "triad_wcsr", "tetrad_wcsr", "tetrad_inv_wcsr", "prediction_coverage"):
            values = [row[policy + "_metrics"] for row in good]
            if not values:
                continue
            weight = "non_no_chord_seconds" if metric.endswith("accuracy") else "evaluated_seconds"
            result[policy]["macro_" + metric] = sum(v[metric] for v in values) / len(values)
            result[policy]["duration_weighted_" + metric] = sum(v[metric] * v[weight] for v in values) / sum(v[weight] for v in values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("old_labs", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--madmom-python", type=Path, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument("--stop-starting-at", required=True, help="Timezone-aware deadline for starting another song")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.stop_starting_at)
    if deadline.tzinfo is None:
        parser.error("Deadline must include a timezone")
    if (args.output / "report.json").exists():
        raise FileExistsError("Use a new experiment directory; existing runs must remain intact")
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate manifest song IDs")
    frozen = []
    for row in rows:
        frozen.append(dict(id=row["id"], audio=file_identity(args.manifest.parent / row["audio"]),
                           reference=file_identity(args.manifest.parent / row["chord_reference"]),
                           old_lab=file_identity(args.old_labs / (row["id"] + ".lab"))))
    args.output.mkdir(parents=True, exist_ok=True)
    labs = args.output / "labs"
    labs.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    metadata = model_metadata(args.madmom_python, environment)
    report = dict(protocol="frozen_complete_manifest_fresh_raw_cnn_crf_no_reference_inputs",
                  manifest=file_identity(args.manifest), frozen_songs=frozen, model=metadata,
                  threads=2, sample_rate=44100, audio_offset=0, requested_duration=None,
                  old_cache_role="Comparison only; never passed to fresh recognition",
                  scoring_protocol="full reference support, exact duration, official mir_eval semantics; missing predictions filled N and coverage reported",
                  stop_starting_at=args.stop_starting_at, songs=[], status="running")
    def save():
        report["summary"] = summarize(report["songs"])
        temporary = args.output / "report.pending.json"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output / "report.json")
    save()
    for item in frozen:
        if datetime.now(deadline.tzinfo) >= deadline:
            report["status"] = "stopped_at_deadline"
            save()
            return 2
        directory = args.output / "runs" / item["id"]
        directory.mkdir(parents=True, exist_ok=True)
        output = labs / (item["id"] + ".lab")
        if output.exists():
            raise FileExistsError(output)
        print(f"Fresh {len(report['songs']) + 1}/{len(frozen)} {item['id']}", flush=True)
        command = [sys.executable, str(REPO_ROOT / "app/madmom_bridge.py"), item["audio"]["path"],
                   "-o", str(output.resolve()), "--sample-rate", "44100", "--madmom-python", str(args.madmom_python)]
        record = dict(id=item["id"], command=command, status="running")
        report["running"] = item["id"]
        save()
        code = run_worker_monitored(command, environment, directory, active_timeout=300, wall_timeout=600)
        record.update(return_code=code, resources=json.loads((directory / "resource_monitor.json").read_text(encoding="utf-8")))
        if code:
            record["status"] = "failed"
            report["songs"].append(record)
            report["status"] = "failed"
            save()
            return code
        for name in ("audio", "reference", "old_lab"):
            if file_identity(Path(item[name]["path"]))["sha256"] != item[name]["sha256"]:
                raise ValueError(f"Frozen {name} changed for {item['id']}")
        reference = validate_timeline(read_reference(Path(item["reference"]["path"])), "Reference")
        old = validate_timeline(read_reference(Path(item["old_lab"]["path"])), "Old prediction")
        fresh = validate_timeline(read_reference(output), "Fresh prediction")
        record.update(status="ok", fresh_lab=file_identity(output),
                      cache_provenance=json.loads(provenance_path(output).read_text(encoding="utf-8")),
                      old_metrics=evaluate(reference, old), fresh_metrics=evaluate(reference, fresh),
                      comparison=compare_labs(Path(item["old_lab"]["path"]), output))
        report["songs"].append(record)
        report["running"] = None
        save()
        print(json.dumps(dict(id=item["id"], equal=record["comparison"]["parsed_events_identical"],
                              triad=record["fresh_metrics"]["triad_accuracy"],
                              elapsed=record["resources"]["wall_seconds"])), flush=True)
    report["model_unchanged_after_run"] = model_metadata(args.madmom_python, environment) == metadata
    report["status"] = "ok" if report["model_unchanged_after_run"] else "model_changed"
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
