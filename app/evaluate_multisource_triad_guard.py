"""Evaluate one predeclared output guard on existing B52-fit/dev14 predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_chords import evaluate, read_reference, validate_timeline
from generate_madmom_chords_from_manifest import output_path_for_row
from multisource_triad_guard import GUARD_RULE, GUARD_SCHEMA, preserve_raw_triad
import multisource_root_guard as root_guard
from train_multisource_seventh_gate import signature, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-labs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard", choices=("triad", "root"), default="triad")
    args = parser.parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    dev_ids = set(training["protocol"]["development_ids"])
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 66 or len(dev_ids) != 14 or any(row.get("split") != "train" for row in rows):
        raise ValueError("Only the fixed training66 manifest and its dev14 predictions are allowed")
    rows = [row for row in rows if row["id"] in dev_ids]
    if len(rows) != 14 or {row["id"] for row in rows} != dev_ids:
        raise ValueError("Missing or duplicate development songs")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise ValueError("Use a new directory; preserve the original declaration")
    guard_function = root_guard.preserve_raw_root if args.guard == "root" else preserve_raw_triad
    guard_schema = root_guard.GUARD_SCHEMA if args.guard == "root" else GUARD_SCHEMA
    guard_rule = root_guard.GUARD_RULE if args.guard == "root" else GUARD_RULE
    guard_file = "multisource_root_guard.py" if args.guard == "root" else "multisource_triad_guard.py"
    protocol = dict(schema=guard_schema, rule=guard_rule, status="declared_before_development_scores",
        training_report=signature(args.training_report), manifest=signature(args.manifest),
        development_ids=sorted(dev_ids), model=training["development_model"],
        implementation=[signature(__file__), signature(Path(__file__).with_name(guard_file))],
        additional_configurations=0, fitting=False, threshold_changes=False,
        selection_condition=("Expose only if root is unchanged, triad >= raw and tetrad > raw" if args.guard == "root" else
                             "Expose this fixed guard if development tetrad is strictly above raw and root/triad are unchanged"),
        motivation="Root agent reported user4 cross-domain root degradation; no external song identities or scores read by this experiment. Rule fixed before SWD scores; a development-selected revision following user4 feedback, not blind to existence of cross-domain failure.",
        preceding_guard_outcome=("Root+triad preservation reduced dev tetrad; root-only is the final permitted fixed rule, no further guard search" if args.guard == "root" else None))
    write_json(args.output / "protocol.json", protocol)
    report = dict(protocol=protocol, songs=[])
    for row in rows:
        raw_path = output_path_for_row(args.raw_labs, row["id"])
        selected_path = output_path_for_row(args.predictions, row["id"])
        raw = validate_timeline(read_reference(raw_path), "Raw")
        selected = validate_timeline(read_reference(selected_path), "Frozen development prediction")
        guarded, details = guard_function(raw, selected)
        reference_path = (args.manifest.parent / row["chord_reference"]).resolve()
        reference = validate_timeline(read_reference(reference_path), "Development reference")
        metrics = dict(baseline=evaluate(reference, raw), selected=evaluate(reference, selected), guarded=evaluate(reference, guarded))
        invariants = ("root_accuracy", "root_wcsr", "prediction_coverage") if args.guard == "root" else ("root_accuracy", "triad_accuracy", "root_wcsr", "triad_wcsr", "prediction_coverage")
        for metric in invariants:
            if abs(metrics["baseline"][metric] - metrics["guarded"][metric]) > 1e-12:
                raise AssertionError("Guard changed raw root/triad or time coverage")
        report["songs"].append(dict(id=row["id"], **metrics, guard=details,
            inputs=dict(raw=signature(raw_path), selected=signature(selected_path), reference=signature(reference_path))))
    metrics = ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy", "tetrad_wcsr")
    report["summary"] = {kind: {metric: float(np.mean([song[kind][metric] for song in report["songs"]]))
        for metric in metrics} for kind in ("baseline", "selected", "guarded")}
    report.update(reverted_intervals=sum(song["guard"]["reverted_intervals"] for song in report["songs"]),
        retained_label_changes=sum(song["guard"]["retained_label_changes"] for song in report["songs"]),
        status="complete", completed=14)
    report["development_condition_passed"] = report["summary"]["guarded"]["tetrad_accuracy"] > report["summary"]["baseline"]["tetrad_accuracy"]
    if args.guard == "root":
        report["development_condition_passed"] &= report["summary"]["guarded"]["triad_accuracy"] >= report["summary"]["baseline"]["triad_accuracy"]
    write_json(args.output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("summary", "reverted_intervals", "retained_label_changes", "development_condition_passed")}, indent=2))


if __name__ == "__main__":
    main()
