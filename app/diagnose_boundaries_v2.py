"""Reference-only boundary ceilings on frozen dev14; never model performance.

The oracle can choose a different overlapping reference label for each head,
but must retain every original raw interval and all its coverage gaps. This is
an offline upper-bound diagnosis, not a recognizer or deployable prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from candidate_data_v2 import read_annotation, read_source
from candidate_diagnostics_v2 import checked_rows
from candidate_targets_v2 import HEADS, ReferenceCoveragePolicy, annotation_coverage, candidate_targets
from chord_semantics_v2 import parse_chord
from experiment_protocol_v2 import signature, write_new_json
from multisource_candidates_v2 import SOURCE_NAMES, interval_values

SCHEMA = "reference_only_fixed_raw_boundary_ceiling_v2"
WARNING = ("References supply oracle labels and correctness. This is not model performance or "
           "a deployable method. Heads maximize independently and need not share one joint label. "
           "All dev14 material is previously observed development data, not a blind test.")


def boundary_oracle(raw, reference, candidate_oracles=None, policy=None):
    """Decompose support, boundary mixing and candidate absence in seconds."""
    policy = policy or ReferenceCoveragePolicy()
    raw, reference = checked_rows(raw), checked_rows(reference)
    coverage = annotation_coverage(reference, policy)
    valid = np.asarray([coverage["valid_seconds"][head] for head in HEADS])
    support, best = np.zeros(len(HEADS)), np.zeros(len(HEADS))
    intervals = []
    for event in raw:
        left, right, _ = interval_values(event)
        labels, identities = [], set()
        for row in reference:
            start, end, label = interval_values(row)
            if min(right, end) <= max(left, start):
                continue
            semantic = parse_chord(label)
            if semantic.state in {"X", "missing"} or semantic.identity in identities:
                continue
            identities.add(semantic.identity)
            labels.append(label)
        if not labels:
            intervals.append(dict(start=left, end=right, overlapping_reference_candidates=0,
                                  heads={head: dict(correct_seconds=0., valid_seconds=0.) for head in HEADS}))
            continue
        targets = candidate_targets(event, labels, reference, policy)
        seconds = np.nan_to_num(targets.values, nan=0.) * targets.valid_seconds
        choice = np.argmax(seconds, axis=0)
        correct = seconds[choice, np.arange(len(HEADS))]
        supported = targets.valid_seconds[0]
        support += supported
        best += correct
        intervals.append(dict(start=left, end=right, overlapping_reference_candidates=len(labels),
            heads={head: dict(correct_seconds=float(correct[i]), valid_seconds=float(supported[i]),
                             oracle_reference_label=labels[int(choice[i])] if supported[i] else None)
                   for i, head in enumerate(HEADS)}))
    tolerance = 1e-8
    if np.any(best < -tolerance) or np.any(best > support+tolerance) or np.any(support > valid+tolerance):
        raise AssertionError("Boundary ceiling exceeded disjoint known reference support")
    fields = {}
    for i, head in enumerate(HEADS):
        fields[head] = dict(valid_reference_seconds=float(valid[i]),
            raw_supported_valid_seconds=float(support[i]),
            raw_uncovered_valid_seconds=float(max(0., valid[i]-support[i])),
            fixed_boundary_mixture_loss_seconds=float(max(0., support[i]-best[i])),
            correct_seconds=float(best[i]), accuracy=float(best[i]/valid[i]) if valid[i] else None,
            raw_support_coverage=float(support[i]/valid[i]) if valid[i] else None,
            candidate_oracles={})
        for variant, values in (candidate_oracles or {}).items():
            record = values[head]
            denominator = float(record["valid_reference_seconds"])
            correct = float(record["correct_seconds"])
            supported = float(record["candidate_support_valid_seconds"])
            if (not np.isfinite([denominator, correct, supported]).all()
                    or abs(denominator-valid[i]) > tolerance or abs(supported-support[i]) > tolerance):
                raise ValueError("Stored candidate oracle uses different reference validity or raw support")
            if correct < -tolerance or correct > best[i]+tolerance:
                raise ValueError("Stored candidate oracle exceeds the reference-label boundary ceiling")
            accuracy = correct/valid[i] if valid[i] else None
            if (record.get("accuracy") is None) != (accuracy is None) or (accuracy is not None
                    and abs(float(record["accuracy"])-accuracy) > tolerance):
                raise ValueError("Stored candidate oracle accuracy disagrees with its seconds")
            fields[head]["candidate_oracles"][variant] = dict(correct_seconds=correct,
                accuracy=accuracy, candidate_absence_loss_seconds=float(max(0., best[i]-correct)),
                total_loss_seconds=float(max(0., valid[i]-correct)))
    return dict(schema=SCHEMA, oracle_only=True, warning=WARNING, policy=policy.to_dict(),
        annotation_coverage=coverage, interval_count=len(raw), fields=fields, intervals=intervals)


def aggregate(songs):
    """Macro omits songs without a field; micro uses all valid field seconds."""
    result = {}
    for head in HEADS:
        rows = [song["boundary_oracle"]["fields"][head] for song in songs]
        known = [row for row in rows if row["valid_reference_seconds"] > 0]
        valid = sum(row["valid_reference_seconds"] for row in rows)
        correct = sum(row["correct_seconds"] for row in rows)
        field = dict(songs_with_valid_reference=len(known), valid_reference_seconds=valid,
            correct_seconds=correct, song_macro_accuracy=float(np.mean([row["accuracy"] for row in known])) if known else None,
            duration_micro_accuracy=correct/valid if valid else None,
            raw_uncovered_valid_seconds=sum(row["raw_uncovered_valid_seconds"] for row in rows),
            fixed_boundary_mixture_loss_seconds=sum(row["fixed_boundary_mixture_loss_seconds"] for row in rows),
            candidate_oracles={})
        for variant in ("v1", "v2"):
            candidate_correct = sum(row["candidate_oracles"][variant]["correct_seconds"] for row in rows)
            field["candidate_oracles"][variant] = dict(correct_seconds=candidate_correct,
                song_macro_accuracy=float(np.mean([row["candidate_oracles"][variant]["accuracy"] for row in known])) if known else None,
                duration_micro_accuracy=candidate_correct/valid if valid else None,
                candidate_absence_loss_seconds=sum(row["candidate_oracles"][variant]["candidate_absence_loss_seconds"] for row in rows))
        result[head] = field
    return result


def checked_identity(expected):
    actual = signature(expected["path"])
    if actual != expected:
        raise ValueError("Frozen input SHA or path changed: " + str(expected["path"]))
    return actual


def diagnose_report(path):
    """Verify frozen E1 identities, then parse only its 14 development songs."""
    path = Path(path)
    report_identity = signature(path)
    original = json.loads(path.read_text(encoding="utf-8"))
    if (original.get("schema") != "semantic_E1_diagnostic_v2"
            or original.get("status") != "complete_diagnostic_not_promoted"):
        raise ValueError("A completed E1 dev14 report is required")
    checked_identity(original["protocol"])
    protocol = json.loads(Path(original["protocol"]["path"]).read_text(encoding="utf-8"))
    expected_ids = protocol["development_ids"]
    songs = original["songs"]
    if (len(expected_ids) != 14 or len(set(expected_ids)) != 14 or len(songs) != 14
            or {row["id"] for row in songs} != set(expected_ids)
            or set(expected_ids) & set(protocol["fit_ids"])):
        raise ValueError("Only the frozen, disjoint dev14 song set may be diagnosed")
    # No model is loaded. Its contract verifies shared target/mask semantics.
    checked_identity(original["contract"])
    contract = json.loads(Path(original["contract"]["path"]).read_text(encoding="utf-8"))
    for name in ("chord_semantics_v2.py", "candidate_targets_v2.py", "multisource_candidates_v2.py"):
        if signature(Path(__file__).with_name(name))["sha256"] != contract["implementation"][name]:
            raise ValueError("Reference/candidate semantics changed since E1: " + name)
    output = []
    for song in songs:
        identities = song["input_identities"]
        if set(identities) != {"audio", "notes", "reference", *SOURCE_NAMES}:
            raise ValueError("Unexpected E1 input identities")
        for identity in identities.values():
            checked_identity(identity)
        raw, raw_repairs = read_source(identities["raw_madmom"]["path"], anchor=True)
        reference, reference_repairs = read_annotation(identities["reference"]["path"])
        policy = ReferenceCoveragePolicy(**song["annotation_coverage"]["policy"])
        diagnosis = boundary_oracle(raw, reference, song["candidates"]["oracle"], policy)
        output.append(dict(id=song["id"], input_identities=identities,
            timeline_repairs=dict(raw_madmom=raw_repairs, reference=reference_repairs),
            boundary_oracle=diagnosis,
            existing_candidate_oracles=song["candidates"]["oracle"],
            unchanged_standard_metrics=song["metrics"]))
    if signature(path) != report_identity:
        raise ValueError("E1 report changed during diagnosis")
    return dict(schema=SCHEMA, status="complete_reference_only_diagnostic", oracle_only=True,
        warning=WARNING, semantic_report=report_identity, semantic_protocol=original["protocol"],
        completed=len(output), development_ids=expected_ids, summary=aggregate(output), songs=output,
        unchanged_standard_summary=original["summary"],
        definitions=dict(denominator="All valid reference seconds for each head, including portions missing raw prediction coverage",
            raw_gaps="Never removed from the denominator; reference gaps and X remain unannotated/masked",
            boundary_loss="Valid seconds inside raw support minus the best reference-label candidate on each original raw interval",
            candidate_loss="Reference-label boundary ceiling minus existing actual-source candidate oracle on the same original raw intervals",
            heads="Independent masked field maxima, not unchanged mir_eval metrics and not jointly attainable",
            standard_metrics="Copied unchanged from E1; no standard evaluator is replaced or reinterpreted"),
        implementation=[signature(__file__), signature(Path(__file__).with_name("candidate_targets_v2.py")),
                        signature(Path(__file__).with_name("candidate_diagnostics_v2.py")),
                        signature(Path(__file__).with_name("candidate_data_v2.py"))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Never overwrite a prior boundary diagnostic")
    wall, cpu = time.perf_counter(), time.process_time()
    report = diagnose_report(args.semantic_report)
    peak = None
    try:
        import psutil
        memory = psutil.Process().memory_info()
        peak = getattr(memory, "peak_wset", memory.rss)
    except ImportError:
        pass
    report["resources"] = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
        peak_working_set_bytes=peak, command=[sys.executable, str(Path(__file__).resolve()),
            "--semantic-report", str(args.semantic_report.resolve()), "--output", str(args.output.resolve())],
        training_calls=0, model_predict_calls=0, acoustic_inference_calls=0,
        nnls_arrays_loaded=0, parsed_song_count=14, input_sha_verification=True)
    write_new_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), completed=report["completed"], summary=report["summary"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
