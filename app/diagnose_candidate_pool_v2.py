"""Known-set candidate diagnostics from existing, hash-verified LAB files.

No model, NNLS cache or audio file is opened. This does not fit or select any
weights, margins or policies. Oracle values are reference-only field bounds,
not achievable predictions and not new independent validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from candidate_data_v2 import ReferenceInterval, read_source, read_annotation
from candidate_diagnostics_v2 import candidate_diagnostics
from candidate_targets_v2 import HEADS, ReferenceCoveragePolicy, annotation_coverage
from chord_semantics_v2 import parse_chord
from multisource_candidates_v2 import SOURCE_NAMES, interval_values

DATASETS = {"regression17": 17, "user4": 4, "swd24": 24}


def signature(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_path(identity: dict, relative_to: Path) -> Path:
    path = Path(identity["path"])
    path = path if path.is_absolute() else relative_to / path
    if signature(path)["sha256"] != identity["sha256"]:
        raise ValueError(f"Input content changed since the completed policy report: {path}")
    return path.resolve()


def source_inventory(rows) -> dict:
    counts, seconds = Counter(), Counter()
    extensions = {str(degree): {"intervals": 0, "seconds": 0.0} for degree in (9, 11, 13)}
    labels = set()
    for row in rows:
        value = parse_chord(row.chord)
        counts[value.state] += 1
        seconds[value.state] += row.end - row.start
        if value.extension_degrees:
            labels.add(row.chord)
        for degree in (9, 11, 13):
            if value.extension(degree):
                extensions[str(degree)]["intervals"] += 1
                extensions[str(degree)]["seconds"] += row.end - row.start
    return {"interval_count": len(rows), "state_counts": dict(counts), "state_seconds": dict(seconds),
            "extended_labels": sorted(labels), "extensions": extensions,
            "duration_note": "Native source-interval totals; candidate support itself uses interval union"}


def diagnostic_reference(path: Path):
    """Offline-only float cleanup; the strict training reader remains unchanged.

    Adjacent JSON endpoints calculated by different floating-point expressions
    may differ by a few ulps. Only <=1e-9 second overlaps are clipped, retaining
    the complete ReferenceInterval including field masks. This is not alignment.
    """
    try:
        rows, metadata = read_annotation(path)
        return rows, {**metadata, "diagnostic_float_overlap_repairs": 0,
                      "diagnostic_float_max_overlap_seconds": 0.0,
                      "diagnostic_float_total_overlap_seconds": 0.0}
    except ValueError as original_error:
        if path.suffix.lower() != ".json":
            raise
        data = json.loads(path.read_text(encoding="utf-8"))
        data = data.get("events", data.get("annotations", [])) if isinstance(data, dict) else data
        rows = [ReferenceInterval(float(row["start"]), float(row["end"]), row.get("chord"),
                                  tuple(row["valid_fields"]) if row.get("valid_fields") is not None else None,
                                  row.get("field_mask")) for row in data]
        rows.sort(key=lambda row: (row.start, row.end))
        count, maximum, total = 0, 0.0, 0.0
        for index, row in enumerate(rows):
            left, _, _ = interval_values(row)
            if left < 0:
                raise ValueError("Negative diagnostic reference time") from original_error
            if index and rows[index - 1].end > left:
                previous = rows[index - 1]
                overlap = previous.end - left
                if overlap > 1e-9 or previous.start >= left:
                    raise ValueError("Reference overlap exceeds offline floating-point tolerance") from original_error
                rows[index - 1] = replace(previous, end=left)
                count += 1
                maximum = max(maximum, overlap)
                total += overlap
        if not count:
            raise original_error
        return rows, {"rounding_overlap_repairs": 0,
                      "diagnostic_float_overlap_repairs": count,
                      "diagnostic_float_max_overlap_seconds": maximum,
                      "diagnostic_float_total_overlap_seconds": total,
                      "policy": "Offline diagnostic only: JSON adjacent overlap <=1e-9s clipped on copied dataclasses with field masks retained; larger overlaps rejected",
                      "strict_training_reader_error": str(original_error)}


def summarize(songs: list[dict]) -> dict:
    diagnostics = [song["candidates"] for song in songs]
    intervals = sum(row["intervals"] for row in diagnostics)
    count = {variant: sum(row["candidates"][variant] for row in diagnostics) for variant in ("v1", "v2")}
    eligible = sum(row["extended_source_candidate_identities"] for row in diagnostics)
    retained = {variant: sum(row["extensions_retained"][variant]["count"] for row in diagnostics)
                for variant in ("v1", "v2")}
    oracle = {}
    for head in HEADS:
        records = {variant: [row["oracle"][variant][head] for row in diagnostics] for variant in ("v1", "v2")}
        denominator = sum(row["valid_reference_seconds"] for row in records["v2"])
        correct = {variant: sum(row["correct_seconds"] for row in values) for variant, values in records.items()}
        macros = {}
        valid_song_count = sum(row["accuracy"] is not None for row in records["v2"])
        for variant, values in records.items():
            valid = [row["accuracy"] for row in values if row["accuracy"] is not None]
            macros[variant] = sum(valid) / len(valid) if valid else None
        oracle[head] = {
            "valid_reference_seconds": denominator, "valid_song_count": valid_song_count,
            "candidate_support_valid_seconds": sum(row["candidate_support_valid_seconds"] for row in records["v2"]),
            "correct_seconds": correct,
            "duration_micro": {variant: value / denominator if denominator else None for variant, value in correct.items()},
            "valid_song_macro": macros,
            "v2_minus_v1_correct_seconds": correct["v2"] - correct["v1"],
            "v2_minus_v1_duration_micro": (correct["v2"] - correct["v1"]) / denominator if denominator else None,
            "v2_minus_v1_valid_song_macro": macros["v2"] - macros["v1"] if valid_song_count else None,
        }
    span = sum(song["annotation_coverage"]["span_seconds"] for song in songs)
    covered = {head: sum(song["annotation_coverage"]["valid_seconds"][head] for song in songs) for head in HEADS}
    return {"songs": len(songs), "intervals": intervals, "candidate_count": count,
            "candidate_count_change": count["v2"] - count["v1"],
            "mean_candidates_per_interval": {variant: value / intervals if intervals else None for variant, value in count.items()},
            "extended_source_candidate_identities": eligible,
            "extensions_retained": {variant: {"count": retained[variant], "fraction": retained[variant] / eligible if eligible else None}
                                     for variant in ("v1", "v2")},
            "seven_head_oracle": oracle,
            "annotation_coverage": {"reference_span_seconds": span, "valid_seconds": covered,
                                    "duration_coverage": {head: seconds / span if span else None for head, seconds in covered.items()}},
            "reference_rounding_repairs": sum(song["repairs"]["reference"]["rounding_overlap_repairs"] for song in songs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are never overwritten")
    parser.add_argument("--prior-completed", type=int, help="Record a stopped prior attempt without inventing partial scores")
    parser.add_argument("--prior-stop-reason", help="Reason for the explicitly recorded prior stopped attempt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.prior_completed is None) != (args.prior_stop_reason is None):
        raise ValueError("Provide both prior attempt count and stop reason, or neither")
    if args.prior_completed is not None and not 0 <= args.prior_completed < sum(DATASETS.values()):
        raise ValueError("Invalid prior attempt count")
    started, cpu_started = time.perf_counter(), time.process_time()
    policy_path = args.policy_summary.resolve()
    summary_identity = signature(policy_path)
    summary = json.loads(policy_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("final_failures") != 0:
        raise ValueError("A completed policy summary without unresolved failures is required")
    policy = ReferenceCoveragePolicy()
    reports, rows_by_dataset = {}, {}
    # Freeze exactly the already completed songs. No score-based subset choice.
    for dataset, count in DATASETS.items():
        path = checked_path(summary["datasets"][dataset]["report"], policy_path.parent)
        report = json.loads(path.read_text(encoding="utf-8"))
        songs = report["songs"]
        if report.get("failure_count") or len(songs) != count or len({song["id"] for song in songs}) != count:
            raise ValueError(f"Incomplete or duplicate {dataset} song set")
        reports[dataset] = signature(path)
        rows_by_dataset[dataset] = songs
    output_songs = []
    for dataset, rows in rows_by_dataset.items():
        report_base = Path(reports[dataset]["path"]).parent
        for song in rows:
            evidence = song["inputs"]
            reference_path = checked_path(evidence["reference"], report_base)
            source_paths, sources, repairs = {}, {}, {}
            source_evidence = {}
            for name, source in zip(SOURCE_NAMES, evidence["sources"], strict=True):
                path = checked_path(source["lab"], report_base)
                source_paths[name] = path
                sources[name], repairs[name] = read_source(path, anchor=name == "raw_madmom")
                source_evidence[name] = {"lab": signature(path),
                                        "prior_generation_identity": source.get("generation_identity"),
                                        "prior_sidecar_evidence": source.get("sidecars", [])}
            reference, repairs["reference"] = diagnostic_reference(reference_path)
            if not sources["raw_madmom"] or not reference:
                raise ValueError(f"Empty raw anchors or reference: {dataset}/{song['id']}")
            diagnostic = candidate_diagnostics(sources["raw_madmom"], sources, reference, policy=policy)
            coverage = annotation_coverage(reference, policy)
            # Verify the small text inputs again after diagnosis, never audio or NPZ.
            checked_path(evidence["reference"], report_base)
            for source in evidence["sources"]:
                checked_path(source["lab"], report_base)
            output_songs.append({"dataset": dataset, "id": song["id"],
                                 "reference_status": song.get("reference_status"),
                                 "input_identities": {"sources": source_evidence, "reference": signature(reference_path)},
                                 "audio_identity_from_prior_report_not_reopened": evidence.get("audio"),
                                 "repairs": repairs,
                                 "source_inventory": {name: source_inventory(source) for name, source in sources.items()},
                                 "candidates": diagnostic, "annotation_coverage": coverage})
            print(f"Candidate-only diagnostic {len(output_songs)}/45: {dataset}/{song['id']}", flush=True)
    if signature(policy_path) != summary_identity:
        raise ValueError("Policy summary changed during the diagnostic")
    for dataset, identity in reports.items():
        checked_path(identity, policy_path.parent)
    result = {
        "schema": "known_candidate_pool_diagnostic_v2", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "policy_summary": summary_identity,
        "source_reports": reports, "skipped_dataset": "dev14 (already reported in E1)",
        "known_regression_only": True, "new_blind_test": False,
        "model_loaded": False, "NNLS_cache_opened": False, "audio_opened": False,
        "training_calls": 0, "model_predict_calls": 0, "acoustic_inference_calls": 0,
        "weights_or_margin_selected": False, "policy": policy.to_dict(),
        "protocol_notes": [
            "All 17 regression, 4 user and 24 SWD songs in the completed policy summary are retained.",
            "LAB and reference hashes must match prior evidence before and after diagnosis.",
            "Oracle numerators use the best constant candidate on each unchanged raw interval, independently for each head.",
            "Oracle denominators retain all valid reference-field duration, including time outside raw predictions.",
            "Field oracles are not standard mir_eval scores or jointly attainable model performance.",
            "No source extension candidates means retention fraction null, never 0 or 1.",
            "The default reference policy treats absent bass/upper extensions as unannotated, not negative.",
            "Offline JSON references alone may receive <=1e-9s floating-point endpoint cleanup with masks preserved; strict v2 training reader and original files remain unchanged.",
            "Old source generation identity assertions are quoted from prior reports; audio and sidecars are not revalidated here.",
        ],
        "implementation": {name: signature(Path(__file__).with_name(name)) for name in
                           (Path(__file__).name, "candidate_data_v2.py", "candidate_diagnostics_v2.py",
                            "candidate_targets_v2.py", "chord_semantics_v2.py", "multisource_candidates_v2.py")},
        "datasets": {dataset: summarize([song for song in output_songs if song["dataset"] == dataset]) for dataset in DATASETS},
        "all_known": summarize(output_songs), "songs": output_songs,
        "attempt_history": ([{"status": "stopped", "completed_songs_before_stop": args.prior_completed,
                              "requested_songs": sum(DATASETS.values()), "reason": args.prior_stop_reason,
                              "partial_report_published": False}] if args.prior_completed is not None else [])
                           + [{"status": "complete", "completed_songs": len(output_songs),
                               "scope": "All fixed songs re-diagnosed from text inputs; no model or audio work"}],
        "resources": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output.resolve()), "all_known": result["all_known"],
                      "resources": result["resources"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
