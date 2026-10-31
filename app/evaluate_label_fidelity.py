#!/usr/bin/env python
"""Measure external-label conversion only, using frozen existing LAB predictions.

No audio is decoded and no model is run. The frozen legacy lookup prevents a
concurrent core fix from replacing the baseline. Raw source/reference hashes
must match across phases; source-native intervals are never merged or snapped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mir_eval
import numpy as np

from evaluate_chords import TimedChord, evaluate, harte_chord, read_reference


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_inv_accuracy",
           "root_wcsr", "triad_wcsr", "tetrad_wcsr", "tetrad_inv_wcsr",
           "prediction_coverage", "missing_prediction_seconds", "unknown_prediction_seconds")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fixed_inputs() -> list[dict]:
    base = ROOT / "benchmarks/reports/session_20260926_accuracy2"
    report = read_json(base / "multisource_candidate_reliability/report.json")
    dev_ids = report["protocol"]["development_ids"]
    training_manifest = Path(report["protocol"]["manifest"]["path"])
    manifest_rows = {row["id"]: row for row in read_csv(training_manifest)}
    report_rows = {row["id"]: row for row in report["inputs"]}
    result = []
    for song_id in dev_ids:
        row, evidence = manifest_rows[song_id], report_rows[song_id]
        reference = Path(row["chord_reference"])
        if not reference.is_absolute():
            reference = training_manifest.parent / reference
        for source, lab in zip(("madmom", "chordino", "largevocab"), evidence["labs"], strict=True):
            result.append({"group": "dev14", "id": song_id, "source": source,
                           "audio_path": evidence["audio"]["path"],
                           "lab": identity(Path(lab["path"])), "reference": identity(reference)})
    for row in read_csv(base / "candidate_frozen_swd24.csv"):
        for source, field in (("madmom", "upstream"), ("chordino", "chordino"),
                              ("largevocab", "largevocab")):
            result.append({"group": "swd24", "id": row["id"], "source": source,
                           "audio_path": row["audio"], "lab": identity(Path(row[field])),
                           "reference": identity(Path(row["reference"]))})
    if len(result) != 114:
        raise ValueError("Expected fixed dev14 + SWD24, each with three sources")
    return result


def source_rows(path: Path) -> list[TimedChord]:
    """Read all native LAB labels; unlike reference readers, never filter X."""
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"Malformed source line: {path}: {line}")
        rows.append(TimedChord(float(fields[0]), float(fields[1]), fields[2]))
    return rows


def label_stats(rows: list[TimedChord]) -> dict:
    counts, seconds = Counter(), Counter()
    for row in rows:
        label = row.chord.strip().strip('"').strip("'")
        category = label.upper() if label.upper() in {"N", "X"} else "chord"
        counts[category] += 1
        seconds[category] += row.end - row.start
        if "/" in label:
            counts["slash"] += 1
            seconds["slash"] += row.end - row.start
        if harte_chord(label) == "X" and category != "X":
            counts["invalid_for_evaluator"] += 1
            seconds["invalid_for_evaluator"] += row.end - row.start
    return {"intervals": len(rows), "counts": dict(counts), "seconds": dict(seconds)}


def semantic_identity(label: str):
    normalized = harte_chord(label)
    if normalized in {"N", "X"}:
        return normalized
    root, bitmap, bass = mir_eval.chord.encode(normalized)
    return int(root), tuple(int(value) for value in bitmap), int(bass)


def fidelity(original: list[TimedChord], converted: list[TimedChord]) -> dict:
    same_boundaries = (len(original) == len(converted) and
                       all((a.start, a.end) == (b.start, b.end)
                           for a, b in zip(original, converted, strict=True)))
    changes = []
    for a, b in zip(original, converted, strict=True):
        if semantic_identity(a.chord) != semantic_identity(b.chord):
            changes.append({"start": a.start, "end": a.end, "source": a.chord,
                            "converted": b.chord})
    return {"boundaries_and_interval_count_unchanged": same_boundaries,
            "boundary_sha256": hashlib.sha256(json.dumps(
                [[r.start, r.end] for r in original], separators=(",", ":")).encode()).hexdigest(),
            "semantic_change_count": len(changes),
            "semantic_change_seconds": sum(r["end"] - r["start"] for r in changes),
            "semantic_change_examples": changes[:5],
            "source_stats": label_stats(original), "converted_stats": label_stats(converted)}


def aggregate(rows: list[dict], phase: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"], row["source"]].append(row)
    result = []
    for (group, source), songs in sorted(groups.items()):
        raw = {metric: float(np.mean([s["raw_metrics"][metric] for s in songs])) for metric in METRICS}
        converted = {metric: float(np.mean([s["converted_metrics"][metric] for s in songs])) for metric in METRICS}
        result.append({"group": group, "source": source, "song_count": len(songs),
                       "raw_macro": raw, phase + "_macro": converted,
                       "conversion_minus_raw_macro": {m: converted[m] - raw[m] for m in METRICS},
                       "source_stats": combine_stats([s["fidelity"]["source_stats"] for s in songs]),
                       "converted_stats": combine_stats([s["fidelity"]["converted_stats"] for s in songs]),
                       "semantic_change_count": sum(s["fidelity"]["semantic_change_count"] for s in songs),
                       "semantic_change_seconds": sum(s["fidelity"]["semantic_change_seconds"] for s in songs),
                       "all_boundaries_unchanged": all(s["fidelity"]["boundaries_and_interval_count_unchanged"] for s in songs)})
    return result


def combine_stats(stats: list[dict]) -> dict:
    counts, seconds = Counter(), Counter()
    for value in stats:
        counts.update(value["counts"])
        seconds.update(value["seconds"])
    return {"intervals": sum(value["intervals"] for value in stats),
            "counts": dict(counts), "seconds": dict(seconds)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("legacy", "fixed"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/reports/label_fidelity_20261003")
    args = parser.parse_args()
    output = args.output.resolve()
    mapping_path = output / "legacy_label_mapping.json"
    snapshot_path = output / "legacy_normalizer_snapshot.json"
    mapping = read_json(mapping_path)
    source_hashes = {str(Path(r["path"]).resolve()): r["sha256"] for r in mapping["source_labs"]}
    if args.phase == "legacy":
        inputs = fixed_inputs()
        convert = mapping["labels"].__getitem__
        core_identity = {"path": str(ROOT / "app/chord_studio.py"),
                         "sha256": mapping["legacy_source_sha256"], "frozen_snapshot": identity(snapshot_path)}
        previous = None
    else:
        from chord_studio import normalize_external_chord_label, read_external_chord_hints
        previous = read_json(output / "legacy.json")
        inputs = previous["inputs"]
        convert = lambda label: normalize_external_chord_label(label, mapping["notation"])
        core_identity = identity(ROOT / "app/chord_studio.py")
        if core_identity["sha256"] == mapping["legacy_source_sha256"]:
            raise ValueError("Core is still the frozen legacy version")
        if identity(ROOT / "app/evaluate_chords.py") != previous["protocol"]["evaluator"]:
            raise ValueError("Evaluator changed between phases")
    rows = []
    for entry in inputs:
        lab = Path(entry["lab"]["path"])
        reference_path = Path(entry["reference"]["path"])
        if digest(lab) != entry["lab"]["sha256"] or digest(lab) != source_hashes[str(lab.resolve())]:
            raise ValueError(f"Source LAB changed: {lab}")
        if digest(reference_path) != entry["reference"]["sha256"]:
            raise ValueError(f"Reference changed: {reference_path}")
        original = source_rows(lab)
        converted = [TimedChord(r.start, r.end, convert(r.chord)) for r in original]
        if args.phase == "fixed":
            ingested = read_external_chord_hints(lab, mapping["notation"])
            if [(r.start, r.end, r.chord) for r in ingested] != [(r.start, r.end, r.chord) for r in converted]:
                raise ValueError(f"Actual ingestion differs from label-only conversion: {lab}")
        reference = read_reference(reference_path)
        rows.append({"group": entry["group"], "id": entry["id"], "source": entry["source"],
                     "raw_metrics": evaluate(reference, original),
                     "converted_metrics": evaluate(reference, converted),
                     "fidelity": fidelity(original, converted)})
    report = {
        "schema": "external_label_fidelity_v1", "phase": args.phase,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scope": "Existing predictions, pure label conversion and LAB ingestion only",
                     "audio_decoded": False, "models_run": False, "model_fit": False,
                     "threshold_search": False, "source_intervals_resampled": False,
                     "metric_contract": "strict-missing-v2; complete known reference support; song macro",
                     "unknown_reference_X": "excluded, not N", "unknown_prediction_X": "retained; not silence",
                     "limitations": ["This measures representation fidelity, not acoustic-model improvement",
                                     "SWD is an external regression audit, never used for parameter selection",
                                     "Actual fusion decisions and final UI are outside this label-only report"],
                     "legacy_mapping": identity(mapping_path), "normalizer": core_identity,
                     "evaluator": identity(ROOT / "app/evaluate_chords.py"),
                     "script": identity(Path(__file__))},
        "inputs": inputs, "aggregate": aggregate(rows, args.phase), "songs": rows,
        "synthetic_label_probes": [{"input": label, "output": convert(label),
                                     "in_scored_corpus": False} for label in
                                    ("N", "X", "x", "C/E", "C:maj/3", "D:min/b3",
                                     "C:maj7/7", "Bb:7/3", "F#:min/5", "unknown")],
    }
    if previous is not None:
        old_groups = {(r["group"], r["source"]): r for r in previous["aggregate"]}
        report["comparison"] = [
            {"group": current["group"], "source": current["source"],
             "fixed_minus_legacy_macro": {
                 metric: current["fixed_macro"][metric] - old_groups[current["group"], current["source"]]["legacy_macro"][metric]
                 for metric in METRICS}}
            for current in report["aggregate"]]
        report["protocol"]["legacy_report"] = identity(output / "legacy.json")
    target = output / (args.phase + ".json")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite frozen report: {target}")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), "tracks_and_sources": len(rows),
                      "aggregate": report["aggregate"], "comparison": report.get("comparison")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
