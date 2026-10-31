"""Training-only automatic chord candidates, fixed voting and oracle bounds.

Candidate generation and voting never receive reference labels. Separate
reference-based oracle functions are unrealizable upper bounds, not models.
No held-out manifest, NNLS evaluation result or model fitting is involved.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import mir_eval
import numpy as np

from evaluate_chords import TimedChord, evaluate, harte_chord, read_reference, validate_timeline
from generate_madmom_chords_from_manifest import output_path_for_row
from train_multisource_seventh_gate import signature, write_json


def chord_identity(label):
    """Pitch-equivalent full labels retain explicit extensions and bass."""
    try:
        normalized = harte_chord(label)
        if normalized == "X":
            return None
        if normalized == "N":
            return ("N",)
        root, notes, bass = mir_eval.chord.encode(normalized)
        return (int(root), tuple(int(v) for v in notes), int(bass))
    except (ValueError, mir_eval.chord.InvalidChordException):
        return None


def triad_identity(label):
    normalized = harte_chord(label)
    if normalized in {"N", "X"}:
        return (normalized,)
    root, notes, _ = mir_eval.chord.encode(normalized)
    # Same reduction used by mir_eval's triad comparator: the bottom eight
    # semitones retain suspended, minor, major and diminished triad tones.
    # Use pairwise mir_eval.triads below to avoid assuming a quality mapping.
    return int(root)


def candidate_pool(raw, external_rows):
    """Return [{label,source_support[3]}], raw first, with no label supervision."""
    raw_key = chord_identity(raw.chord)
    candidates = [dict(label=raw.chord, source_support=np.array([1., 0., 0.]))]
    lookup = {raw_key: 0} if raw_key is not None else {}
    duration = raw.end - raw.start
    for source, rows in enumerate(external_rows, start=1):
        for row in rows:
            if row.end <= raw.start:
                continue
            if row.start >= raw.end:
                break
            overlap = min(raw.end, row.end) - max(raw.start, row.start)
            key = chord_identity(row.chord)
            if overlap <= 0 or key is None:
                continue
            if key not in lookup:
                lookup[key] = len(candidates)
                candidates.append(dict(label=row.chord, source_support=np.zeros(3)))
            candidates[lookup[key]]["source_support"][source] += overlap / duration
    return candidates


def fixed_vote(candidates, group_triads=False):
    support = np.asarray([c["source_support"].sum() for c in candidates])
    if not group_triads:
        winner = int(np.argmax(support))
        return winner if support[winner] > 1.5 else 0
    groups = []
    for index, candidate in enumerate(candidates):
        label = harte_chord(candidate["label"])
        assigned = False
        for group in groups:
            representative = harte_chord(candidates[group[0]]["label"])
            if mir_eval.chord.triads([representative], [label])[0] == 1:
                group.append(index)
                assigned = True
                break
        if not assigned:
            groups.append([index])
    group_support = [float(support[group].sum()) for group in groups]
    winner = int(np.argmax(group_support))
    if group_support[winner] <= 1.5:
        return 0
    group = groups[winner]
    return group[int(np.argmax(support[group]))]


def reference_based_oracle(raw, candidates, reference, comparator):
    """UNREALIZABLE: use reference overlap to find a per-interval upper bound."""
    labels, seconds = [], []
    for row in reference:
        if row.end <= raw.start:
            continue
        if row.start >= raw.end:
            break
        overlap = min(raw.end, row.end) - max(raw.start, row.start)
        label = harte_chord(row.chord)
        if overlap > 0 and label not in {"N", "X"}:
            labels.append(label)
            seconds.append(overlap)
    if not labels:
        return 0
    weights = np.asarray(seconds)
    scores = [float(weights @ comparator(labels, [harte_chord(c["label"])] * len(labels))) for c in candidates]
    return int(np.argmax(scores))


def analyze_song(raw, external, reference):
    paths = {name: [] for name in ("raw", "full_majority", "triad_majority", "oracle_root", "oracle_triad", "oracle_tetrad")}
    candidates_count = []
    changed = {name: 0 for name in paths}
    for row in raw:
        candidates = candidate_pool(row, external)
        candidates_count.append(len(candidates))
        chosen = dict(raw=0, full_majority=fixed_vote(candidates), triad_majority=fixed_vote(candidates, True),
            oracle_root=reference_based_oracle(row, candidates, reference, mir_eval.chord.root),
            oracle_triad=reference_based_oracle(row, candidates, reference, mir_eval.chord.triads),
            oracle_tetrad=reference_based_oracle(row, candidates, reference, mir_eval.chord.tetrads))
        for name, choice in chosen.items():
            label = candidates[choice]["label"]
            paths[name].append(TimedChord(row.start, row.end, label))
            changed[name] += int(chord_identity(label) != chord_identity(row.chord))
    metrics = {name: evaluate(reference, rows) for name, rows in paths.items()}
    return dict(metrics=metrics, changed=changed, intervals=len(raw),
                mean_candidate_count=float(np.mean(candidates_count)), max_candidate_count=max(candidates_count, default=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-protocol", type=Path, required=True)
    parser.add_argument("--madmom-labs", type=Path, required=True)
    parser.add_argument("--chordino-labs", type=Path, required=True)
    parser.add_argument("--largevocab-labs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise ValueError("Do not overwrite a frozen study")
    started = time.perf_counter()
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    dev_ids = set(split["split_ids"])
    ids = {row["id"] for row in rows}
    if len(rows) != 66 or len(ids) != 66 or len(dev_ids) != 14 or not dev_ids <= ids or any(row.get("split") != "train" for row in rows):
        raise ValueError("Only the fixed training66 and its 52/14 split are allowed")
    protocol = dict(status="predeclared_before_scores", protocol="automatic_multisource_candidate_potential_train_only",
        manifest=signature(args.manifest), split_protocol=signature(args.split_protocol),
        candidate_rule="Raw plus every valid full chord with positive external overlap; source supports are duration fractions; explicit N valid, X/unknown/missing never become N",
        fixed_rules={"full_majority": "Unique maximum full-chord support >1.5 of 3 source votes; raw first wins ties; otherwise raw",
                     "triad_majority": "Sum full candidates within mir_eval triad-equivalent groups; winning group must exceed1.5; choose highest supported full label within group; raw first wins ties"},
        oracle_warning="Reference-based unattainable independent root/triad/tetrad per-interval upper bounds. Never used as a classifier or candidate generator.",
        source_identity_limitation="Current same-named manifest association only; legacy LABs lack original audio/model/config SHA sidecars",
        fit_ids=[row["id"] for row in rows if row["id"] not in dev_ids],
        dev_ids=[row["id"] for row in rows if row["id"] in dev_ids],
        model_fitting=False, heldout_labels_opened=False, implementation=signature(__file__))
    write_json(args.output / "protocol.json", protocol)
    report = dict(protocol=protocol, songs=[])
    for index, row in enumerate(rows):
        paths = [output_path_for_row(directory, row["id"]) for directory in
                 (args.madmom_labs, args.chordino_labs, args.largevocab_labs)]
        labs = [validate_timeline(read_reference(path), path.name) if path.is_file() else [] for path in paths]
        if not labs[0]:
            raise ValueError("Raw upstream LAB is required")
        reference = validate_timeline(read_reference(args.manifest.parent / row["chord_reference"]), "Reference")
        result = analyze_song(labs[0], labs[1:], reference)
        result.update(id=row["id"], split="dev14" if row["id"] in dev_ids else "fit52",
            inputs=[signature(path) if path.is_file() else dict(path=str(path.resolve()), missing=True) for path in paths])
        report["songs"].append(result)
        print(f"Candidate potential {index+1}/66: {row['id']}", flush=True)
    report["summary"] = {}
    for split_name in ("fit52", "dev14"):
        songs = [song for song in report["songs"] if song["split"] == split_name]
        summary = {}
        for mode in songs[0]["metrics"]:
            summary[mode] = {metric: float(np.mean([song["metrics"][mode][metric] for song in songs])) for metric in
                ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_wcsr")}
            summary[mode]["changed_intervals"] = sum(song["changed"][mode] for song in songs)
        summary["mean_candidates_per_interval"] = float(sum(song["mean_candidate_count"] * song["intervals"] for song in songs) / sum(song["intervals"] for song in songs))
        report["summary"][split_name] = summary
    report.update(status="complete_training_side_potential_only", elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / "report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
