"""Two frozen candidate-reliability models on B52; select only on dev14.

All candidate/evidence construction is reference-independent. Existing NNLS
caches are mandatory: this program cannot start Sonic or external recognizers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import joblib
import mir_eval
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import harmonic_note_features as notes
import chord_studio as studio
from analyze_multisource_candidate_potential import candidate_pool, fixed_vote, chord_identity
from evaluate_chords import TimedChord, evaluate, harte_chord, read_reference, validate_timeline
from generate_madmom_chords_from_manifest import output_path_for_row
from nnls_seventh_model import acoustic_windows
from train_multisource_seventh_gate import signature, write_json, HGB_PARAMETERS

SCHEMA = "cached_nnls_multisource_relative_candidate_utility_v1"
DIMENSION = 136
MARGINS = [.30, .20, .15, .10, .05, .02, 0.]
UTILITY_WEIGHTS = (.25, .50, .25)
MODELS = {"hgb_regression": HGB_PARAMETERS, "ridge_regression": dict(alpha=10.)}


def load_existing_evidence(audio, directory):
    identity = notes.extraction_identity(audio)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(directory) / (key + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Required NNLS cache missing; no inference fallback is allowed: {path}")
    with np.load(path, allow_pickle=False) as data:
        evidence = notes.NoteEvidence(data["times"], data["activations"], json.loads(str(data["metadata"])))
    if evidence.metadata.get("identity") != identity:
        raise ValueError("NNLS cache identity mismatch")
    notes._validate_evidence(evidence)
    return evidence, signature(path)


def _candidate_components(label):
    normalized = harte_chord(label)
    if normalized in {"N", "X"}:
        mask, bass = np.zeros(12), -1
        surrogate = "N"
    else:
        root, mask, bass = mir_eval.chord.encode(normalized)
        surrogate = studio.note_name(root, "sharps") + ("m" if mask[3] and not mask[4] else "")
    bass_vector = np.zeros(13)
    bass_vector[int(bass) if bass >= 0 else 12] = 1
    return np.asarray(mask), bass_vector, surrogate


def automatic_candidate_features(raw, external, evidence):
    """No reference argument; candidate-label pitch masks and NNLS rotations."""
    groups, rows, auxiliary = [], [], []
    for event in raw:
        candidates = candidate_pool(event, external)
        labels = [harte_chord(candidate["label"]) for candidate in candidates]
        full_support = np.asarray([candidate["source_support"] for candidate in candidates])
        coverage = [1.]
        for source in external:
            coverage.append(sum(max(0., min(event.end, r.end) - max(event.start, r.start)) for r in source) / (event.end - event.start))
        start = len(rows)
        for candidate, label in zip(candidates, labels):
            root_matches = mir_eval.chord.root(labels, [label] * len(labels)) == 1
            triad_matches = mir_eval.chord.triads(labels, [label] * len(labels)) == 1
            support = np.column_stack([full_support[root_matches].sum(axis=0),
                                       full_support[triad_matches].sum(axis=0), candidate["source_support"]]).reshape(-1)
            mask, bass, surrogate = _candidate_components(candidate["label"])
            rows.append(TimedChord(event.start, event.end, surrogate))
            auxiliary.append(np.concatenate([support, coverage,
                [np.log1p(event.end - event.start), float(label == "N")], mask, bass]))
        groups.append(dict(start=start, end=len(rows), candidates=candidates))
    windows = acoustic_windows(rows, evidence.metadata["duration_seconds"])
    rotated = notes.features_for_intervals(evidence, windows)
    features = np.concatenate([rotated, np.asarray(auxiliary).reshape(-1, 39)], axis=1).astype(np.float32)
    if features.shape != (len(rows), DIMENSION) or not np.isfinite(features).all():
        raise ValueError("Candidate features must be finite 136-dimensional rows")
    return features, groups


def reference_utilities(raw, groups, reference):
    """Soft supervised utility for candidate rows; called on B52 when fitting."""
    targets = np.full(groups[-1]["end"] if groups else 0, np.nan)
    weights = np.zeros(len(targets))
    comparators = (mir_eval.chord.root, mir_eval.chord.triads, mir_eval.chord.tetrads)
    total_known = sum(row.end - row.start for row in reference if harte_chord(row.chord) != "X")
    for event, group in zip(raw, groups):
        labels, durations = [], []
        for ref in reference:
            overlap = min(event.end, ref.end) - max(event.start, ref.start)
            label = harte_chord(ref.chord)
            if overlap > 0 and label != "X":
                labels.append(label)
                durations.append(overlap)
        if not labels:
            continue
        durations = np.asarray(durations)
        known = float(durations.sum())
        for index, candidate in enumerate(group["candidates"], group["start"]):
            predicted = [harte_chord(candidate["label"])] * len(labels)
            utilities = [float(durations @ comparator(labels, predicted) / known) for comparator in comparators]
            targets[index] = float(np.dot(UTILITY_WEIGHTS, utilities))
            weights[index] = known / max(total_known, 1e-12) / len(group["candidates"])
    if np.any(targets[np.isfinite(targets)] < -1e-8) or np.any(targets[np.isfinite(targets)] > 1 + 1e-8):
        raise ValueError("Unsupported reference comparison produced invalid target utility")
    return targets, weights


def training_arrays(songs):
    x, y, weights = [], [], []
    for song in songs:
        target, weight = reference_utilities(song["raw"], song["groups"], song["reference"])
        keep = np.isfinite(target) & (weight > 0)
        x.append(song["features"][keep])
        y.append(target[keep])
        weights.append(weight[keep])
    x, y, weights = np.concatenate(x), np.concatenate(y), np.concatenate(weights)
    weights /= weights.mean()
    return x, y, weights


def fit_model(name, x, y, weights):
    if name == "hgb_regression":
        return HistGradientBoostingRegressor(**HGB_PARAMETERS).fit(x, y, sample_weight=weights)
    if name == "ridge_regression":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        return model.fit(x, y, ridge__sample_weight=weights)
    raise ValueError(name)


def predict_song(song, method, model=None, margin=0.):
    scores = None if model is None else np.clip(model.predict(song["features"]), 0., 1.)
    output = []
    for event, group in zip(song["raw"], song["groups"]):
        if method == "identity":
            chosen = 0
        elif method in {"full_majority", "triad_majority"}:
            chosen = fixed_vote(group["candidates"], method == "triad_majority")
        else:
            local = scores[group["start"]:group["end"]]
            chosen = int(np.argmax(local))
            if local[chosen] <= local[0] + margin + 1e-12:
                chosen = 0
        output.append(TimedChord(event.start, event.end, group["candidates"][chosen]["label"]))
    return output


def score(songs, method, model=None, margin=0., output=None):
    results = []
    for song in songs:
        prediction = predict_song(song, method, model, margin)
        if [(r.start, r.end) for r in prediction] != [(r.start, r.end) for r in song["raw"]]:
            raise AssertionError("Candidate reliability changed raw timing")
        metrics = evaluate(song["reference"], prediction)
        results.append(dict(id=song["id"], metrics=metrics,
            changed=sum(chord_identity(a.chord) != chord_identity(b.chord) for a, b in zip(song["raw"], prediction))))
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            output_path_for_row(output, song["id"]).write_text("".join(f"{r.start:.9f} {r.end:.9f} {r.chord}\n" for r in prediction), encoding="utf-8")
    summary = {metric: float(np.mean([r["metrics"][metric] for r in results])) for metric in
        ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "root_wcsr", "triad_wcsr", "tetrad_wcsr")}
    summary["changed_intervals"] = sum(r["changed"] for r in results)
    return dict(summary=summary, songs=results)


def ordering(candidate):
    values = candidate["summary"]
    return (values["triad_accuracy"], values["root_accuracy"], values["tetrad_accuracy"], -values["changed_intervals"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-protocol", type=Path, required=True)
    parser.add_argument("--madmom-labs", type=Path, required=True)
    parser.add_argument("--chordino-labs", type=Path, required=True)
    parser.add_argument("--largevocab-labs", type=Path, required=True)
    parser.add_argument("--note-cache", type=Path, default=Path(".tmp/nnls_seventh_features/notes"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise ValueError("Do not overwrite a predeclared study")
    started = time.perf_counter()
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    dev_ids = set(split["split_ids"])
    ids = {row["id"] for row in manifest}
    if (len(manifest) != 66 or len(ids) != 66 or len(dev_ids) != 14 or not dev_ids <= ids
            or any(row.get("split") != "train" for row in manifest)):
        raise ValueError("Only frozen B52/dev14 inside train66 are allowed")
    protocol = dict(protocol="train_only_multisource_candidate_soft_utility", status="predeclared_before_fit_or_new_scores",
        feature_schema=SCHEMA, feature_dimension=DIMENSION,
        feature_order=["nnls_candidate_relative97", "per_source_root_triad_full_support9", "source_coverage3",
                       "logduration_and_isN2", "relative_pitchmask12", "bass_relative_onehot13"],
        utility_weights=dict(root=.25, triad=.50, tetrad=.25), include_reference_N=True,
        sample_weight="Known duration/song known duration/candidate count; normalize mean weight to1",
        models=MODELS, candidate_margins=MARGINS, additional_candidates=["identity", "full_majority", "triad_majority"],
        score_interpretation="Predicted duration utility, clipped0..1; not a calibrated chord probability",
        selection_rule="Dev root,triad,tetrad chord-only each must not drop below raw. Lexicographically maximize triad, root, tetrad; then fewer changes. Exact ties retain earlier fixed candidate.",
        training_ids=[row["id"] for row in manifest if row["id"] not in dev_ids],
        development_ids=[row["id"] for row in manifest if row["id"] in dev_ids],
        manifest=signature(args.manifest), split=signature(args.split_protocol),
        implementation=[signature(__file__), signature(notes.__file__), signature(Path(__file__).with_name("analyze_multisource_candidate_potential.py"))],
        heldout_labels_opened=False, sonic_inference_forbidden=True,
        cache_limitation="Legacy external LAB original source/model/config identity unverified; current hashes and manifest association only",
        exploration_history="Candidate oracle potential and two fixed majority rules on B52/dev14 inspected before declaring these models; neither model uses oracle choices as features")
    write_json(args.output / "protocol.json", protocol)
    report = dict(protocol=protocol, status="preparing_train66_cached_candidates", inputs=[], candidates=[])
    songs = []
    for index, row in enumerate(manifest):
        audio = (args.manifest.parent / row["audio"]).resolve()
        paths = [output_path_for_row(directory, row["id"]) for directory in
                 (args.madmom_labs, args.chordino_labs, args.largevocab_labs)]
        labs = [validate_timeline(read_reference(path), str(path)) if path.is_file() else [] for path in paths]
        if not labs[0]:
            raise ValueError("Raw LAB required")
        evidence, note_identity = load_existing_evidence(audio, args.note_cache)
        features, groups = automatic_candidate_features(labs[0], labs[1:], evidence)
        reference = validate_timeline(read_reference(args.manifest.parent / row["chord_reference"]), "Reference")
        songs.append(dict(id=row["id"], audio=str(audio), raw=labs[0], groups=groups, features=features, reference=reference))
        report["inputs"].append(dict(id=row["id"], audio=signature(audio), notes=note_identity,
            labs=[signature(path) if path.is_file() else dict(path=str(path.resolve()), missing=True) for path in paths]))
        print(f"Candidate features {index+1}/66: {row['id']}", flush=True)
    fit = [song for song in songs if song["id"] not in dev_ids]
    dev = [song for song in songs if song["id"] in dev_ids]
    with threadpool_limits(limits=2):
        x, y, weights = training_arrays(fit)
        report["fit_shape"] = list(x.shape)
        report["fit_target_quantiles"] = np.quantile(y, [0, .25, .5, .75, 1]).tolist()
        winner, winning_model = None, None
        baseline = score(dev, "identity")
        report["identity"] = baseline

        def consider(method, model=None, margin=None):
            nonlocal winner, winning_model
            result = score(dev, method, model, margin or 0.)
            summary = result["summary"]
            eligible = all(summary[metric] >= baseline["summary"][metric] - 1e-12 for metric in
                           ("root_accuracy", "triad_accuracy", "tetrad_accuracy"))
            candidate = dict(method=method, margin=margin, eligible=eligible, summary=summary)
            report["candidates"].append(candidate)
            if eligible and (winner is None or ordering(candidate) > ordering(winner)):
                winner, winning_model = candidate, model
            return candidate

        for fixed in protocol["additional_candidates"]:
            consider(fixed)
        for name in MODELS:
            model = fit_model(name, x, y, weights)
            for margin in MARGINS:
                consider(name, model, margin)
            print(f"Validated {name}; best={winner}", flush=True)
            report["status"] = "development_selection"
            write_json(args.output / "report.json", report)
        report["selected"] = winner
        write_json(args.output / "selection_frozen.json", dict(selected=winner, protocol=signature(args.output / "protocol.json")))
        report["status"] = "selection_frozen_before_refit"
        artifact = dict(schema=SCHEMA, feature_dimension=DIMENSION, method=winner["method"], margin=winner["margin"],
            model=winning_model, training_song_ids=[song["id"] for song in fit] if winning_model is not None else [],
            training_audio_paths=[song["audio"] for song in fit] if winning_model is not None else [],
            protocol=protocol)
        selected_path = args.output / "selected_development_head.joblib"
        joblib.dump(artifact, selected_path)
        report["development_model"] = signature(selected_path)
        report["selected_development"] = score(dev, winner["method"], winning_model, winner["margin"] or 0., args.output / "development_predictions")
        if winning_model is not None:
            x_all, y_all, weight_all = training_arrays(songs)
            refit = {**artifact, "model": fit_model(winner["method"], x_all, y_all, weight_all),
                "training_song_ids": [song["id"] for song in songs], "training_audio_paths": [song["audio"] for song in songs],
                "refit_note": "Model refit on B66 after freezing development choice; no further development scoring or external evaluation"}
            path = args.output / "selected_train66_head.joblib"
            joblib.dump(refit, path)
            report["refit_model"] = signature(path)
            report["refit_shape"] = list(x_all.shape)
        report.update(status="complete_no_external_evaluation", elapsed_seconds=time.perf_counter()-started,
                      original_timing_preserved=True, default_changed=False)
        write_json(args.output / "report.json", report)
    print(json.dumps(dict(selected=winner, elapsed_seconds=report["elapsed_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
