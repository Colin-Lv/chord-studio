"""Development-only fixed-root seventh gate from cached external LAB support.

Fit Beatles52; choose among two predeclared heads on Beatles14. No held-out
manifest argument exists. Refit all train66 only after writing frozen choice.
Legacy external source identities are explicitly unverified.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import chord_extension_model as extension
import chord_studio as studio
from evaluate_chords import (TimedChord, convert_annotation_chord, evaluate,
                            harte_chord, read_reference, validate_timeline)
from generate_madmom_chords_from_manifest import output_path_for_row
from train_chord_extensions import read_songs, arrays

SCHEMA = "v2_acoustic_and_two_external_overlap_seventh_v1"
CATEGORIES = ("same_triad_no7", "same_triad_minor7", "same_triad_major7",
              "other_root", "other_triad", "explicit_N", "unknown", "missing")
THRESHOLDS = [1., .9, .8, .7, .6, .5, .4]
HGB_PARAMETERS = dict(max_iter=160, max_leaf_nodes=15, learning_rate=.06,
    min_samples_leaf=30, l2_regularization=3., max_bins=63,
    early_stopping=False, random_state=20260926)
LOGISTIC_PARAMETERS = dict(C=1., max_iter=1000, solver="lbfgs", random_state=20260926)
FEATURE_DIMENSION = 166


def signature(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _category(base, label):
    try:
        normalized = harte_chord(label)
        if normalized == "N":
            return 5
        if normalized == "X":
            return 6
        root, _, _ = mir_eval.chord.encode(normalized)
        if root != base[0]:
            return 3
        attributes = extension.chord_attributes(label)
        if attributes is None or attributes[1] != base[1]:
            return 4
        return attributes[2]
    except (ValueError, mir_eval.chord.InvalidChordException):
        return 6


def overlap_features(upstream, external):
    """Eight disjoint duration fractions; absent prediction never becomes N."""
    external = validate_timeline(external, "External cached LAB")
    output = np.zeros((len(upstream), 8), dtype=np.float32)
    pointer = 0
    for i, event in enumerate(upstream):
        base = extension.chord_attributes(event.chord)
        duration = event.end - event.start
        if duration <= 0:
            raise ValueError("Upstream intervals must have positive duration")
        while pointer < len(external) and external[pointer].end <= event.start:
            pointer += 1
        index = pointer
        while index < len(external) and external[index].start < event.end:
            row = external[index]
            overlap = max(0., min(event.end, row.end) - max(event.start, row.start))
            output[i, _category(base, row.chord) if base else 6] += overlap / duration
            index += 1
        output[i, 7] = max(0., 1. - float(output[i, :7].sum()))
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1., atol=2e-6):
        raise ValueError("External overlap fractions must form a finite partition")
    return output


def fuse_features(song, external_labs):
    extra = [overlap_features(song["upstream"], rows) for rows in external_labs]
    properties = np.asarray([[np.log1p(row.end - row.start),
        (extension.chord_attributes(row.chord) or (0, 0, 0))[1]] for row in song["upstream"]], dtype=np.float32).reshape(-1, 2)
    features = np.concatenate([song["features"], *extra, properties], axis=1)
    if features.shape != (len(song["upstream"]), FEATURE_DIMENSION) or not np.isfinite(features).all():
        raise ValueError("Expected finite 166-dimensional fusion features")
    return features


def gate_rows(rows, features, artifact, threshold=None):
    threshold = artifact["threshold"] if threshold is None else float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Gate threshold must lie within [0,1]")
    features = np.asarray(features)
    if features.shape != (len(rows), FEATURE_DIMENSION) or not np.isfinite(features).all():
        raise ValueError("Invalid gate feature shape/value")
    if threshold >= 1 or not rows:
        return list(rows)
    probabilities = artifact["model"].predict_proba(features)
    classes = artifact["model"].classes_
    output = []
    for row, probability in zip(rows, probabilities):
        best = int(np.argmax(probability))
        seventh = int(classes[best])
        chord = row.chord
        base = extension.chord_attributes(chord)
        if base and base[2] == 0 and seventh in (1, 2) and probability[best] >= threshold:
            if harte_chord(chord).split("/", 1)[0].endswith((":maj", ":min")):
                suffix = {(0, 1): "7", (0, 2): "maj7", (1, 1): "m7", (1, 2): "mMaj7"}[base[1], seventh]
                chord = studio.note_name(base[0], "sharps") + suffix
                if "/" in row.chord:
                    chord += "/" + convert_annotation_chord(row.chord).split("/", 1)[1]
        output.append(TimedChord(row.start, row.end, chord))
    return output


def score_development(songs, artifact, threshold, output=None):
    results = []
    for song in songs:
        prediction = gate_rows(song["upstream"], song["features"], artifact, threshold)
        baseline = evaluate(song["reference"], song["upstream"])
        score = evaluate(song["reference"], prediction)
        if any(abs(score[metric] - baseline[metric]) > 1e-12 for metric in
               ("root_accuracy", "triad_accuracy", "root_wcsr", "triad_wcsr")):
            raise AssertionError("Seventh gate altered root/triad score")
        if [(r.start, r.end) for r in prediction] != [(r.start, r.end) for r in song["upstream"]]:
            raise AssertionError("Seventh gate altered original timing")
        results.append(dict(id=song["id"], baseline=baseline, gate=score,
            changed=sum(a.chord != b.chord for a, b in zip(song["upstream"], prediction))))
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            output_path_for_row(output, song["id"]).write_text("".join(
                f"{r.start:.9f} {r.end:.9f} {r.chord}\n" for r in prediction), encoding="utf-8")
    summary = {f"{kind}_macro_{metric}": float(np.mean([r[kind][metric] for r in results]))
        for kind in ("baseline", "gate") for metric in
        ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "tetrad_wcsr", "tetrad_inv_accuracy")}
    summary["changed_intervals"] = sum(r["changed"] for r in results)
    return dict(summary=summary, songs=results)


def make_model(name):
    if name == "hgb_uniform":
        return HistGradientBoostingClassifier(**HGB_PARAMETERS)
    if name == "logistic_uniform":
        return make_pipeline(StandardScaler(), LogisticRegression(**LOGISTIC_PARAMETERS))
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--split-protocol", type=Path, required=True)
    parser.add_argument("--madmom-labs", type=Path, required=True)
    parser.add_argument("--chordino-labs", type=Path, required=True)
    parser.add_argument("--largevocab-labs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/extension_features"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    if report_path.exists() or (args.output / "protocol.json").exists():
        raise ValueError("Use a new output directory; frozen experiments are never overwritten")
    started = time.perf_counter()
    with args.train_manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    ids = [r["id"] for r in manifest_rows]
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    dev_ids = set(split["split_ids"])
    if len(ids) != 66 or len(set(ids)) != 66 or len(dev_ids) != 14 or not dev_ids <= set(ids):
        raise ValueError("This protocol requires exactly frozen Beatles52/14 within train66")
    if any(r.get("split") != "train" for r in manifest_rows):
        raise ValueError("Only explicit train rows may enter this development experiment")
    if signature(args.train_manifest)["sha256"] != split["training_manifest"]["sha256"]:
        raise ValueError("The frozen train66 manifest changed")
    protocol = dict(protocol="cached_multisource_fixed_root_triad_seventh_development_only",
        status="predeclared_before_feature_extraction_and_scoring", feature_schema=SCHEMA,
        feature_dimension=FEATURE_DIMENSION, acoustic_schema=extension.FEATURE_SCHEMA_V2,
        external_category_order=CATEGORIES, source_order=["chordino", "ISMIR2019_largvocab"],
        trailing_features=["log1p_interval_duration", "base_is_minor"],
        source_matching="Named train66 manifest association only; legacy source audio/model/config identity is not retrospectively verified",
        models={"hgb_uniform": HGB_PARAMETERS, "logistic_uniform": {**LOGISTIC_PARAMETERS, "standard_scaler": True}},
        thresholds=THRESHOLDS, weighting="one sample per eligible raw interval, uniform",
        target_rule="Same reference root/triad/seventh class occupies >=80% of whole raw interval; otherwise target=-1",
        selection_rule="Strictly improve dev14 macro tetrad chord-only accuracy and do not lower macro tetrad WCSR versus identity. Maximize chord-only accuracy. Identity wins ties; earlier fixed model and higher threshold win remaining ties.",
        training_ids=[identifier for identifier in ids if identifier not in dev_ids],
        development_ids=[identifier for identifier in ids if identifier in dev_ids],
        train_manifest=signature(args.train_manifest), split_protocol=signature(args.split_protocol),
        implementation=[signature(__file__), signature(extension.__file__), signature(Path(__file__).with_name("train_chord_extensions.py"))],
        labels_outside_train66_opened=False, external_models_run=False, development_used_to_fit_selection_heads=False,
        refit_rule="Refit selected family on train66 only after the selection JSON is frozen; no held-out evaluation")
    write_json(args.output / "protocol.json", protocol)
    report = dict(protocol=protocol, status="extracting_train66_features", inputs=[], candidates=[])
    write_json(report_path, report)
    with threadpool_limits(limits=2):
        songs = read_songs(args.train_manifest.resolve(), args.madmom_labs.resolve(), args.cache_dir,
                           feature_schema=extension.FEATURE_SCHEMA_V2)
        for song in songs:
            external, identities = [], {}
            for source_name, directory in (("chordino", args.chordino_labs), ("ISMIR2019", args.largevocab_labs)):
                path = output_path_for_row(directory, song["id"])
                external.append(read_reference(path) if path.is_file() else [])
                identities[source_name] = signature(path) if path.is_file() else dict(path=str(path.resolve()), missing=True)
            song["features"] = fuse_features(song, external)
            report["inputs"].append(dict(id=song["id"], audio=signature(song["source_audio"]),
                raw=signature(output_path_for_row(args.madmom_labs, song["id"])), external=identities))
        fit = [song for song in songs if song["id"] not in dev_ids]
        dev = [song for song in songs if song["id"] in dev_ids]
        x, y = arrays(fit)
        report["fit_shape"] = list(x.shape)
        report["fit_class_counts"] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        identity = dict(feature_schema=SCHEMA, threshold=1., model=None)
        baseline = score_development(dev, identity, 1.)
        report["identity"] = baseline
        winner = dict(model="identity", threshold=1., objective=baseline["summary"]["gate_macro_tetrad_accuracy"])
        selected_artifact = identity
        for model_name in protocol["models"]:
            model = make_model(model_name).fit(x, y)
            artifact = dict(feature_schema=SCHEMA, feature_dimension=FEATURE_DIMENSION, threshold=1., model=model,
                training_song_ids=[song["id"] for song in fit], training_audio_paths=[song["audio"] for song in fit],
                source_order=protocol["source_order"], external_category_order=CATEGORIES)
            for threshold in THRESHOLDS:
                score = score_development(dev, artifact, threshold)
                metrics = score["summary"]
                objective = metrics["gate_macro_tetrad_accuracy"]
                eligible = (objective > baseline["summary"]["gate_macro_tetrad_accuracy"] + 1e-12
                    and metrics["gate_macro_tetrad_wcsr"] >= baseline["summary"]["gate_macro_tetrad_wcsr"] - 1e-12)
                candidate = dict(model=model_name, threshold=threshold, objective=objective, eligible=eligible, summary=metrics)
                report["candidates"].append(candidate)
                if eligible and objective > winner["objective"] + 1e-12:
                    winner = candidate
                    selected_artifact = {**artifact, "threshold": threshold}
            print(f"Developed {model_name}; selected={winner['model']} threshold={winner['threshold']} tetrad={winner['objective']:.6f}", flush=True)
            report["status"] = "development_selection"
            write_json(report_path, report)
        report["selected"] = winner
        report["status"] = "selection_frozen_before_refit"
        write_json(args.output / "selection_frozen.json", dict(selected=winner, protocol_sha256=signature(args.output / "protocol.json")["sha256"]))
        write_json(report_path, report)
        selected_artifact["training_report"] = report
        selected_path = args.output / "selected_development_head.joblib"
        joblib.dump(selected_artifact, selected_path)
        report["development_model"] = signature(selected_path)
        report["selected_development"] = score_development(dev, selected_artifact, winner["threshold"], args.output / "development_predictions")
        if winner["model"] != "identity":
            x_all, y_all = arrays(songs)
            refit = {**selected_artifact, "model": make_model(winner["model"]).fit(x_all, y_all),
                "training_song_ids": [song["id"] for song in songs],
                "training_audio_paths": [song["audio"] for song in songs],
                "refit_note": "Development choice frozen first; B66 refit includes dev14 and cannot be scored on dev14 as held-out"}
            path = args.output / "selected_train66_head.joblib"
            joblib.dump(refit, path)
            report["refit_model"] = signature(path)
            report["refit_shape"] = list(x_all.shape)
        report.update(status="complete_no_heldout_evaluation", elapsed_seconds=time.perf_counter() - started,
            root_triad_and_timing_preserved=True, default_changed=False,
            limitations=["Legacy external LAB generation identity unverified", "External pretraining overlap unknown",
                         "Beatles-only development domain", "Fixed whole raw intervals miss mixed-quality changes", "No held-out score claimed"])
        write_json(report_path, report)
    print(json.dumps(dict(selected=winner, elapsed_seconds=report["elapsed_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
