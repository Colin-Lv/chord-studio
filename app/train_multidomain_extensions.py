"""Frozen two-domain experiment for a conditional acoustic seventh head.

GuitarSet performed chords are weak, score-informed supervision, not independent
human gold labels. Neither held-out set is used to choose features or thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

import chord_studio as studio
import chord_extension_model as extension
from train_chord_extensions import arrays, read_songs, score_songs


def signature(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def add_features(songs, schema, cache_dir):
    if schema == extension.FEATURE_SCHEMA:
        return songs
    result = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    implementation_hash = signature(extension.__file__)["sha256"]
    for index, song in enumerate(songs):
        source_audio = Path(song.get("source_audio", song["audio"]))
        source_hash = signature(source_audio)["sha256"]
        cache_key = json.dumps([schema, implementation_hash, source_hash,
            [(r.start, r.end, r.chord) for r in song["upstream"]]])
        path = cache_dir / (hashlib.sha256(cache_key.encode()).hexdigest() + ".npz")
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                features = saved["features"]
        else:
            features, _ = extension.features_for_rows(studio.decode_audio(source_audio), song["upstream"], feature_schema=schema)
            np.savez_compressed(path, features=features)
        if features.shape != (len(song["upstream"]), extension.FEATURE_DIMENSIONS[schema]) or not np.isfinite(features).all():
            raise ValueError("Invalid cached feature shape/value")
        result.append({**song, "features": features})
        print(f"Temporal features {index + 1}/{len(songs)}: {song['id']}", flush=True)
    return result


def assert_disjoint(*collections):
    ids, paths = set(), set()
    for songs in collections:
        current_ids, current_paths = {s["id"] for s in songs}, {s["audio"].casefold() for s in songs}
        if ids & current_ids or paths & current_paths:
            raise ValueError("Training/development/test song or audio identity overlap")
        ids.update(current_ids)
        paths.update(current_paths)


def write_report(path, result):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def target_coverage(songs):
    intervals = [(song, index, row) for song in songs for index, row in enumerate(song["upstream"])]
    valid = [(song, index, row) for song, index, row in intervals if song["targets"][index] >= 0]
    return dict(songs=len(songs), upstream_intervals=len(intervals), eligible_target_intervals=len(valid),
        upstream_seconds=sum(row.end - row.start for _, _, row in intervals),
        eligible_target_seconds=sum(row.end - row.start for _, _, row in valid),
        target_class_counts={str(value): int(sum(song["targets"][index] == value for song, index, _ in valid)) for value in (0, 1, 2)},
        exclusions="Intervals without >=80% support for the same reference root, major/minor and seventh class, or labels outside this fixed-root seventh task, have target=-1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beatles-train", type=Path, required=True)
    parser.add_argument("--beatles-eval", type=Path, required=True)
    parser.add_argument("--beatles-labs", type=Path, required=True)
    parser.add_argument("--guitar-train", type=Path, required=True)
    parser.add_argument("--guitar-validation", type=Path, required=True)
    parser.add_argument("--guitar-test", type=Path, required=True)
    parser.add_argument("--guitar-labs", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/extension_features"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "training_report.json"
    if report_path.exists():
        raise ValueError("Use a new output directory; do not overwrite a frozen experiment")
    started = time.perf_counter()
    schemas = [extension.FEATURE_SCHEMA, extension.FEATURE_SCHEMA_V2]
    thresholds = [1., .9, .8, .7, .6, .5, .4]
    parameters = dict(max_iter=160, max_leaf_nodes=15, learning_rate=.06, min_samples_leaf=30,
                      l2_regularization=3., max_bins=63, early_stopping=False, random_state=20260926)
    report = dict(protocol="fixed_root_triad_seventh_two_domain_development_selection", status="predeclared",
        schemas=schemas, thresholds=thresholds, weight_modes=["uniform", "balanced"], parameters=parameters,
        selection_rule="Both Beatles dev14 and GuitarSet validation10 macro tetrad WCSR must not fall below identity; maximize equal-domain mean; identity wins ties",
        guitar_label_status="Performed labels: semi-automatic, manually verified, with score-informed roots/segmentation; not independent human chord gold",
        input_manifests={name: signature(getattr(args, name)) for name in
            ("beatles_train", "beatles_eval", "guitar_train", "guitar_validation", "guitar_test")},
        implementation=[signature(__file__), signature(extension.__file__)],
        held_out_results_used_for_selection=False, candidates=[])
    write_report(report_path, report)
    beatles = read_songs(args.beatles_train.resolve(), args.beatles_labs.resolve(), args.cache_dir)
    guitar_train = read_songs(args.guitar_train.resolve(), args.guitar_labs.resolve(), args.cache_dir)
    guitar_dev = read_songs(args.guitar_validation.resolve(), args.guitar_labs.resolve(), args.cache_dir)
    if any(song["split"] != "train" for song in beatles) or any(song["split"] not in {"train", "train40"} for song in guitar_train):
        raise ValueError("Training rows must be explicitly labeled train")
    if any(song["split"] not in {"val", "val10", "validation", "dev"} for song in guitar_dev):
        raise ValueError("Guitar development rows must be explicitly labeled validation")
    assert_disjoint(beatles, guitar_train, guitar_dev)
    report["target_coverage"] = {name: target_coverage(songs) for name, songs in
        (("beatles_train66", beatles), ("guitar_train40", guitar_train), ("guitar_validation10", guitar_dev))}
    training, development = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260926).split(
        np.arange(len(beatles)), groups=[song["id"] for song in beatles]))
    report["development_song_ids"] = dict(beatles=[beatles[i]["id"] for i in development], guitar=[s["id"] for s in guitar_dev])
    report["selection_training_song_ids"] = [beatles[i]["id"] for i in training] + [s["id"] for s in guitar_train]
    feature_sets = {}
    winner = None
    with threadpool_limits(limits=args.threads):
        for schema in schemas:
            combined = add_features(beatles + guitar_train + guitar_dev, schema, args.cache_dir / "temporal_v2")
            b, gt, gd = combined[:len(beatles)], combined[len(beatles):len(beatles) + len(guitar_train)], combined[len(beatles) + len(guitar_train):]
            feature_sets[schema] = combined
            fit = [b[i] for i in training] + gt
            validation_domains = dict(beatles=[b[i] for i in development], guitar=gd)
            x, y = arrays(fit)
            for weighting in report["weight_modes"]:
                weights = None if weighting == "uniform" else compute_sample_weight("balanced", y)
                model = HistGradientBoostingClassifier(**parameters).fit(x, y, sample_weight=weights)
                artifact = dict(feature_schema=schema, threshold=1., model=model)
                for threshold in thresholds:
                    domains = {domain: score_songs(songs, artifact, threshold)["summary"] for domain, songs in validation_domains.items()}
                    eligible = all(values["extension_macro_tetrad_wcsr"] >= values["baseline_macro_tetrad_wcsr"] - 1e-12 for values in domains.values())
                    objective = float(np.mean([values["extension_macro_tetrad_wcsr"] for values in domains.values()]))
                    candidate = dict(schema=schema, weighting=weighting, threshold=threshold, eligible=eligible, objective=objective, domains=domains)
                    report["candidates"].append(candidate)
                    if eligible and (winner is None or objective > winner["objective"] + 1e-12):
                        winner = candidate
                print(f"Validated {schema}, {weighting}; best={winner['objective']:.6f}, threshold={winner['threshold']}", flush=True)
                report["status"] = "selecting_on_development_only"
                write_report(report_path, report)
        report["selected"] = winner
        report["status"] = "selection_frozen_before_heldout_scoring"
        write_report(report_path, report)
        selected_songs = feature_sets[winner["schema"]]
        x, y = arrays(selected_songs)
        weights = None if winner["weighting"] == "uniform" else compute_sample_weight("balanced", y)
        artifact = dict(feature_schema=winner["schema"], threshold=winner["threshold"],
            model=HistGradientBoostingClassifier(**parameters).fit(x, y, sample_weight=weights),
            training_song_ids=[s["id"] for s in selected_songs], training_audio_paths=[s["audio"] for s in selected_songs],
            training_report=report,
            refit_note="After freezing selection, refit on all Beatles train66 plus GuitarSet train40 and validation10; held-out sets remain excluded")
        report["refit_class_counts"] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        model_path = args.output / "conditional_seventh.joblib"
        joblib.dump(artifact, model_path)
        report["model"] = signature(model_path)
        report["status"] = "model_frozen_before_heldout"
        write_report(report_path, report)
        # Only after freezing the rule/model do we open held-out chord labels.
        test_collections = dict(
            beatles=read_songs(args.beatles_eval.resolve(), args.beatles_labs.resolve(), args.cache_dir),
            guitar=read_songs(args.guitar_test.resolve(), args.guitar_labs.resolve(), args.cache_dir))
        assert_disjoint(selected_songs, *test_collections.values())
        report["held_out"] = {}
        for domain, songs in test_collections.items():
            songs = add_features(songs, winner["schema"], args.cache_dir / "temporal_v2")
            report["held_out"][domain] = score_songs(songs, artifact, output=args.output / (domain + "_predictions"))
            write_report(report_path, report)
    report.update(status="complete", elapsed_seconds=time.perf_counter() - started,
        default_changed=False,
        limitations=["GuitarSet weak performed supervision", "Upstream pretraining overlap unknown", "Small fixed development/test sets", "Major/minor roots and boundaries are frozen", "Scores are uncalibrated"])
    write_report(report_path, report)
    print(json.dumps(dict(selected=winner, refit_class_counts=report["refit_class_counts"],
        held_out={name: value["summary"] for name, value in report["held_out"].items()}, elapsed_seconds=report["elapsed_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
