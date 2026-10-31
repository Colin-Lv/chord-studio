"""Further development-only consensus experiment after the two-domain study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

import chord_extension_model as extension
from seventh_consensus import ConsensusSeventhModel
from train_chord_extensions import arrays, read_songs, score_songs
from train_multidomain_extensions import add_features, assert_disjoint, signature, target_coverage, write_report


GROUPS = {"uniform_two_views": ["v1_uniform", "v2_uniform"],
          "balanced_two_views": ["v1_balanced", "v2_balanced"],
          "all_four": ["v1_uniform", "v1_balanced", "v2_uniform", "v2_balanced"]}


def fit_heads(songs, indices, parameters):
    heads = {}
    for schema, dimensions in (("v1", 87), ("v2", 148)):
        x, y = arrays([songs[i] for i in indices])
        for weighting in ("uniform", "balanced"):
            weights = None if weighting == "uniform" else compute_sample_weight("balanced", y)
            heads[f"{schema}_{weighting}"] = HistGradientBoostingClassifier(**parameters).fit(x[:, :dimensions], y, sample_weight=weights)
    return heads


def make_model(heads, group):
    names = GROUPS[group]
    return ConsensusSeventhModel([heads[name] for name in names], [87 if name.startswith("v1_") else 148 for name in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-study", type=Path, required=True)
    parser.add_argument("--beatles-labs", type=Path, required=True)
    parser.add_argument("--guitar-labs", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/extension_features"))
    args = parser.parse_args()
    previous = json.loads(args.previous_study.read_text(encoding="utf-8"))
    manifests = {key: Path(value["path"]) for key, value in previous["input_manifests"].items()}
    if any(signature(path)["sha256"] != previous["input_manifests"][name]["sha256"] for name, path in manifests.items()):
        raise ValueError("A frozen input manifest changed")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "training_report.json"
    if path.exists():
        raise ValueError("Use a new output directory")
    started = time.perf_counter()
    report = dict(protocol="further_development_only_agreement_heads", status="predeclared_before_consensus_scores",
        previous_study=signature(args.previous_study), feature_schema=extension.FEATURE_SCHEMA_V2,
        groups=GROUPS, thresholds=[1., .99, .97, .95, .9, .8, .7, .6, .5], parameters=previous["parameters"],
        selection_rule=previous["selection_rule"], input_manifests=previous["input_manifests"],
        exploratory_history="First study selected disabled identity; consensus families were proposed after inspecting only its development scores. No held-out model scores inspected for this choice.",
        labels="GuitarSet performed supervision is weak and score-informed; not independently annotated chord gold",
        validation_refit_forbidden=True, held_out_results_used_for_selection=False,
        implementation=[signature(__file__), signature(extension.__file__), signature(Path(__file__).with_name("seventh_consensus.py"))], candidates=[])
    write_report(path, report)
    b = read_songs(manifests["beatles_train"], args.beatles_labs.resolve(), args.cache_dir)
    gt = read_songs(manifests["guitar_train"], args.guitar_labs.resolve(), args.cache_dir)
    gd = read_songs(manifests["guitar_validation"], args.guitar_labs.resolve(), args.cache_dir)
    assert_disjoint(b, gt, gd)
    if any(s["split"] != "train" for s in b) or any(s["split"] not in {"train", "train40"} for s in gt) or any(s["split"] not in {"val", "val10", "validation"} for s in gd):
        raise ValueError("Invalid training/development role")
    training, development = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260926).split(np.arange(len(b)), groups=[s["id"] for s in b]))
    if [b[i]["id"] for i in development] != previous["development_song_ids"]["beatles"]:
        raise ValueError("The fixed Beatles development split changed")
    combined = add_features(b + gt + gd, extension.FEATURE_SCHEMA_V2, args.cache_dir / "temporal_v2")
    train_indices = list(training) + list(range(len(b), len(b) + len(gt)))
    validation = dict(beatles=[combined[i] for i in development], guitar=combined[len(b) + len(gt):])
    report["selection_training_song_ids"] = [combined[i]["id"] for i in train_indices]
    report["development_song_ids"] = {key: [s["id"] for s in songs] for key, songs in validation.items()}
    report["target_coverage"] = {"selection_training": target_coverage([combined[i] for i in train_indices]),
                               **{key: target_coverage(songs) for key, songs in validation.items()}}
    winner = None
    with threadpool_limits(limits=args.threads):
        # Refit selection heads only on B52+G40, never the previous final artifact.
        heads = fit_heads(combined, train_indices, report["parameters"])
        for group in GROUPS:
            artifact = dict(feature_schema=extension.FEATURE_SCHEMA_V2, threshold=1., model=make_model(heads, group))
            for threshold in report["thresholds"]:
                domains = {name: score_songs(songs, artifact, threshold)["summary"] for name, songs in validation.items()}
                eligible = all(v["extension_macro_tetrad_wcsr"] >= v["baseline_macro_tetrad_wcsr"] - 1e-12 for v in domains.values())
                objective = float(np.mean([v["extension_macro_tetrad_wcsr"] for v in domains.values()]))
                candidate = dict(group=group, threshold=threshold, domains=domains, objective=objective, eligible=eligible)
                report["candidates"].append(candidate)
                if eligible and (winner is None or objective > winner["objective"] + 1e-12):
                    winner = candidate
            report["status"] = "development_selection"
            write_report(path, report)
            print(f"Consensus {group}: best={winner['objective']:.6f} threshold={winner['threshold']}", flush=True)
        report.update(status="selection_frozen", selected=winner)
        write_report(path, report)
        heads = fit_heads(combined, list(range(len(combined))), report["parameters"])
        artifact = dict(feature_schema=extension.FEATURE_SCHEMA_V2, threshold=winner["threshold"], model=make_model(heads, winner["group"]),
            training_song_ids=[s["id"] for s in combined], training_audio_paths=[s["audio"] for s in combined],
            training_report=report, refit_note="Refit B66+G40+Gvalidation10 only after development selection; held-outs excluded")
        model_path = args.output / "consensus_seventh.joblib"
        joblib.dump(artifact, model_path)
        report.update(status="model_frozen_before_heldout", model=signature(model_path))
        write_report(path, report)
        tests = dict(beatles=read_songs(manifests["beatles_eval"], args.beatles_labs.resolve(), args.cache_dir),
                     guitar=read_songs(manifests["guitar_test"], args.guitar_labs.resolve(), args.cache_dir))
        assert_disjoint(combined, *tests.values())
        if any(s["split"] not in {"test", "test10"} for s in tests["guitar"]):
            raise ValueError("Guitar held-outs must have the test role")
        report["held_out"] = {}
        for name, songs in tests.items():
            songs = add_features(songs, extension.FEATURE_SCHEMA_V2, args.cache_dir / "temporal_v2")
            report["held_out"][name] = score_songs(songs, artifact, output=args.output / (name + "_predictions"))
            write_report(path, report)
    report.update(status="complete", elapsed_seconds=time.perf_counter() - started, default_changed=False,
        limitations=["Further development-set exploration, not a preregistered comparison of every method", "Weak score-informed guitar labels", "Guitar performances share compositions between splits", "Only seventh quality can change, not missing thirds/fifths or roots"])
    write_report(path, report)
    print(json.dumps(dict(selected=winner, held_out={name: value["summary"] for name, value in report["held_out"].items()},
                         elapsed_seconds=report["elapsed_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
