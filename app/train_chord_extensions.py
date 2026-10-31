"""Fit a fixed-root seventh head on train songs; evaluate separate manifests.

Uses cached upstream acoustic chord intervals, never reference beat grids or
keys. The upstream model's own pretraining overlap is unknown.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

import chord_studio as studio
from chord_extension_model import (FEATURE_SCHEMA, FEATURE_DIMENSIONS, chord_attributes,
                                   extend_rows, features_for_rows, normalize_feature_schema)
from evaluate_chords import evaluate, read_reference, validate_timeline


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_implementation_sha256():
    # Include transitive local decoding, chroma, candidate-vector and label
    # parsing implementations. Whole-file hashing intentionally invalidates
    # conservatively if another part of a shared implementation changes.
    filenames = ("chord_extension_model.py", "acoustic_chord_model.py",
                 "chord_studio.py", "chord_recognizer.py", "audio_features.py",
                 "evaluate_chords.py")
    hashes = {name: _sha256_file(Path(__file__).with_name(name)) for name in filenames}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _validate_features(features, rows, schema, source):
    expected = (rows, FEATURE_DIMENSIONS[schema])
    if (features.shape != expected or not np.issubdtype(features.dtype, np.number)
            or np.iscomplexobj(features) or not np.all(np.isfinite(features))):
        raise ValueError(f"Invalid extension features in {source}: expected finite real array {expected}")


def read_songs(manifest, lab_dir, cache_dir, feature_schema=FEATURE_SCHEMA):
    feature_schema = normalize_feature_schema(feature_schema)
    implementation_sha256 = _feature_implementation_sha256()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    resolved_paths = [(manifest.parent / row["audio"]).resolve() for row in rows]
    paths = [str(path).casefold() for path in resolved_paths]
    if len(set(paths)) != len(paths) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate songs/audio paths must be grouped before splitting")
    cache_dir.mkdir(parents=True, exist_ok=True)
    songs = []
    for row, audio_path, audio in zip(rows, paths, resolved_paths):
        lab = lab_dir / (row["id"] + ".lab")
        upstream = validate_timeline(read_reference(lab), "Upstream")
        reference = validate_timeline(read_reference(manifest.parent / row["chord_reference"]), "Reference")
        key = json.dumps(dict(cache_version=2, feature_schema=feature_schema,
                              feature_implementation_sha256=implementation_sha256,
                              audio_path=audio_path, audio_size=audio.stat().st_size,
                              audio_mtime_ns=audio.stat().st_mtime_ns,
                              audio_sha256=_sha256_file(audio), lab_sha256=_sha256_file(lab)), sort_keys=True)
        cache = cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                features = saved["features"]
            _validate_features(features, len(upstream), feature_schema, cache)
        else:
            features, _ = features_for_rows(studio.decode_audio(audio), upstream, feature_schema=feature_schema)
            _validate_features(features, len(upstream), feature_schema, "fresh extraction")
            np.savez_compressed(cache, features=features, cache_identity=np.asarray(key))
        targets = np.full(len(upstream), -1, dtype=np.int8)
        weights = np.zeros((len(upstream), 3))
        for i, event in enumerate(upstream):
            base = chord_attributes(event.chord)
            if not base:
                continue
            for ref in reference:
                overlap = max(0, min(event.end, ref.end) - max(event.start, ref.start))
                truth = chord_attributes(ref.chord) if overlap > 0 else None
                if truth and truth[:2] == base[:2]:
                    weights[i, truth[2]] += overlap
            best = int(np.argmax(weights[i]))
            if weights[i, best] >= .8 * (event.end - event.start):
                targets[i] = best
        songs.append(dict(id=row["id"], audio=audio_path, source_audio=str(audio), split=row.get("split"), upstream=upstream,
                          reference=reference, features=features, targets=targets))
        print(f"Features {len(songs)}/{len(rows)}: {row['id']}", flush=True)
    return songs


def arrays(songs):
    x, y = [], []
    for song in songs:
        keep = song["targets"] >= 0
        x.append(song["features"][keep])
        y.append(song["targets"][keep])
    return np.concatenate(x), np.concatenate(y)


def score_songs(songs, artifact, threshold=None, output=None):
    result = []
    for song in songs:
        prediction, evidence = extend_rows(song["upstream"], song["features"], artifact, threshold)
        base = evaluate(song["reference"], song["upstream"])
        refined = evaluate(song["reference"], prediction)
        if abs(base["root_accuracy"] - refined["root_accuracy"]) > 1e-10 or abs(base["triad_accuracy"] - refined["triad_accuracy"]) > 1e-10:
            raise AssertionError("Seventh head changed root or triad accuracy")
        result.append(dict(id=song["id"], baseline=base, extension=refined,
                           changed=sum(a.chord != b.chord for a, b in zip(song["upstream"], prediction))))
        if output:
            output.mkdir(parents=True, exist_ok=True)
            (output / (song["id"] + ".lab")).write_text("".join(f"{r.start:.6f} {r.end:.6f} {r.chord}\n" for r in prediction), encoding="utf-8")
            (output / (song["id"] + "_evidence.json")).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    summary = {}
    for mode in ("baseline", "extension"):
        for metric in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "triad_wcsr", "tetrad_wcsr"):
            summary[f"{mode}_macro_{metric}"] = float(np.mean([r[mode][metric] for r in result]))
    return dict(summary=summary, songs=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("lab_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Evaluate this frozen model; otherwise train on train-only manifest")
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/extension_features"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if not args.model and any(row.get("split") != "train" for row in rows):
        parser.error("Only train-labeled songs may fit or select extension thresholds")
    started = time.perf_counter()
    artifact = joblib.load(args.model) if args.model else None
    songs = read_songs(args.manifest.resolve(), args.lab_dir.resolve(), args.cache_dir,
                       feature_schema=artifact["feature_schema"] if artifact else FEATURE_SCHEMA)
    with threadpool_limits(limits=args.threads):
        if args.model:
            for song in songs:
                if song["id"] in artifact["training_song_ids"] or song["audio"] in artifact["training_audio_paths"]:
                    raise ValueError("A training song/audio cannot be a held-out evaluation")
            report = score_songs(songs, artifact, output=args.output / "predictions")
            args.output.mkdir(parents=True, exist_ok=True)
            output = args.output / "report.json"
        else:
            train, dev = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260926).split(
                np.arange(len(songs)), groups=[song["id"] for song in songs]))
            params = dict(max_iter=160, max_leaf_nodes=15, learning_rate=.06, min_samples_leaf=30,
                          l2_regularization=3., max_bins=63, early_stopping=False, random_state=20260926)
            x, y = arrays([songs[i] for i in train])
            artifact = dict(feature_schema=FEATURE_SCHEMA, threshold=1.,
                            model=HistGradientBoostingClassifier(**params).fit(x, y))
            # Preference for the conservative setting on ties; evaluation songs
            # never enter this threshold selection.
            thresholds = [1., .9, .8, .7, .6, .5, .4]
            scores = {str(t): score_songs([songs[i] for i in dev], artifact, t)["summary"]["extension_macro_tetrad_wcsr"] for t in thresholds}
            threshold = float(max(scores, key=scores.get))
            x, y = arrays(songs)
            artifact.update(model=HistGradientBoostingClassifier(**params).fit(x, y), threshold=threshold,
                            training_song_ids=[s["id"] for s in songs], training_audio_paths=[s["audio"] for s in songs])
            report = dict(validation_macro_tetrad_wcsr_by_threshold=scores, selected_threshold=threshold,
                          validation_song_ids=[songs[i]["id"] for i in dev], parameters=params,
                          training_song_ids=[s["id"] for s in songs],
                          sample_weighting="One sample per eligible upstream interval; no duration/song/class reweighting",
                          training_class_counts={str(k): int(v) for k,v in zip(*np.unique(y, return_counts=True))})
            artifact["training_report"] = report
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.with_suffix(".json")
    report.update(protocol="upstream_acoustic_segments_fixed_root_and_triad_seventh_head",
                  limitations=["Upstream pretraining overlap unknown",
                               "Training-domain coverage is defined by the artifact training report",
                               "Upstream boundaries/roots are frozen", "Scores uncalibrated"],
                  feature_schema=artifact["feature_schema"],
                  manifest_path=str(args.manifest.resolve()), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-started)
    if args.model:
        report["model"] = {"path": str(args.model.resolve()),
                           "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                           "frozen_threshold": artifact["threshold"]}
    if not args.model:
        artifact["manifest_sha256"] = report["manifest_sha256"]
        artifact["training_report"] = report
        joblib.dump(artifact, args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.get("summary", {k:v for k,v in report.items() if k != "songs"}), indent=2), flush=True)


if __name__ == "__main__":
    main()
