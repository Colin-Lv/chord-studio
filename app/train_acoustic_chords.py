"""Train a transposition-invariant acoustic ranker with song-level validation.

The manifest must explicitly contain train rows. Reference beat grids are used
for this controlled experiment; its score is never end-to-end performance.
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
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

import chord_studio as studio
from acoustic_chord_model import FEATURE_SCHEMA, candidate_features, decode_emissions, emission_scores
from evaluate_chords import harte_chord, read_reference, validate_timeline

TARGET_SCHEMA = "validated_harte_major_minor_v2"


def reference_rows(path):
    rows = []
    for event in validate_timeline(read_reference(path), "Training reference"):
        label = harte_chord(event.chord)
        if label in {"N", "X"}:
            continue
        root, tones, _ = mir_eval.chord.encode(label)
        if root < 0 or not tones[0] or not tones[7] or bool(tones[3]) == bool(tones[4]):
            continue
        rows.append((event.start, event.end, int(root) * 2 + int(bool(tones[3]))))
    return rows


def song_data(row, base, cache_dir, feature_mode, unit):
    audio = (base / row["audio"]).resolve()
    beats = (base / row["beat_reference"]).resolve()
    reference = (base / row["chord_reference"]).resolve()
    # File identity and annotation content prevent stale alignment-cache reuse.
    signature = json.dumps([FEATURE_SCHEMA, feature_mode, unit, str(audio), audio.stat().st_size,
                            audio.stat().st_mtime_ns, hashlib.sha256(beats.read_bytes()).hexdigest(),
                            hashlib.sha256(reference.read_bytes()).hexdigest()])
    legacy_cache = cache_dir / (hashlib.sha256(signature.encode()).hexdigest()[:24] + ".npz")
    target_signature = json.dumps([TARGET_SCHEMA, signature])
    cache = cache_dir / (hashlib.sha256(target_signature.encode()).hexdigest()[:24] + ".npz")
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    y = studio.decode_audio(audio)
    meter = studio.meter_from_downbeat_rows(studio.read_beat_grid_rows(beats), len(y) / studio.SAMPLE_RATE, None)
    if meter is None:
        raise ValueError(f"No usable beat grid for {row['id']}")
    intervals = studio.build_beat_intervals(meter, len(y) / studio.SAMPLE_RATE)
    cells = studio.harmonic_intervals_from_beats(intervals, unit, meter.beats_per_bar)
    duration = np.asarray([cell.end - cell.start for cell in cells], dtype=np.float32)
    # The extractor is unchanged. Reuse expensive legacy CQT/fast features but
    # never its unvalidated targets; recomputing the beat cells is inexpensive.
    features = None
    if legacy_cache.exists():
        with np.load(legacy_cache, allow_pickle=False) as saved:
            if saved["duration"].shape == duration.shape and np.array_equal(saved["duration"], duration):
                features = saved["vectors"].copy(), saved["bass"].copy()
    if features is None:
        chroma = studio.extract_chroma(y, studio.SAMPLE_RATE, 2048, mode=feature_mode)
        bass = studio.extract_bass_chroma(y, studio.SAMPLE_RATE, 2048, mode=feature_mode)
        vectors = studio.chroma_for_intervals(chroma, y, studio.SAMPLE_RATE, 2048, cells)
        bv, _, _ = studio.bass_for_intervals(bass, studio.SAMPLE_RATE, 2048, cells, "sharps")
    else:
        vectors, bv = features
    targets = np.zeros((len(cells), 24), dtype=np.float32)
    for start, end, chord in reference_rows(reference):
        for i, cell in enumerate(cells):
            targets[i, chord] += max(0., min(end, cell.end) - max(start, cell.start))
    data = dict(vectors=vectors, bass=bv, targets=targets, duration=duration)
    np.savez_compressed(cache, **data)
    return data


def training_arrays(songs):
    xs, ys, weights = [], [], []
    for data in songs:
        dominant = np.argmax(data["targets"], axis=1)
        agreement = np.max(data["targets"], axis=1) / np.maximum(data["duration"], 1e-8)
        keep = agreement >= 0.8
        features = candidate_features(data["vectors"], data["bass"]).reshape(-1, 24, 87)
        y = (np.arange(24)[None] == dominant[:, None]).astype(np.int8)
        xs.append(features[keep].reshape(-1, 87))
        ys.append(y[keep].ravel())
        # Equal interval weight avoids letting a slow song dominate; the
        # positive weight balances candidate negatives without resampling songs.
        weights.append(np.where(y[keep], 12., 1.).ravel())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


def evaluate(artifact, songs, strengths=(0, 0.25, 0.5, 1.0)):
    records = []
    for data in songs:
        scores = emission_scores(artifact, data["vectors"], data["bass"])
        result = {}
        total = float(data["targets"].sum())
        for strength in strengths:
            path = decode_emissions(scores, strength)
            correct = float(data["targets"][np.arange(len(path)), path].sum())
            result[str(strength)] = correct / max(total, 1e-8)
        records.append(result)
    return {str(strength): float(np.mean([item[str(strength)] for item in records])) for strength in strengths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".tmp/acoustic_features"))
    parser.add_argument("--feature-mode", choices=["fast", "cqt"], default="fast")
    parser.add_argument("--harmonic-unit", choices=["bar", "half-bar", "beat"], default="half-bar")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Training accepts only explicitly labeled train rows; pass a train-only manifest.")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate song IDs in training manifest.")
    audio_paths = [str((manifest.parent / row["audio"]).resolve()).casefold() for row in rows]
    if len(set(audio_paths)) != len(rows):
        raise ValueError("Repeated audio paths must be grouped before training.")
    if args.limit:
        rows = rows[:args.limit]
    if len(rows) < 5:
        raise ValueError("At least five independent songs are required.")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    songs = []
    for i, row in enumerate(rows):
        songs.append(song_data(row, manifest.parent, args.cache_dir, args.feature_mode, args.harmonic_unit))
        print(f"Features {i + 1}/{len(rows)}: {row['id']}", flush=True)
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(
        np.arange(len(rows)), groups=[row["id"] for row in rows]))
    params = dict(max_iter=args.iterations, max_leaf_nodes=15, learning_rate=0.08,
                  min_samples_leaf=40, l2_regularization=2., max_bins=63,
                  early_stopping=False, random_state=args.seed)
    with threadpool_limits(limits=args.threads):
        x, y, weights = training_arrays([songs[i] for i in train])
        model = HistGradientBoostingClassifier(**params).fit(x, y, sample_weight=weights)
        artifact = {"feature_schema": FEATURE_SCHEMA, "model": model}
        dev = evaluate(artifact, [songs[i] for i in validation])
        strength = float(max(dev, key=dev.get))
        report = {"protocol": "reference_beat_grid_major_minor_only_song_held_out",
            "limitations": ["One artist/recording domain", "Uses reference beat grid", "24 major/minor outputs only",
                            "Extensions excluded from target quality", "Ranking scores are not calibrated confidence"],
            "train_ids": [rows[i]["id"] for i in train], "validation_ids": [rows[i]["id"] for i in validation],
            "validation_macro_accuracy_by_transition": dev, "selected_transition_strength": strength,
            "feature_schema": FEATURE_SCHEMA, "feature_mode": args.feature_mode, "harmonic_unit": args.harmonic_unit,
            "target_schema": TARGET_SCHEMA,
            "sample_weighting": "Each retained interval contributes 23 negative candidates of weight 1 and one positive of weight 12; no duration/song reweighting",
            "model_parameters": params, "fit_candidate_rows": len(y), "seed": args.seed}
        print(json.dumps(report, indent=2), flush=True)
        x, y, weights = training_arrays(songs)
        model = HistGradientBoostingClassifier(**params).fit(x, y, sample_weight=weights)
    artifact.update(model=model, report=report, target_schema=TARGET_SCHEMA, feature_mode=args.feature_mode,
                    harmonic_unit=args.harmonic_unit, transition_strength=strength,
                    training_song_ids=[row["id"] for row in rows],
                    training_audio_paths=audio_paths[:len(rows)],
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    report["elapsed_seconds"] = time.perf_counter() - started
    joblib.dump(artifact, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved experimental model: {args.output}", flush=True)


if __name__ == "__main__":
    main()
