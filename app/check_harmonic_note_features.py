"""Small synthetic mechanism checks and optional train-audio timing only.

No annotation files, fitted classifier, threshold search or evaluation set are
used. Passing --train-audio is an explicit caller declaration of training audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from chord_recognizer import extract_chroma
from harmonic_note_features import (SAMPLE_RATE, HOP_LENGTH, MIDI_NOTES,
    FEATURE_SCHEMA, FEATURE_NAMES, extract_note_evidence, features_for_intervals,
    sha256_file)


def tone(midi, duration=4., onset=0., offset=None, harmonic_shape=.7):
    times = np.arange(round(duration * SAMPLE_RATE)) / SAMPLE_RATE
    frequency = 440 * 2 ** ((midi - 69) / 12)
    wave = np.zeros(len(times))
    for harmonic in range(1, 17):
        if frequency * harmonic >= SAMPLE_RATE / 2:
            break
        wave += harmonic_shape ** (harmonic - 1) * np.sin(2 * np.pi * frequency * harmonic * times + .137 * harmonic)
    end = duration if offset is None else offset
    envelope = np.minimum(np.clip((times - onset) / .025, 0, 1), np.clip((end - times) / .025, 0, 1))
    return wave * envelope


def synthetic_checks(output):
    output.mkdir(parents=True, exist_ok=True)
    base = sum(tone(midi) for midi in [48, 52, 55])
    cases = {
        "root_harmonics_only": tone(48),
        "major_triad": base,
        "minor_seventh": base + tone(58),
        "major_seventh": base + tone(59),
        "brief_major_seventh_melody": base + tone(59, onset=1.8, offset=2.2),
        "transposed_major_seventh": sum(tone(midi) for midi in [50, 54, 57, 61]),
        "silence": np.zeros(4 * SAMPLE_RATE),
    }
    result = {}
    for name, y in cases.items():
        path = output / (name + ".wav")
        y = (.5 * y / max(1., np.max(np.abs(y)))).astype(np.float32)
        sf.write(path, y, SAMPLE_RATE, subtype="FLOAT")
        evidence = extract_note_evidence(path, cache_dir=output / "cache")
        root = "D" if name == "transposed_major_seventh" else "C"
        values = features_for_intervals(evidence, [SimpleNamespace(start=.5, end=3.5, chord=root)])[0]
        stft = extract_chroma(y, SAMPLE_RATE, 2048, "fast")
        stft = stft / np.maximum(stft.sum(axis=0, keepdims=True), 1e-12)
        stft_times = np.arange(stft.shape[1]) * 2048 / SAMPLE_RATE
        stft_mean = stft[:, (stft_times >= .5) & (stft_times < 3.5)].mean(axis=1)
        stft_mean = stft_mean[(np.arange(12) + (2 if root == "D" else 0)) % 12]
        expected_times = np.arange(0, len(y), HOP_LENGTH) / SAMPLE_RATE
        np.testing.assert_array_equal(evidence.times, expected_times)
        assert values.shape == (97,) and np.all(np.isfinite(values))
        result[name] = dict(audio=str(path.resolve()), audio_sha256=sha256_file(path),
            elapsed_seconds=evidence.metadata["elapsed_seconds"], frame_count=len(evidence.times),
            nnls_mean_relative_chroma=values[:12].tolist(),
            existing_fast_stft_mean_relative_chroma=stft_mean.tolist(),
            minor_seventh_support=float(values[36 + 10]),
            major_seventh_support=float(values[36 + 11]),
            major_seventh_longest_support=float(values[84 + 11]),
            midi59_mean_activation=float(evidence.activations[59 - 21, (evidence.times >= .5) & (evidence.times < 3.5)].mean()),
            evidence_identity=evidence.metadata["identity"])
    triad = result["major_triad"]["nnls_mean_relative_chroma"]
    assert result["minor_seventh"]["nnls_mean_relative_chroma"][10] > triad[10]
    assert result["major_seventh"]["nnls_mean_relative_chroma"][11] > triad[11]
    assert not any(result["silence"]["nnls_mean_relative_chroma"])
    result["mechanism_checks"] = dict(
        complete_exact_sample_clock=True, finite_97_dimensions=True,
        actual_seventh_exceeds_triad_evidence=True,
        sustained_seventh_exceeds_brief_melody_persistence=bool(result["major_seventh"]["major_seventh_longest_support"] > result["brief_major_seventh_melody"]["major_seventh_longest_support"]),
        actual_seventh_note_midi59_exceeds_triad=bool(result["major_seventh"]["midi59_mean_activation"] > result["major_triad"]["midi59_mean_activation"]),
        silence_remains_zero=True,
        transposition_mean_chroma_max_difference=float(np.max(np.abs(
            np.asarray(result["major_seventh"]["nnls_mean_relative_chroma"]) -
            np.asarray(result["transposed_major_seventh"]["nnls_mean_relative_chroma"])))),
        caveat="These are mechanism checks, not a chord accuracy benchmark or calibrated probabilities",
        observed_limitation="Global pitch-class persistence can remain high for upper partials; full note/register evidence is necessary. No support threshold was adjusted to make this check pass.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-audio", type=Path)
    parser.add_argument("--train-duration", type=float, default=20.)
    args = parser.parse_args()
    report = dict(feature_schema=FEATURE_SCHEMA, feature_dimension=len(FEATURE_NAMES),
                  synthetic=synthetic_checks(args.output / "synthetic"),
                  reference_files_read=False, classifier_fitted=False, threshold_selection=False)
    if args.train_audio:
        started = time.perf_counter()
        evidence = extract_note_evidence(args.train_audio, cache_dir=args.output / "train_cache", duration=args.train_duration)
        boundaries = np.arange(0, evidence.metadata["duration_seconds"], 1.)
        rows = [SimpleNamespace(start=float(start), end=min(float(start + 1), evidence.metadata["duration_seconds"]), chord="C") for start in boundaries]
        # C here is an arbitrary coordinate anchor, not an audio/key prediction.
        features = features_for_intervals(evidence, rows)
        report["train_audio_timing"] = dict(audio=str(args.train_audio.resolve()),
            duration=evidence.metadata["duration_seconds"], frame_count=len(evidence.times),
            features_shape=list(features.shape), all_finite=bool(np.all(np.isfinite(features))),
            elapsed_seconds=time.perf_counter() - started,
            arbitrary_coordinate_anchor="C; no key/chord/reference read or evaluated",
            evidence_metadata=evidence.metadata)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(mechanism_checks=report["synthetic"]["mechanism_checks"],
                         train_audio_timing={k: v for k, v in report.get("train_audio_timing", {}).items() if k != "evidence_metadata"}), indent=2))


if __name__ == "__main__":
    main()
