"""Evaluate section boundaries against local Isophonics structure annotations.

Reference bars are an explicit oracle-grid diagnostic, not end-to-end beat
tracking accuracy. Audio features are cached; structure labels never enter the
detector. Excludes global track endpoints from boundary precision/recall.
Cached features bind source bytes, decoded PCM, exact grid and extraction code.
Analysis offsets are applied to decoding and source-time reference boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chord_recognizer import SAMPLE_RATE, decode_audio
from song_structure import StructureConfig, analyze_feature_structure, extract_bar_features, valid_bars
from structure_feature_cache import WIDTHS, cache_identity, cache_path, file_digest, load_features, save_features


def boundary_metrics(reference: list[float], estimated: list[float], tolerance: float) -> dict[str, float]:
    """Maximum one-to-one matching of sorted 1D boundary times."""
    reference, estimated = sorted(reference), sorted(estimated)
    i = j = matches = 0
    while i < len(reference) and j < len(estimated):
        if abs(reference[i] - estimated[j]) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif reference[i] < estimated[j]:
            i += 1
        else:
            j += 1
    precision = matches / len(estimated) if estimated else float(not reference)
    recall = matches / len(reference) if reference else float(not estimated)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "matches": matches, "reference_count": len(reference), "estimated_count": len(estimated)}


def locate_annotations(root: Path, track_id: str) -> Path | None:
    from import_annotations import safe_id

    for path in (root / "raw/isophonics/beatles/seglab").rglob("*.lab"):
        # Sanitize the path once, as the original importer does. Joining its
        # parts with underscores first changes "Help!/" into "Help!_" and
        # incorrectly loses the complete Help! album during matching.
        suffix = safe_id(str(path.relative_to(root / "raw/isophonics/beatles/seglab").with_suffix("")))
        if track_id == "isophonics_all_" + suffix:
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--grid", choices=["reference", "prediction"], default="reference")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=StructureConfig().novelty_threshold)
    parser.add_argument("--min-section-bars", type=int, default=StructureConfig().min_section_bars)
    parser.add_argument("--novelty-widths", default=",".join(map(str, StructureConfig().novelty_widths)))
    parser.add_argument("--cache", type=Path, default=Path("benchmarks/reports/session_20260926/structure_features"))
    parser.add_argument("--save-payloads", action="store_true")
    args = parser.parse_args()
    root = Path("benchmarks").resolve()
    config = StructureConfig(novelty_threshold=args.threshold, min_section_bars=args.min_section_bars, novelty_widths=tuple(int(value) for value in args.novelty_widths.split(",")))
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]
    args.cache.mkdir(parents=True, exist_ok=True)
    results, skipped = [], []
    for row in rows:
        track_id = row["id"]
        reference_path = root / "references/isophonics" / (track_id + "_reference_analysis.json")
        payload_path = reference_path if args.grid == "reference" else root / row["prediction"]
        annotation = locate_annotations(root, track_id)
        audio = root / row["audio"]
        if not payload_path.exists() or annotation is None or not audio.exists():
            skipped.append({"id": track_id, "reason": "Missing audio, analysis grid or section annotation"})
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8-sig"))
        # Reject inconsistent row identities before decoding or creating a
        # cache. The detector's later validation is too late for these costs.
        bars = valid_bars(payload)
        if len(bars) > config.max_bars:
            raise ValueError(f"Structure analysis supports at most {config.max_bars} bars per window")
        reference_payload = payload if args.grid == "reference" else json.loads(reference_path.read_text(encoding="utf-8-sig")) if reference_path.exists() else {}
        current_grid = np.asarray([[bar["start"], bar["end"]] for bar in payload["bars"]])
        reference_grid = np.asarray([[bar["start"], bar["end"]] for bar in reference_payload.get("bars", [])])
        grid_matches_reference = bool(len(current_grid) and float(payload.get("audio_offset", 0)) == float(reference_payload.get("audio_offset", 0))
                                      and current_grid.shape == reference_grid.shape and np.allclose(current_grid, reference_grid, atol=1e-4, rtol=0))
        grid_signature = np.asarray([[bar["start"], bar["end"]] for bar in payload["bars"]], dtype=float).reshape(-1, 2)
        # Decode the requested window every time; source bytes and actual PCM
        # bind cached features even after same-size/mtime changes or decoder
        # differences. Legacy per-track caches are preserved, never trusted.
        offset, duration = float(payload.get("audio_offset", 0)), float(payload["duration"])
        audio_sha256 = file_digest(audio)
        y = decode_audio(audio, sample_rate=SAMPLE_RATE, offset=offset, duration=duration)
        if file_digest(audio) != audio_sha256:
            raise ValueError("Audio changed while decoding structure features")
        identity = cache_identity(audio_sha256, y, SAMPLE_RATE, offset, duration, grid_signature, extract_bar_features)
        feature_path = cache_path(args.cache, identity)
        features = load_features(feature_path, identity, grid_signature)
        cache_reused = features is not None
        invalid_backup = None
        if features is None:
            features = extract_bar_features(y, SAMPLE_RATE, payload["bars"]) if len(grid_signature) else {name: np.empty((0, width)) for name, width in WIDTHS.items()}
            invalid_backup = save_features(feature_path, identity, grid_signature, features)
        structure = analyze_feature_structure(payload, features, config)
        labels = [line.split(maxsplit=2) for line in annotation.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        # Annotations use source time; predictions use analysis-window time.
        # Exclude crop edges just as full-track endpoints are excluded.
        window_end = offset + len(y) / SAMPLE_RATE
        reference = [float(line[0]) - offset for line in labels[1:] if offset < float(line[0]) < window_end]
        estimated = [float(item["time"]) for item in structure["boundaries"]]
        # Report both complete internal annotations and musical transitions, since
        # a partial pickup/silence is often shorter than a single bar.
        musical = [float(labels[index][0]) - offset for index in range(1, len(labels))
                   if labels[index][2] != "silence" and labels[index - 1][2] != "silence" and offset < float(labels[index][0]) < window_end]
        item = {"id": track_id, "annotation": str(annotation), "grid": str(payload_path), "grid_matches_reference": grid_matches_reference, "section_count": len(structure["sections"]), "reference": reference, "estimated": estimated, "metrics": {str(tolerance): boundary_metrics(reference, estimated, tolerance) for tolerance in (0.5, 3.0)}, "musical_metrics": {str(tolerance): boundary_metrics(musical, estimated, tolerance) for tolerance in (0.5, 3.0)}}
        item["feature_cache"] = {"schema": identity["schema"], "path": str(feature_path), "reused": cache_reused,
                                 "sha256": file_digest(feature_path), "input_identity": identity,
                                 "invalid_predecessor_preserved": str(invalid_backup) if invalid_backup else None,
                                 "legacy_unbound_cache_used": False}
        item["analysis_window"] = {"source_offset": offset, "decoded_duration": len(y) / SAMPLE_RATE,
                                   "reference_clock": "source boundaries shifted to local window; crop endpoints excluded"}
        results.append(item)
        print(f"{track_id}: sections={len(structure['sections'])} F1@3s={item['metrics']['3.0']['f1']:.3f}", flush=True)
        if args.save_payloads:
            payload["structure"] = structure
            destination = args.output.parent / "structure_payloads" / (track_id + "_structure_analysis.json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"method": "bar_multifeature_novelty", "grid_type": args.grid, "oracle_grid": args.grid == "reference" or bool(results and all(item["grid_matches_reference"] for item in results)), "reference_grid_match_count": sum(item["grid_matches_reference"] for item in results), "config": vars(config), "manifest": str(args.manifest), "track_count": len(results), "macro_metrics": {str(tolerance): {metric: float(np.mean([item["metrics"][str(tolerance)][metric] for item in results])) if results else None for metric in ("precision", "recall", "f1")} for tolerance in (0.5, 3.0)}, "tracks": results, "skipped": skipped}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"tracks", "skipped"}}, indent=2))


if __name__ == "__main__":
    main()
