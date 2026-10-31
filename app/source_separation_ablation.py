#!/usr/bin/env python
"""Compare pitch evidence across full mix and separated/proxy stems."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from chord_studio import SAMPLE_RATE
from evaluate_pitch_evidence import (
    build_acoustic_vectors,
    chord_label,
    evaluate,
    evaluate_from_vectors,
    iter_spans,
    load_json,
    resolve_path,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "source"


def load_audio(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def write_audio(path: Path, y: np.ndarray) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    sf.write(path, y, SAMPLE_RATE)
    return path


def find_stem(stems_dir: Path, name: str) -> Path | None:
    if not stems_dir.exists():
        return None
    target = name.lower()
    candidates = [
        path
        for path in stems_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}
        and path.stem.lower() == target
    ]
    if candidates:
        return sorted(candidates, key=lambda path: len(path.parts))[0]
    return None


def stem_mix(stems: list[Path], output: Path) -> Path | None:
    arrays = []
    min_len = None
    for stem in stems:
        if stem is None or not stem.exists():
            return None
        y = load_audio(stem)
        arrays.append(y)
        min_len = y.size if min_len is None else min(min_len, y.size)
    if not arrays or min_len is None or min_len <= 0:
        return None
    mix = np.sum([array[:min_len] for array in arrays], axis=0)
    return write_audio(output, mix)


def hpss_harmonic(audio: Path, output: Path) -> Path:
    import librosa

    y = load_audio(audio)
    harmonic, _ = librosa.effects.hpss(y)
    return write_audio(output, harmonic)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    peak = float(np.max(values)) if values.size else 0.0
    return values if peak <= 0 else values / peak


def fuse_vectors_max(primary: list[np.ndarray], secondary: list[np.ndarray], secondary_weight: float) -> list[np.ndarray]:
    fused = []
    for first, second in zip(primary, secondary):
        fused.append(np.maximum(normalize_vector(first), normalize_vector(second) * secondary_weight))
    return fused


def fuse_vectors_weighted(
    primary: list[np.ndarray],
    secondary: list[np.ndarray],
    primary_weight: float,
    secondary_weight: float,
) -> list[np.ndarray]:
    fused = []
    for first, second in zip(primary, secondary):
        fused.append(normalize_vector(first) * primary_weight + normalize_vector(second) * secondary_weight)
    return fused


def source_paths(args: argparse.Namespace, work_dir: Path) -> list[tuple[str, Path]]:
    audio = resolve_path(args.audio)
    sources: list[tuple[str, Path]] = [("full_mix", audio)]
    if args.hpss:
        sources.append(("hpss_harmonic", hpss_harmonic(audio, work_dir / "hpss_harmonic.wav")))
    stems_dir = resolve_path(args.stems_dir) if args.stems_dir else None
    if stems_dir and stems_dir.exists():
        stems = {name: find_stem(stems_dir, name) for name in ("vocals", "drums", "bass", "other")}
        for name, path in stems.items():
            if path is not None:
                sources.append((f"stem_{name}", path))
        bass_other = stem_mix(
            [path for path in (stems.get("bass"), stems.get("other")) if path is not None],
            work_dir / "stem_bass_other.wav",
        )
        if bass_other:
            sources.append(("stem_bass_other", bass_other))
        no_vocals = stem_mix(
            [path for path in (stems.get("drums"), stems.get("bass"), stems.get("other")) if path is not None],
            work_dir / "stem_no_vocals.wav",
        )
        if no_vocals:
            sources.append(("stem_no_vocals", no_vocals))
    return sources


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_roman_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        roman = str(row.get("reference_roman") or "N")
        grouped.setdefault(roman, []).append(row)
    breakdown = []
    for roman, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        recalls = [float(row["pitch_recall"]) for row in items if row.get("pitch_recall") not in {"", None}]
        all_present = sum(1 for row in items if not row.get("missing_tones")) / max(len(items), 1)
        breakdown.append(
            {
                "roman": roman,
                "count": len(items),
                "pitch_mean_recall": sum(recalls) / max(len(recalls), 1) if recalls else None,
                "pitch_all_tones_present": all_present,
            }
        )
    return breakdown


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(resolve_path(args.analysis))
    spans = [span for span in iter_spans(payload) if chord_label(span) != "N"]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = resolve_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_summaries = []
    local_args = SimpleNamespace(
        mode=args.mode,
        notation=args.notation,
        feature_mode=args.feature_mode,
        hop_length=args.hop_length,
        top_n=args.top_n,
        pitch_threshold=args.pitch_threshold,
    )
    for source_name, source_audio in source_paths(args, work_dir):
        rows, summary = evaluate(payload, source_audio, local_args)
        summary["source"] = source_name
        summary["audio"] = str(source_audio)
        summary["roman_breakdown"] = source_roman_breakdown(rows)
        source_summaries.append(summary)
        stem = safe_name(source_name)
        write_rows(output_dir / f"{stem}_pitch_evidence.csv", rows)
        (output_dir / f"{stem}_pitch_evidence.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    stems_dir = resolve_path(args.stems_dir) if args.stems_dir else None
    if stems_dir and stems_dir.exists():
        bass = find_stem(stems_dir, "bass")
        other = find_stem(stems_dir, "other")
        if bass and other:
            bass_vectors = build_acoustic_vectors(bass, spans, args.feature_mode, args.hop_length)
            other_vectors = build_acoustic_vectors(other, spans, args.feature_mode, args.hop_length)
            fused_sources = {
                "feature_other_bass_max055": fuse_vectors_max(other_vectors, bass_vectors, 0.55),
                "feature_other_bass_max075": fuse_vectors_max(other_vectors, bass_vectors, 0.75),
                "feature_other_bass_weighted": fuse_vectors_weighted(other_vectors, bass_vectors, 0.85, 0.25),
            }
            for source_name, vectors in fused_sources.items():
                rows, summary = evaluate_from_vectors(payload, vectors, local_args, spans=spans)
                summary["source"] = source_name
                summary["audio"] = f"feature-fusion:{other}+{bass}"
                summary["roman_breakdown"] = source_roman_breakdown(rows)
                source_summaries.append(summary)
                stem = safe_name(source_name)
                write_rows(output_dir / f"{stem}_pitch_evidence.csv", rows)
                (output_dir / f"{stem}_pitch_evidence.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    consolidated = {
        "analysis": str(resolve_path(args.analysis)),
        "sources": source_summaries,
    }
    (output_dir / "source_ablation_summary.json").write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_rows = []
    for item in source_summaries:
        row = {
            "source": item["source"],
            "items": item["items"],
            "pitch_mean_recall": item["pitch_mean_recall"],
            "pitch_all_tones_present": item["pitch_all_tones_present"],
            "oracle_exact_top1": item["oracle_exact_top1"],
            "oracle_root_top1": item["oracle_root_top1"],
        }
        for roman_item in item["roman_breakdown"]:
            roman = safe_name(roman_item["roman"])
            row[f"{roman}_count"] = roman_item["count"]
            row[f"{roman}_mean_recall"] = roman_item["pitch_mean_recall"]
            row[f"{roman}_all_present"] = roman_item["pitch_all_tones_present"]
        summary_rows.append(row)
    all_fields = []
    for row in summary_rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    with (output_dir / "source_ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return consolidated


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--stems-dir", type=Path, help="Directory containing Demucs stems: vocals/drums/bass/other.")
    parser.add_argument("--hpss", action="store_true", help="Add librosa HPSS harmonic as a lightweight proxy source.")
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="seventh")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="flats")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--pitch-threshold", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".tmp" / "source_ablation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
