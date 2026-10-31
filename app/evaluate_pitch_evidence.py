#!/usr/bin/env python
"""Evaluate pitch-class evidence and oracle chord interpretation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from chord_recognizer import build_templates, extract_chroma
from chord_studio import SAMPLE_RATE, BeatInterval, chroma_for_intervals, chord_quality_family, parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_audio(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def iter_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for bar in payload.get("bars", []):
        for span in bar.get("spans") or []:
            item = dict(span)
            item["bar"] = int(bar.get("bar", item.get("bar", 0)))
            spans.append(item)
    return sorted(spans, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))


def chord_label(span: dict[str, Any]) -> str:
    return str(span.get("chord_display") or span.get("chord") or "N").split("/", 1)[0]


def chord_tones(chord: str) -> tuple[int, ...]:
    chord = str(chord or "N").split("/", 1)[0]
    root, suffix = parse_chord_name(chord)
    if root is None:
        return tuple()
    suffix = suffix or ""
    lowered = suffix.lower()
    if lowered.startswith("m7b5"):
        intervals = [0, 3, 6, 10]
    elif lowered.startswith("dim7"):
        intervals = [0, 3, 6, 9]
    elif lowered.startswith("dim"):
        intervals = [0, 3, 6]
    elif lowered.startswith("aug"):
        intervals = [0, 4, 8]
    elif lowered.startswith("sus2"):
        intervals = [0, 2, 7]
    elif lowered.startswith("sus4") or lowered.startswith("sus"):
        intervals = [0, 5, 7]
    elif lowered.startswith("m"):
        intervals = [0, 3, 7]
        extension = lowered[1:]
        if "maj7" in extension:
            intervals.append(11)
        elif "7" in extension or "9" in extension:
            intervals.append(10)
        if "6" in extension:
            intervals.append(9)
        if "9" in extension or "add9" in extension:
            intervals.append(2)
    else:
        intervals = [0, 4, 7]
        if "maj7" in lowered:
            intervals.append(11)
        elif "7" in lowered or "9" in lowered:
            intervals.append(10)
        if "6" in lowered:
            intervals.append(9)
        if "9" in lowered or "add9" in lowered:
            intervals.append(2)
    return tuple(sorted({(root + interval) % 12 for interval in intervals}))


def normalized_tone_vector(tones: tuple[int, ...]) -> np.ndarray:
    vector = np.zeros(12, dtype=np.float32)
    for pc in tones:
        vector[pc] = 1.0
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def root_pc(chord: str) -> int | None:
    root, _ = parse_chord_name(str(chord or "N").split("/", 1)[0])
    return root


def family(chord: str) -> str:
    _, suffix = parse_chord_name(str(chord or "N").split("/", 1)[0])
    return chord_quality_family(suffix)


def rank_of(candidates: list[str], predicate) -> int | None:
    for index, chord in enumerate(candidates, start=1):
        if predicate(chord):
            return index
    return None


def evidence_stats(vector: np.ndarray, tones: tuple[int, ...], threshold: float) -> dict[str, Any]:
    if not tones:
        return {
            "pitch_recall": "",
            "mean_tone_energy": "",
            "min_tone_energy": "",
            "missing_tones": "",
            "nonchord_max_energy": "",
        }
    values = np.asarray(vector, dtype=np.float32)
    peak = float(np.max(values)) if values.size else 0.0
    if peak > 0:
        values = values / peak
    tone_values = [float(values[pc]) for pc in tones]
    missing = [str(pc) for pc in tones if float(values[pc]) < threshold]
    nonchord = [float(values[pc]) for pc in range(12) if pc not in tones]
    return {
        "pitch_recall": f"{(len(tones) - len(missing)) / max(len(tones), 1):.6f}",
        "mean_tone_energy": f"{float(np.mean(tone_values)):.6f}",
        "min_tone_energy": f"{float(np.min(tone_values)):.6f}",
        "missing_tones": " ".join(missing),
        "nonchord_max_energy": f"{max(nonchord) if nonchord else 0.0:.6f}",
    }


def build_acoustic_vectors(audio: Path | None, spans: list[dict[str, Any]], feature_mode: str, hop_length: int) -> list[np.ndarray] | None:
    if audio is None:
        return None
    y = load_audio(audio)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    intervals = [
        BeatInterval(
            bar=int(span.get("bar", 0)),
            beat=1,
            start=float(span.get("start", 0.0)),
            end=float(span.get("end", 0.0)),
        )
        for span in spans
    ]
    return list(chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals))


def evaluate_from_vectors(
    payload: dict[str, Any],
    acoustic_vectors: list[np.ndarray] | None,
    args: argparse.Namespace,
    spans: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if spans is None:
        spans = [span for span in iter_spans(payload) if chord_label(span) != "N"]
    templates = build_templates(mode=args.mode, notation=args.notation)
    matrix = np.vstack([template.vector for template in templates])
    template_names = [template.name for template in templates]
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for index, span in enumerate(spans):
        reference = chord_label(span)
        tones = chord_tones(reference)
        oracle_vector = normalized_tone_vector(tones)
        scores = oracle_vector @ matrix.T
        order = np.argsort(scores)[::-1]
        ranked = [template_names[int(item)] for item in order]
        exact_rank = rank_of(ranked, lambda chord: chord == reference)
        root_rank = rank_of(ranked, lambda chord: root_pc(chord) == root_pc(reference))
        family_rank = rank_of(
            ranked,
            lambda chord: root_pc(chord) == root_pc(reference) and family(chord) == family(reference),
        )
        top1 = ranked[0] if ranked else "N"
        counters["items"] += 1
        counters["oracle_exact_top1"] += int(top1 == reference)
        counters["oracle_root_top1"] += int(root_pc(top1) == root_pc(reference))
        counters["oracle_family_top1"] += int(
            root_pc(top1) == root_pc(reference) and family(top1) == family(reference)
        )
        counters["oracle_exact_topn"] += int(exact_rank is not None and exact_rank <= args.top_n)
        counters["oracle_root_topn"] += int(root_rank is not None and root_rank <= args.top_n)
        row = {
            "bar": span.get("bar", ""),
            "beat_start": span.get("beat_start", ""),
            "beat_end": span.get("beat_end", ""),
            "start": f"{float(span.get('start', 0.0)):.3f}",
            "end": f"{float(span.get('end', 0.0)):.3f}",
            "reference_chord": reference,
            "reference_roman": span.get("roman_display") or span.get("roman") or "",
            "tone_pcs": " ".join(str(pc) for pc in tones),
            "oracle_top1": top1,
            "oracle_exact_rank": "" if exact_rank is None else exact_rank,
            "oracle_root_rank": "" if root_rank is None else root_rank,
            "oracle_family_rank": "" if family_rank is None else family_rank,
            "oracle_topn": " | ".join(f"{rank + 1}:{ranked[rank]}:{float(scores[order[rank]]):.3f}" for rank in range(min(args.top_n, len(ranked)))),
        }
        if acoustic_vectors is not None:
            row.update(evidence_stats(acoustic_vectors[index], tones, args.pitch_threshold))
            if row["pitch_recall"] != "":
                counters["pitch_items"] += 1
                counters["pitch_recall_sum"] += float(row["pitch_recall"])
                counters["pitch_all_tones_present"] += int(float(row["pitch_recall"]) >= 1.0)
        rows.append(row)
    total = max(counters["items"], 1)
    pitch_total = max(counters["pitch_items"], 1)
    summary = {
        "items": counters["items"],
        "mode": args.mode,
        "notation": args.notation,
        "top_n": args.top_n,
        "oracle_exact_top1": counters["oracle_exact_top1"] / total,
        "oracle_root_top1": counters["oracle_root_top1"] / total,
        "oracle_family_top1": counters["oracle_family_top1"] / total,
        "oracle_exact_topn": counters["oracle_exact_topn"] / total,
        "oracle_root_topn": counters["oracle_root_topn"] / total,
        "pitch_mean_recall": counters["pitch_recall_sum"] / pitch_total if counters["pitch_items"] else None,
        "pitch_all_tones_present": counters["pitch_all_tones_present"] / pitch_total if counters["pitch_items"] else None,
        "pitch_threshold": args.pitch_threshold if counters["pitch_items"] else None,
    }
    return rows, summary


def evaluate(payload: dict[str, Any], audio: Path | None, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spans = [span for span in iter_spans(payload) if chord_label(span) != "N"]
    acoustic_vectors = build_acoustic_vectors(audio, spans, args.feature_mode, args.hop_length)
    return evaluate_from_vectors(payload, acoustic_vectors, args, spans=spans)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, help="Reference/corrected Chord Studio analysis JSON.")
    parser.add_argument("--audio", type=Path, help="Optional audio file for acoustic pitch evidence.")
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="seventh")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="flats")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--pitch-threshold", type=float, default=0.25)
    parser.add_argument("-o", "--output", type=Path, required=True, help="CSV output path.")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = load_json(resolve_path(args.analysis))
    rows, summary = evaluate(payload, resolve_path(args.audio) if args.audio else None, args)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.summary:
        summary_path = resolve_path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
