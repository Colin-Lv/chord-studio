#!/usr/bin/env python
"""Diagnose why an automatic analysis differs from a corrected analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from chord_recognizer import build_templates, extract_chroma
from chord_studio import (
    SAMPLE_RATE,
    BeatInterval,
    apply_bass_bias,
    bass_for_intervals,
    chroma_for_intervals,
    chord_to_roman_display,
    extract_bass_chroma,
    note_name_for_pc,
    parse_chord_name,
    template_biases,
)
from downbeat_external_fusion import roman_extended_label, roman_function_label


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


def overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(left.get("end", 0.0)), float(right.get("end", 0.0)))
        - max(float(left.get("start", 0.0)), float(right.get("start", 0.0))),
    )


def best_cover(span: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_amount = 0.0
    for candidate in candidates:
        amount = overlap(span, candidate)
        if amount > best_amount:
            best = candidate
            best_amount = amount
    return best, best_amount


def label(item: dict[str, Any] | None, field: str) -> str:
    if not item:
        return "N"
    if field == "roman":
        return str(item.get("roman_display") or item.get("roman") or "N")
    return str(item.get("chord_display") or item.get("chord") or "N")


def root_name(chord: str) -> str:
    root, _ = parse_chord_name(chord)
    return "N" if root is None else note_name_for_pc(root)


def root_pc(chord: str) -> int | None:
    root, _ = parse_chord_name(chord)
    return root


def root_motion(prev_chord: str, chord: str) -> str:
    prev = root_pc(prev_chord)
    curr = root_pc(chord)
    if prev is None or curr is None:
        return ""
    interval = (curr - prev) % 12
    names = {
        0: "stay",
        1: "up_m2",
        2: "up_M2",
        3: "up_m3",
        4: "up_M3",
        5: "up_P4",
        7: "up_P5",
        8: "down_M3",
        9: "down_m3",
        10: "down_M2",
        11: "down_m2",
    }
    return names.get(interval, f"pc_{interval}")


def read_prior_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    raw = load_json(resolve_path(path))
    scores = raw.get("pair_scores") if isinstance(raw, dict) else {}
    return {str(key): float(value) for key, value in (scores or {}).items()}


def prior_key(prev_roman: str, curr_roman: str, extended: bool) -> str:
    if extended:
        return f"{roman_extended_label(prev_roman)}->{roman_extended_label(curr_roman)}"
    return f"{roman_function_label(prev_roman)}->{roman_function_label(curr_roman)}"


def acoustic_candidates(
    audio: Path,
    corrected_spans: list[dict[str, Any]],
    key: str,
    mode: str,
    notation: str,
    feature_mode: str,
    hop_length: int,
    top_n: int,
    bass_weight: float,
) -> dict[int, list[dict[str, Any]]]:
    y = load_audio(audio)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    bass_chroma = extract_bass_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    intervals = [
        BeatInterval(
            bar=int(span.get("bar", 0)),
            beat=1,
            start=float(span.get("start", 0.0)),
            end=float(span.get("end", 0.0)),
        )
        for span in corrected_spans
    ]
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals)
    bass_vectors, bass_roots, bass_confidence = bass_for_intervals(
        bass_chroma, SAMPLE_RATE, hop_length, intervals, notation
    )
    templates = build_templates(mode=mode, notation=notation)
    matrix = np.vstack([template.vector for template in templates])
    names = [template.name for template in templates]
    scores = vectors @ matrix.T
    for index in range(scores.shape[0]):
        scores[index] += template_biases(names, key)
    scores = apply_bass_bias(scores, bass_vectors, bass_confidence, names, bass_weight)
    result: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(scores):
        order = np.argsort(row)[::-1][:top_n]
        shifted = row - float(np.max(row))
        probs = np.exp(shifted / 0.12)
        probs = probs / (float(np.sum(probs)) + 1e-9)
        candidates = []
        for rank, template_index in enumerate(order, start=1):
            chord = names[int(template_index)]
            candidates.append(
                {
                    "rank": rank,
                    "chord": chord,
                    "roman": chord_to_roman_display(chord, key, "N", 0.0, 1.0),
                    "score": float(row[int(template_index)]),
                    "confidence": float(probs[int(template_index)]),
                }
            )
        bass_root = bass_roots[index] if index < len(bass_roots) else None
        if bass_root is None:
            bass_name = "N"
        elif isinstance(bass_root, str):
            bass_name = bass_root
        else:
            bass_name = note_name_for_pc(int(bass_root), key)
        for candidate in candidates:
            candidate["bass_root"] = bass_name
            candidate["bass_confidence"] = float(bass_confidence[index]) if index < len(bass_confidence) else 0.0
        result[index] = candidates
    return result


def diagnose(args: argparse.Namespace) -> list[dict[str, Any]]:
    corrected = load_json(resolve_path(args.corrected))
    automatic = load_json(resolve_path(args.automatic))
    corrected_spans = iter_spans(corrected)
    automatic_spans = iter_spans(automatic)
    key = str(args.key or corrected.get("key") or automatic.get("key") or "C")
    acoustic = acoustic_candidates(
        resolve_path(args.audio),
        corrected_spans,
        key=key,
        mode=args.mode,
        notation=args.notation,
        feature_mode=args.feature_mode,
        hop_length=args.hop_length,
        top_n=args.top_n,
        bass_weight=args.bass_weight,
    )
    functional_prior = read_prior_scores(args.functional_prior)
    extended_prior = read_prior_scores(args.extended_prior)

    rows: list[dict[str, Any]] = []
    previous_corrected = None
    for index, span in enumerate(corrected_spans):
        auto, auto_overlap = best_cover(span, automatic_spans)
        ref_chord = label(span, "chord")
        ref_roman = label(span, "roman")
        auto_chord = label(auto, "chord")
        auto_roman = label(auto, "roman")
        top = acoustic.get(index, [])
        top_chords = " | ".join(
            f"{item['rank']}:{item['chord']}:{item['roman']}:{item['confidence']:.3f}" for item in top
        )
        ref_rank = next((item["rank"] for item in top if item["chord"] == ref_chord), "")
        ref_root_rank = next((item["rank"] for item in top if root_name(item["chord"]) == root_name(ref_chord)), "")
        auto_rank = next((item["rank"] for item in top if item["chord"] == auto_chord), "")
        top1 = top[0] if top else {}
        prev_roman = label(previous_corrected, "roman")
        prev_chord = label(previous_corrected, "chord")
        rows.append(
            {
                "bar": span.get("bar", ""),
                "beat_start": span.get("beat_start", ""),
                "beat_end": span.get("beat_end", ""),
                "start": f"{float(span.get('start', 0.0)):.3f}",
                "end": f"{float(span.get('end', 0.0)):.3f}",
                "duration": f"{float(span.get('end', 0.0)) - float(span.get('start', 0.0)):.3f}",
                "corrected_chord": ref_chord,
                "corrected_roman": ref_roman,
                "automatic_chord": auto_chord,
                "automatic_roman": auto_roman,
                "automatic_overlap": f"{auto_overlap:.3f}",
                "auto_matches_corrected": int(auto_chord == ref_chord),
                "auto_same_root": int(root_name(auto_chord) == root_name(ref_chord) and ref_chord != "N"),
                "acoustic_top1_chord": top1.get("chord", ""),
                "acoustic_top1_roman": top1.get("roman", ""),
                "acoustic_top1_confidence": f"{float(top1.get('confidence', 0.0)):.6f}" if top1 else "",
                "corrected_in_topn_rank": ref_rank,
                "corrected_root_in_topn_rank": ref_root_rank,
                "automatic_in_topn_rank": auto_rank,
                "topn": top_chords,
                "bass_root": top1.get("bass_root", ""),
                "bass_confidence": f"{float(top1.get('bass_confidence', 0.0)):.6f}" if top1 else "",
                "prev_corrected_chord": prev_chord,
                "prev_corrected_roman": prev_roman,
                "corrected_root_motion": root_motion(prev_chord, ref_chord),
                "functional_prior_key": prior_key(prev_roman, ref_roman, extended=False),
                "functional_prior_score": f"{functional_prior.get(prior_key(prev_roman, ref_roman, extended=False), 0.0):.6f}",
                "extended_prior_key": prior_key(prev_roman, ref_roman, extended=True),
                "extended_prior_score": f"{extended_prior.get(prior_key(prev_roman, ref_roman, extended=True), 0.0):.6f}",
                "edited": int(bool(span.get("edited"))),
            }
        )
        previous_corrected = span
    return rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--functional-prior", type=Path)
    parser.add_argument("--extended-prior", type=Path)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="rich")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="flats")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--bass-weight", type=float, default=0.55)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    rows = diagnose(args)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    mismatches = [row for row in rows if row["auto_matches_corrected"] == 0]
    corrected_in_topn = [row for row in mismatches if row["corrected_in_topn_rank"] != ""]
    corrected_root_in_topn = [row for row in mismatches if row["corrected_root_in_topn_rank"] != ""]
    print(f"Wrote {len(rows)} diagnostic row(s): {output}")
    print(f"mismatched rows: {len(mismatches)}")
    print(f"correct exact chord in top-N among mismatches: {len(corrected_in_topn)}")
    print(f"correct root in top-N among mismatches: {len(corrected_root_in_topn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
