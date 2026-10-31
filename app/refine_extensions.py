#!/usr/bin/env python
"""Conservatively refine triads into 7/maj7/add9 variants using chroma evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_topn_candidate_training_set import load_audio
from chord_recognizer import extract_chroma
from chord_studio import (
    SAMPLE_RATE,
    BeatInterval,
    chord_to_roman,
    chord_to_roman_display,
    chroma_for_intervals,
    note_name_for_pc,
    parse_chord_name,
    write_html,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def span_interval(span: dict[str, Any]) -> BeatInterval:
    return BeatInterval(
        bar=int(span.get("bar", 0)),
        beat=int(span.get("beat_start") or 1),
        start=float(span.get("start", 0.0)),
        end=float(span.get("end", 0.0)),
    )


def energy(vector: np.ndarray, pc: int) -> float:
    return float(vector[int(pc) % 12])


def simple_suffix(chord: str) -> tuple[int | None, str]:
    root, suffix = parse_chord_name(chord)
    if root is None or "/" in chord:
        return root, suffix
    return root, suffix


def candidate_extension(
    chord: str,
    key: str,
    vector: np.ndarray,
    duration: float,
    confidence: float,
    min_duration: float,
    flat7_threshold: float,
    maj7_threshold: float,
    add9_threshold: float,
    enable_dom7: bool,
    enable_maj7: bool,
    enable_add9: bool,
) -> tuple[str, dict[str, float | str]] | None:
    root, suffix = simple_suffix(chord)
    if root is None or chord == "N" or "/" in chord:
        return None
    if suffix not in {"", "m"}:
        return None
    if duration < min_duration or confidence < 0.38:
        return None

    third = (root + (3 if suffix == "m" else 4)) % 12
    fifth = (root + 7) % 12
    flat7 = (root + 10) % 12
    maj7 = (root + 11) % 12
    ninth = (root + 2) % 12
    triad_mean = float(np.mean([energy(vector, root), energy(vector, third), energy(vector, fifth)]))
    flat7_over = energy(vector, flat7) - triad_mean
    maj7_over = energy(vector, maj7) - triad_mean
    ninth_over = energy(vector, ninth) - triad_mean
    dom_vs_maj7 = energy(vector, flat7) - energy(vector, maj7)

    evidence = {
        "root": note_name_for_pc(root, key),
        "triad_mean": triad_mean,
        "flat7_energy": energy(vector, flat7),
        "maj7_energy": energy(vector, maj7),
        "ninth_energy": energy(vector, ninth),
        "flat7_over_triad": flat7_over,
        "maj7_over_triad": maj7_over,
        "ninth_over_triad": ninth_over,
        "dom7_vs_maj7": dom_vs_maj7,
    }

    root_name = note_name_for_pc(root, key)
    if suffix == "m":
        if enable_dom7 and flat7_over >= flat7_threshold and dom_vs_maj7 >= -0.03:
            return f"{root_name}m7", evidence
        if enable_add9 and ninth_over >= add9_threshold and flat7_over < flat7_threshold + 0.02:
            return f"{root_name}madd9", evidence
        return None

    if enable_dom7 and flat7_over >= flat7_threshold and dom_vs_maj7 >= 0.035:
        return f"{root_name}7", evidence
    if enable_maj7 and maj7_over >= maj7_threshold and -dom_vs_maj7 >= 0.035:
        return f"{root_name}maj7", evidence
    if enable_add9 and ninth_over >= add9_threshold and max(flat7_over, maj7_over) < add9_threshold + 0.04:
        return f"{root_name}add9", evidence
    return None


def extension_suggestion(
    chord: str,
    key: str,
    vector: np.ndarray,
    duration: float,
    confidence: float,
    min_duration: float,
) -> dict[str, Any] | None:
    root, suffix = simple_suffix(chord)
    if root is None or chord == "N" or "/" in chord or suffix not in {"", "m"}:
        return None
    third = (root + (3 if suffix == "m" else 4)) % 12
    fifth = (root + 7) % 12
    flat7 = (root + 10) % 12
    maj7 = (root + 11) % 12
    ninth = (root + 2) % 12
    triad_mean = float(np.mean([energy(vector, root), energy(vector, third), energy(vector, fifth)]))
    flat7_over = energy(vector, flat7) - triad_mean
    maj7_over = energy(vector, maj7) - triad_mean
    ninth_over = energy(vector, ninth) - triad_mean
    dom_vs_maj7 = energy(vector, flat7) - energy(vector, maj7)
    root_name = note_name_for_pc(root, key)
    candidates: list[dict[str, Any]] = []
    if suffix == "m":
        candidates.append({"chord": f"{root_name}m7", "kind": "m7", "score": flat7_over + max(0.0, dom_vs_maj7) * 0.25})
        candidates.append({"chord": f"{root_name}madd9", "kind": "madd9", "score": ninth_over})
    else:
        candidates.append({"chord": f"{root_name}7", "kind": "7", "score": flat7_over + max(0.0, dom_vs_maj7) * 0.5})
        candidates.append({"chord": f"{root_name}maj7", "kind": "maj7", "score": maj7_over + max(0.0, -dom_vs_maj7) * 0.5})
        candidates.append({"chord": f"{root_name}add9", "kind": "add9", "score": ninth_over})
    best = max(candidates, key=lambda item: float(item["score"]))
    return {
        "from": chord,
        "suggested": best["chord"],
        "kind": best["kind"],
        "suggestion_score": round(float(best["score"]), 6),
        "eligible": int(duration >= min_duration and confidence >= 0.38),
        "duration": round(float(duration), 6),
        "confidence": round(float(confidence), 6),
        "triad_mean": round(triad_mean, 6),
        "flat7_over_triad": round(flat7_over, 6),
        "maj7_over_triad": round(maj7_over, 6),
        "ninth_over_triad": round(ninth_over, 6),
        "dom7_vs_maj7": round(dom_vs_maj7, 6),
    }


def update_span(span: dict[str, Any], chord: str, evidence: dict[str, float | str], reason: str) -> None:
    key = str(span.get("local_key") or "")
    old_chord = str(span.get("chord_display") or span.get("chord") or "N")
    span["chord"] = chord
    span["chord_display"] = chord
    if key:
        span["roman"] = chord_to_roman(chord, key)
        span["roman_display"] = chord_to_roman_display(
            chord,
            key,
            str(span.get("bass_root") or "N"),
            float(span.get("bass_confidence") or 0.0),
            float(span.get("confidence") or 1.0),
        )
    span["extension_refined"] = True
    span["extension_refinement"] = {
        "from": old_chord,
        "to": chord,
        "reason": reason,
        **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in evidence.items()},
    }


def sync_beats_to_spans(bar: dict[str, Any]) -> None:
    spans = bar.get("spans") or []
    for beat in bar.get("beats") or []:
        midpoint = (float(beat.get("start", 0.0)) + float(beat.get("end", 0.0))) / 2.0
        for span in spans:
            if float(span.get("start", 0.0)) <= midpoint < float(span.get("end", 0.0)) + 1e-6:
                for field in (
                    "chord",
                    "chord_display",
                    "roman",
                    "roman_display",
                    "extension_refined",
                    "extension_refinement",
                ):
                    if field in span:
                        beat[field] = span[field]
                break


def sync_events_to_spans(payload: dict[str, Any]) -> None:
    events = payload.get("events") or []
    spans = [
        span
        for bar in payload.get("bars", [])
        for span in (bar.get("spans") or [])
        if span.get("extension_refined")
    ]
    for event in events:
        for span in spans:
            if (
                abs(float(event.get("start", 0.0)) - float(span.get("start", 0.0))) < 0.02
                and abs(float(event.get("end", 0.0)) - float(span.get("end", 0.0))) < 0.02
            ):
                for field in (
                    "chord",
                    "chord_display",
                    "roman",
                    "roman_display",
                    "extension_refined",
                    "extension_refinement",
                ):
                    if field in span:
                        event[field] = span[field]
                break


def refine_payload(
    payload: dict[str, Any],
    audio_path: Path,
    feature_mode: str,
    hop_length: int,
    min_duration: float,
    flat7_threshold: float,
    maj7_threshold: float,
    add9_threshold: float,
    enable_dom7: bool,
    enable_maj7: bool,
    enable_add9: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    y = load_audio(audio_path)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    spans: list[dict[str, Any]] = []
    for bar in payload.get("bars", []):
        for span in bar.get("spans") or []:
            span.setdefault("bar", bar.get("bar", 0))
            spans.append(span)
    intervals = [span_interval(span) for span in spans]
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals) if intervals else []
    changes: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for span, vector in zip(spans, vectors):
        chord = str(span.get("chord_display") or span.get("chord") or "N")
        key = str(span.get("local_key") or payload.get("key") or "")
        duration = float(span.get("end", 0.0)) - float(span.get("start", 0.0))
        confidence = float(span.get("confidence") or 1.0)
        suggestion = extension_suggestion(chord, key, vector, duration, confidence, min_duration)
        if suggestion:
            suggestion_row = {
                "bar": span.get("bar"),
                "start": span.get("start"),
                "end": span.get("end"),
                "roman": span.get("roman_display") or span.get("roman"),
                **suggestion,
            }
            suggestions.append(suggestion_row)
            span["extension_suggestion"] = suggestion
        result = candidate_extension(
            chord,
            key,
            vector,
            duration,
            confidence,
            min_duration,
            flat7_threshold,
            maj7_threshold,
            add9_threshold,
            enable_dom7,
            enable_maj7,
            enable_add9,
        )
        if result is None:
            continue
        new_chord, evidence = result
        update_span(span, new_chord, evidence, "extension evidence")
        changes.append(
            {
                "bar": span.get("bar"),
                "start": span.get("start"),
                "end": span.get("end"),
                "from": chord,
                "to": new_chord,
                **span["extension_refinement"],
            }
        )
    for bar in payload.get("bars", []):
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["extension_refinements"] = changes
    payload["extension_suggestions"] = suggestions
    payload["extension_refinement_settings"] = {
        "feature_mode": feature_mode,
        "hop_length": hop_length,
        "min_duration": min_duration,
        "flat7_threshold": flat7_threshold,
        "maj7_threshold": maj7_threshold,
        "add9_threshold": add9_threshold,
        "enable_dom7": enable_dom7,
        "enable_maj7": enable_maj7,
        "enable_add9": enable_add9,
    }
    return payload, changes, suggestions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    parser.add_argument("--suggestions-output", type=Path)
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--min-duration", type=float, default=0.85)
    parser.add_argument("--flat7-threshold", type=float, default=0.08)
    parser.add_argument("--maj7-threshold", type=float, default=0.10)
    parser.add_argument("--add9-threshold", type=float, default=0.16)
    parser.add_argument("--disable-dom7", action="store_true")
    parser.add_argument("--enable-maj7", action="store_true")
    parser.add_argument("--enable-add9", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_path = resolve_path(args.analysis)
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    refined, changes, suggestions = refine_payload(
        payload,
        resolve_path(args.audio),
        args.feature_mode,
        args.hop_length,
        args.min_duration,
        args.flat7_threshold,
        args.maj7_threshold,
        args.add9_threshold,
        not args.disable_dom7,
        args.enable_maj7,
        args.enable_add9,
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        write_html(resolve_path(args.html_output), refined)
    if args.changes_output:
        changes_output = resolve_path(args.changes_output)
        changes_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for change in changes for key in change.keys()})
        with changes_output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(changes)
    if args.suggestions_output:
        suggestions_output = resolve_path(args.suggestions_output)
        suggestions_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for suggestion in suggestions for key in suggestion.keys()})
        with suggestions_output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(suggestions)
    print(f"Wrote {output} with {len(changes)} extension refinement(s)")
    print(f"extension suggestions: {len(suggestions)}")
    for change in changes[:20]:
        print(
            f"bar {change.get('bar')}: {change.get('from')} -> {change.get('to')} "
            f"flat7={change.get('flat7_over_triad')} maj7={change.get('maj7_over_triad')} 9={change.get('ninth_over_triad')}"
        )
    if len(changes) > 20:
        print(f"... {len(changes) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
