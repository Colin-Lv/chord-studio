#!/usr/bin/env python
"""Refine functional/line dominant colors such as III -> III7.

This is intentionally narrower than the general extension refiner: it only
upgrades an existing major-triad root to a dominant seventh with the same root
when audio evidence and local context are strong enough.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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
from classify_boundary_segments import roman_root
from refine_extensions import sync_beats_to_spans, sync_events_to_spans


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


def iter_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans = []
    for bar in payload.get("bars", []) or []:
        for span in bar.get("spans") or []:
            span.setdefault("bar", bar.get("bar", 0))
            spans.append(span)
    return spans


def energy(vector: np.ndarray, pc: int) -> float:
    return float(vector[int(pc) % 12])


def load_kb_resolution(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    output: dict[str, dict[str, float]] = {}
    for roman, rows in (payload.get("functional_dom7_resolution", {}) or {}).items():
        output[str(roman)] = {str(item.get("item", "")): float(item.get("prob", 0.0) or 0.0) for item in rows}
    return output


def next_roman_roots(spans: list[dict[str, Any]]) -> list[str]:
    output = []
    for index, _ in enumerate(spans):
        if index + 1 >= len(spans):
            output.append("")
            continue
        next_roman = str(spans[index + 1].get("roman_display") or spans[index + 1].get("roman") or "")
        output.append(roman_root(next_roman))
    return output


def pattern_counts(spans: list[dict[str, Any]]) -> Counter[tuple[str, str, int]]:
    counts: Counter[tuple[str, str, int]] = Counter()
    next_roots = next_roman_roots(spans)
    for span, next_root in zip(spans, next_roots):
        roman = roman_root(str(span.get("roman_display") or span.get("roman") or ""))
        if not roman:
            continue
        bar_mod = int(span.get("bar", 0) or 0) % 4
        counts[(roman, next_root, bar_mod)] += 1
    return counts


def candidate_evidence(chord: str, vector: np.ndarray, key: str) -> dict[str, float | str] | None:
    root, suffix = parse_chord_name(chord)
    if root is None or suffix not in {"", "maj"}:
        return None
    third = (root + 4) % 12
    fifth = (root + 7) % 12
    flat7 = (root + 10) % 12
    maj7 = (root + 11) % 12
    triad_mean = float(np.mean([energy(vector, root), energy(vector, third), energy(vector, fifth)]))
    return {
        "root": note_name_for_pc(root, key),
        "triad_mean": triad_mean,
        "flat7_energy": energy(vector, flat7),
        "maj7_energy": energy(vector, maj7),
        "flat7_over_triad": energy(vector, flat7) - triad_mean,
        "dom7_vs_maj7": energy(vector, flat7) - energy(vector, maj7),
    }


MAJOR_ROMAN_TO_PC = {
    "I": 0,
    "bII": 1,
    "II": 2,
    "bIII": 3,
    "III": 4,
    "IV": 5,
    "#IV": 6,
    "V": 7,
    "bVI": 8,
    "VI": 9,
    "bVII": 10,
    "VII": 11,
}


def chord_for_roman7(key: str, roman: str) -> str | None:
    key_root, _ = parse_chord_name(key or "C")
    offset = MAJOR_ROMAN_TO_PC.get(roman)
    if key_root is None or offset is None:
        return None
    return f"{note_name_for_pc((key_root + offset) % 12, key)}7"


def should_upgrade(
    span: dict[str, Any],
    evidence: dict[str, float | str],
    allowed_roots: set[str],
    next_root: str,
    kb_resolution: dict[str, dict[str, float]],
    repeats: Counter[tuple[str, str, int]],
    min_duration: float,
    flat7_threshold: float,
    strong_flat7_threshold: float,
    min_resolution_prob: float,
    include_edited: bool,
) -> tuple[bool, str]:
    if span.get("edited") and not include_edited:
        return False, "skip_edited"
    duration = float(span.get("end", 0.0)) - float(span.get("start", 0.0))
    if duration < min_duration:
        return False, "too_short"
    roman = roman_root(str(span.get("roman_display") or span.get("roman") or ""))
    if roman not in allowed_roots:
        return False, "roman_not_allowed"
    candidate_roman = roman + "7"
    flat7_over = float(evidence["flat7_over_triad"])
    dom_vs_maj7 = float(evidence["dom7_vs_maj7"])
    if flat7_over < flat7_threshold or dom_vs_maj7 < -0.02:
        return False, "weak_flat7"
    resolution_prob = kb_resolution.get(candidate_roman, {}).get(next_root, 0.0)
    bar_mod = int(span.get("bar", 0) or 0) % 4
    repeated_pattern = repeats[(roman, next_root, bar_mod)] >= 2
    if resolution_prob >= min_resolution_prob:
        return True, "kb_resolution"
    if repeated_pattern and flat7_over >= flat7_threshold + 0.015:
        return True, "repeated_line_pattern"
    if flat7_over >= strong_flat7_threshold and dom_vs_maj7 >= 0.02:
        return True, "strong_flat7"
    return False, "context_not_enough"


def update_span(span: dict[str, Any], new_chord: str, evidence: dict[str, float | str], reason: str, next_root: str) -> None:
    old_chord = str(span.get("chord_display") or span.get("chord") or "N")
    key = str(span.get("local_key") or "")
    span["chord"] = new_chord
    span["chord_display"] = new_chord
    if key:
        span["roman"] = chord_to_roman(new_chord, key)
        span["roman_display"] = chord_to_roman_display(
            new_chord,
            key,
            str(span.get("bass_root") or "N"),
            float(span.get("bass_confidence") or 0.0),
            float(span.get("confidence") or 1.0),
        )
    span["extension_refined"] = True
    span["extension_refinement"] = {
        "from": old_chord,
        "to": new_chord,
        "reason": f"functional line dominant: {reason}",
        "next_roman_root": next_root,
        **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in evidence.items()},
    }


def refine_payload(
    payload: dict[str, Any],
    audio: Path,
    knowledge_base: Path | None,
    allowed_roots: set[str],
    feature_mode: str,
    hop_length: int,
    min_duration: float,
    flat7_threshold: float,
    strong_flat7_threshold: float,
    min_resolution_prob: float,
    include_edited: bool,
    allow_root_substitution: bool,
    substitution_roots: set[str],
    substitution_margin: float,
    override_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    y = load_audio(audio)
    chroma = extract_chroma(y, SAMPLE_RATE, hop_length=hop_length, mode=feature_mode)
    spans = iter_spans(payload)
    intervals = [span_interval(span) for span in spans]
    vectors = chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, intervals) if intervals else []
    next_roots = next_roman_roots(spans)
    repeats = pattern_counts(spans)
    kb_resolution = load_kb_resolution(knowledge_base)
    changes: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for span, vector, next_root in zip(spans, vectors, next_roots):
        chord = str(span.get("chord_display") or span.get("chord") or "N")
        key = override_key or str(span.get("local_key") or payload.get("key") or "")
        base_evidence = candidate_evidence(chord, vector, key)
        if base_evidence is None:
            continue
        evidence = base_evidence
        ok, reason = should_upgrade(
            span,
            evidence,
            allowed_roots,
            next_root,
            kb_resolution,
            repeats,
            min_duration,
            flat7_threshold,
            strong_flat7_threshold,
            min_resolution_prob,
            include_edited,
        )
        current_roman = roman_root(str(span.get("roman_display") or span.get("roman") or ""))
        chosen_chord = chord
        if not ok and allow_root_substitution and not span.get("edited"):
            for candidate_root in substitution_roots:
                candidate = chord_for_roman7(key, candidate_root)
                if not candidate:
                    continue
                candidate_evi = candidate_evidence(candidate.replace("7", ""), vector, key)
                if candidate_evi is None:
                    continue
                probe_span = dict(span)
                probe_span["roman"] = candidate_root
                probe_span["roman_display"] = candidate_root
                probe_ok, probe_reason = should_upgrade(
                    probe_span,
                    candidate_evi,
                    allowed_roots,
                    next_root,
                    kb_resolution,
                    repeats,
                    min_duration,
                    flat7_threshold,
                    strong_flat7_threshold,
                    min_resolution_prob,
                    include_edited,
                )
                root, _ = parse_chord_name(chord)
                candidate_root_pc, _ = parse_chord_name(candidate)
                old_root_energy = 0.0 if root is None else energy(vector, root)
                new_root_energy = 0.0 if candidate_root_pc is None else energy(vector, candidate_root_pc)
                if probe_ok and new_root_energy + substitution_margin >= old_root_energy:
                    ok = True
                    reason = f"root_substitution_{probe_reason}"
                    evidence = candidate_evi
                    chosen_chord = candidate
                    break
        diagnostics.append(
            {
                "bar": span.get("bar"),
                "start": span.get("start"),
                "end": span.get("end"),
                "chord": chord,
                "roman_root": current_roman,
                "next_roman_root": next_root,
                "decision": "upgrade" if ok else "skip",
                "reason": reason,
                **evidence,
            }
        )
        if not ok:
            continue
        new_chord = chosen_chord
        old_chord = chord
        if override_key:
            span["local_key"] = override_key
        update_span(span, new_chord, evidence, reason, next_root)
        changes.append(
            {
                "bar": span.get("bar"),
                "start": span.get("start"),
                "end": span.get("end"),
                "from": old_chord,
                "to": new_chord,
                "roman": span.get("roman_display") or span.get("roman"),
                "reason": reason,
                **span["extension_refinement"],
            }
        )
    for bar in payload.get("bars", []) or []:
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["functional_line_refinements"] = changes
    payload["functional_line_refinement_settings"] = {
        "allowed_roots": sorted(allowed_roots),
        "feature_mode": feature_mode,
        "hop_length": hop_length,
        "min_duration": min_duration,
        "flat7_threshold": flat7_threshold,
        "strong_flat7_threshold": strong_flat7_threshold,
        "min_resolution_prob": min_resolution_prob,
        "knowledge_base": "" if knowledge_base is None else str(resolve_path(knowledge_base)),
        "allow_root_substitution": allow_root_substitution,
        "substitution_roots": sorted(substitution_roots),
        "substitution_margin": substitution_margin,
        "override_key": override_key,
    }
    return payload, changes, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--knowledge-base", type=Path)
    parser.add_argument("--allowed-roots", default="III,I,II,VI,V")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--min-duration", type=float, default=0.75)
    parser.add_argument("--flat7-threshold", type=float, default=0.055)
    parser.add_argument("--strong-flat7-threshold", type=float, default=0.105)
    parser.add_argument("--min-resolution-prob", type=float, default=0.003)
    parser.add_argument("--include-edited", action="store_true")
    parser.add_argument("--allow-root-substitution", action="store_true")
    parser.add_argument("--substitution-roots", default="III")
    parser.add_argument("--substitution-margin", type=float, default=0.06)
    parser.add_argument("--override-key", default="")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(resolve_path(args.analysis).read_text(encoding="utf-8"))
    allowed_roots = {item.strip() for item in args.allowed_roots.split(",") if item.strip()}
    substitution_roots = {item.strip() for item in args.substitution_roots.split(",") if item.strip()}
    refined, changes, diagnostics = refine_payload(
        payload,
        resolve_path(args.audio),
        args.knowledge_base,
        allowed_roots,
        args.feature_mode,
        args.hop_length,
        args.min_duration,
        args.flat7_threshold,
        args.strong_flat7_threshold,
        args.min_resolution_prob,
        args.include_edited,
        args.allow_root_substitution,
        substitution_roots,
        args.substitution_margin,
        args.override_key,
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
    if args.diagnostics_output:
        diagnostics_output = resolve_path(args.diagnostics_output)
        diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in diagnostics for key in row.keys()})
        with diagnostics_output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diagnostics)
    print(f"Wrote {output} with {len(changes)} functional line refinement(s)")
    for change in changes[:20]:
        print(
            f"bar {change.get('bar')}: {change.get('from')} -> {change.get('to')} "
            f"roman={change.get('roman')} reason={change.get('reason')} flat7={change.get('flat7_over_triad')}"
        )
    if len(changes) > 20:
        print(f"... {len(changes) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
