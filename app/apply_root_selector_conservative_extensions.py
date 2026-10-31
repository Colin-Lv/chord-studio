#!/usr/bin/env python
"""Apply a roman-root selector, then keep conservative chord spellings.

The first stage chooses the most likely key-relative function/root.  The second
stage prefers the simplest chord in that function/root and only keeps an
extension when it is explicitly enabled and supported by strong evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from chord_studio import chord_to_roman, chord_to_roman_display, write_html
from refine_extensions import sync_beats_to_spans, sync_events_to_spans


REPO_ROOT = Path(__file__).resolve().parent.parent


SIMPLE_SUFFIXES = {"", "m", "dim", "aug", "sus4", "sus2"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def roman_root(roman: str) -> str:
    value = str(roman or "N").strip()
    if value in {"", "N"}:
        return "N"
    value = value.split("/")[0]
    value = re.sub(r"(maj7|M7|m7b5|m7|dim7|sus4|sus2|add9|9|7|6|m)$", "", value)
    return value or "N"


def chord_suffix(chord: str) -> str:
    value = str(chord or "N").strip()
    if value in {"", "N"}:
        return "N"
    value = value.split("/")[0]
    match = re.match(r"^[A-G](?:#|b)?(.*)$", value)
    return match.group(1) if match else ""


def simplify_chord(chord: str) -> str:
    value = str(chord or "N").strip()
    if value in {"", "N"}:
        return "N"
    bass = ""
    body = value
    if "/" in value:
        body, bass = value.split("/", 1)
        bass = "/" + bass
    match = re.match(r"^([A-G](?:#|b)?)(.*)$", body)
    if not match:
        return value
    root, suffix = match.groups()
    if suffix == "m7b5":
        return root + "m7b5" + bass
    if suffix.startswith("m") and not suffix.startswith("maj"):
        return root + "m" + bass
    if suffix in {"dim", "aug", "sus4", "sus2"}:
        return root + suffix + bass
    return root + bass


def is_simple_chord(chord: str) -> bool:
    return chord_suffix(chord) in SIMPLE_SUFFIXES


def complexity_penalty(chord: str) -> int:
    suffix = chord_suffix(chord)
    if suffix in {"", "m"}:
        return 0
    if suffix in {"dim", "aug", "sus4", "sus2"}:
        return 1
    if suffix in {"7", "m7", "maj7", "6"}:
        return 2
    return 3


def span_key(bar: int, start: float, end: float) -> tuple[int, float, float]:
    return (int(bar), round(float(start), 3), round(float(end), 3))


def containing_span_key(
    keys: list[tuple[int, float, float]],
    bar: int,
    start: float,
    end: float,
) -> tuple[int, float, float] | None:
    """Find a wider candidate interval that contains the current span midpoint."""
    midpoint = (float(start) + float(end)) / 2.0
    matches = [
        key
        for key in keys
        if key[0] == int(bar) and key[1] - 0.03 <= midpoint <= key[2] + 0.03
    ]
    if not matches:
        return None
    return min(matches, key=lambda key: (key[2] - key[1], abs(((key[1] + key[2]) / 2.0) - midpoint)))


def load_selected_roots(path: Path) -> dict[tuple[int, float, float], str]:
    selected: dict[tuple[int, float, float], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = span_key(int(float(row.get("bar") or 0)), float(row.get("reference_start") or 0.0), float(row.get("reference_end") or 0.0))
            selected[key] = roman_root(row.get("candidate_roman") or "")
    return selected


def load_candidates(path: Path) -> dict[tuple[int, float, float], list[dict[str, str]]]:
    groups: dict[tuple[int, float, float], list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = span_key(int(float(row.get("bar") or 0)), float(row.get("reference_start") or 0.0), float(row.get("reference_end") or 0.0))
            groups.setdefault(key, []).append(row)
    return groups


def score_candidate(row: dict[str, str], allow_extensions: bool, extension_margin: float) -> tuple[int, float, float]:
    chord = str(row.get("candidate_chord") or "N")
    confidence = float(row.get("candidate_confidence") or row.get("candidate_score") or 0.0)
    rank = float(row.get("candidate_rank") or 999.0)
    is_simple = is_simple_chord(chord)
    if not allow_extensions and not is_simple:
        return (-100, -complexity_penalty(chord), -rank)
    if not is_simple:
        flat7 = float(row.get("audio_flat7_over_triad") or 0.0)
        maj7 = float(row.get("audio_maj7_over_triad") or 0.0)
        ninth = float(row.get("audio_9_over_triad") or 0.0)
        if max(flat7, maj7, ninth) < extension_margin:
            return (-10, -complexity_penalty(chord), -rank)
    return (1 if is_simple else 0, confidence - complexity_penalty(chord) * 0.08, -rank)


def choose_chromatic_approach7(
    group: list[dict[str, str]],
    selected_root: str,
    enable: bool,
    min_root_energy: float,
    max_maj7_energy: float,
) -> dict[str, str] | None:
    if not enable or selected_root not in {"V", "VI"}:
        return None
    candidates = [
        row
        for row in group
        if roman_root(row.get("candidate_roman") or "") == "bVI"
        and str(row.get("candidate_roman") or "").endswith("7")
        and str(row.get("harm_candidate_expected_resolution") or "") == "VIm"
        and float(row.get("audio_root_energy") or 0.0) >= min_root_energy
        and float(row.get("audio_maj7_energy") or 0.0) <= max_maj7_energy
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.get("harm_candidate_is_functional_dom7") or 0.0),
            float(row.get("harm_candidate_is_chromatic_to_key") or 0.0),
            float(row.get("candidate_confidence") or 0.0),
            -float(row.get("candidate_rank") or 999.0),
        ),
    )


def choose_candidate(
    group: list[dict[str, str]],
    selected_root: str,
    current_chord: str,
    allow_extensions: bool,
    extension_margin: float,
    enable_chromatic_approach7: bool,
    chromatic_min_root_energy: float,
    chromatic_max_maj7_energy: float,
) -> dict[str, str] | None:
    chromatic = choose_chromatic_approach7(
        group,
        selected_root,
        enable_chromatic_approach7,
        chromatic_min_root_energy,
        chromatic_max_maj7_energy,
    )
    if chromatic is not None:
        return chromatic
    same_root = [row for row in group if roman_root(row.get("candidate_roman") or "") == selected_root]
    if not same_root:
        return None
    current_same = [
        row
        for row in same_root
        if str(row.get("candidate_chord") or "") == current_chord and (allow_extensions or is_simple_chord(current_chord))
    ]
    simple = [row for row in same_root if is_simple_chord(str(row.get("candidate_chord") or ""))]
    pool = current_same or simple or same_root
    return max(pool, key=lambda row: score_candidate(row, allow_extensions, extension_margin))


def apply(
    payload: dict[str, Any],
    selected_roots: dict[tuple[int, float, float], str],
    candidate_groups: dict[tuple[int, float, float], list[dict[str, str]]],
    allow_extensions: bool,
    extension_margin: float,
    enable_chromatic_approach7: bool,
    chromatic_min_root_energy: float,
    chromatic_max_maj7_energy: float,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    selected_keys = list(selected_roots.keys())
    candidate_keys = list(candidate_groups.keys())
    for bar in payload.get("bars", []) or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        for span in bar.get("spans") or []:
            key = span_key(bar_number, float(span.get("start") or 0.0), float(span.get("end") or 0.0))
            selected_key = key if key in selected_roots else containing_span_key(
                selected_keys,
                bar_number,
                float(span.get("start") or 0.0),
                float(span.get("end") or 0.0),
            )
            candidate_key = key if key in candidate_groups else containing_span_key(
                candidate_keys,
                bar_number,
                float(span.get("start") or 0.0),
                float(span.get("end") or 0.0),
            )
            selected_root = selected_roots.get(selected_key) if selected_key else None
            group = candidate_groups.get(candidate_key) if candidate_key else []
            if not selected_root or not group:
                continue
            old_chord = str(span.get("chord_display") or span.get("chord") or "N")
            row = choose_candidate(
                group,
                selected_root,
                old_chord,
                allow_extensions,
                extension_margin,
                enable_chromatic_approach7,
                chromatic_min_root_energy,
                chromatic_max_maj7_energy,
            )
            if row is None:
                continue
            new_chord = str(row.get("candidate_chord") or old_chord)
            if not allow_extensions and not str(row.get("candidate_roman") or "").endswith("7"):
                new_chord = simplify_chord(new_chord)
            local_key = str(span.get("local_key") or row.get("local_key") or payload.get("key") or "C")
            span["chord"] = new_chord
            span["chord_display"] = new_chord
            span["roman"] = chord_to_roman(new_chord, local_key)
            span["roman_display"] = chord_to_roman_display(new_chord, local_key)
            span["root_selector_refined"] = True
            span["root_selector_refinement"] = {
                "from": old_chord,
                "to": new_chord,
                "selected_roman_root": selected_root,
                "candidate_source": row.get("candidate_source"),
                "candidate_score": row.get("candidate_score"),
                "mode": "conservative_extensions" if allow_extensions else "simple_root",
            }
            if old_chord != new_chord:
                changes.append(
                    {
                        "bar": bar_number,
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "from": old_chord,
                        "to": new_chord,
                        "selected_roman_root": selected_root,
                        "candidate_source": row.get("candidate_source"),
                    }
                )
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["root_selector_refinements"] = changes
    return changes


def write_changes(path: Path, changes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in changes for key in row.keys()}) if changes else ["bar", "start", "end", "from", "to"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--selected-roots", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--allow-extensions", action="store_true")
    parser.add_argument("--enable-chromatic-approach7", action="store_true")
    parser.add_argument("--chromatic-min-root-energy", type=float, default=0.10)
    parser.add_argument("--chromatic-max-maj7-energy", type=float, default=0.26)
    parser.add_argument("--extension-margin", type=float, default=0.12)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(resolve_path(args.analysis).read_text(encoding="utf-8"))
    changes = apply(
        payload,
        load_selected_roots(resolve_path(args.selected_roots)),
        load_candidates(resolve_path(args.candidates)),
        args.allow_extensions,
        args.extension_margin,
        args.enable_chromatic_approach7,
        args.chromatic_min_root_energy,
        args.chromatic_max_maj7_energy,
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        write_html(resolve_path(args.html_output), payload)
    if args.changes_output:
        write_changes(resolve_path(args.changes_output), changes)
    print(f"Wrote {output} with {len(changes)} change(s)")
    if args.html_output:
        print(f"Wrote {resolve_path(args.html_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
