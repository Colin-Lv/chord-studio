#!/usr/bin/env python
"""Build training rows from segment path proposals and corrected analyses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_candidate_training_set import best_reference_span, iter_spans, label_of
from classify_boundary_segments import roman_root
from chord_studio import parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chord_root(chord: str) -> str:
    root, _ = parse_chord_name(chord)
    return "" if root is None else str(root)


def path_root_changes(chords: list[str]) -> int:
    roots = [chord_root(chord) for chord in chords if chord != "N"]
    return sum(1 for left, right in zip(roots, roots[1:]) if left and right and left != right)


def duration(span: dict[str, Any]) -> float:
    return max(float(span.get("end", 0.0)) - float(span.get("start", 0.0)), 0.0)


def evaluate_path(spans: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, float]:
    total = sum(duration(span) for span in spans)
    if total <= 1e-9:
        return {"overlap": 0.0, "roman_root": 0.0, "exact": 0.0}
    overlap = 0.0
    roman_root_score = 0.0
    exact_score = 0.0
    for span in spans:
        reference, amount = best_reference_span(span, references)
        overlap += amount
        if reference is None:
            continue
        chord = label_of(span, "chord")
        roman = label_of(span, "roman")
        ref_chord = label_of(reference, "chord")
        ref_roman = label_of(reference, "roman")
        if roman_root(roman) == roman_root(ref_roman) and ref_roman != "N":
            roman_root_score += amount
        if chord == ref_chord and ref_chord != "N":
            exact_score += amount
    return {
        "overlap": overlap / total,
        "roman_root": roman_root_score / total,
        "exact": exact_score / total,
    }


def rows_for_pair(song_id: str, proposals_path: Path, reference_path: Path, min_accept_delta: float) -> list[dict[str, Any]]:
    proposals = load_json(proposals_path)
    reference = load_json(reference_path)
    references = iter_spans(reference)
    rows: list[dict[str, Any]] = []
    for bar_item in proposals.get("proposals") or []:
        current_path = next((item for item in bar_item.get("paths") or [] if item.get("name") == "current"), None)
        current_metrics = evaluate_path(current_path.get("spans") or [], references) if current_path else {
            "roman_root": 0.0,
            "exact": 0.0,
            "overlap": 0.0,
        }
        best_roman = current_metrics["roman_root"]
        best_exact = current_metrics["exact"]
        path_metrics: list[tuple[dict[str, Any], dict[str, float]]] = []
        for path in bar_item.get("paths") or []:
            metrics = evaluate_path(path.get("spans") or [], references)
            path_metrics.append((path, metrics))
            best_roman = max(best_roman, metrics["roman_root"])
            best_exact = max(best_exact, metrics["exact"])

        for path, metrics in path_metrics:
            chords = [str(item) for item in path.get("chords") or []]
            roots = [chord_root(chord) for chord in chords if chord != "N"]
            score = float(path.get("score") or 0.0)
            current_score = bar_item.get("current_score")
            current_score_value = float(current_score) if current_score is not None else score
            gain = score - current_score_value
            roman_gain = metrics["roman_root"] - current_metrics["roman_root"]
            exact_gain = metrics["exact"] - current_metrics["exact"]
            rows.append(
                {
                    "song_id": song_id,
                    "bar": int(bar_item.get("bar", 0)),
                    "path_name": path.get("name", ""),
                    "path_score": f"{score:.6f}",
                    "gain_over_current_score": f"{gain:.6f}",
                    "boundary_score": path.get("boundary_score", 0.0),
                    "span_count": int(path.get("span_count") or len(path.get("spans") or [])),
                    "boundary_count": len(bar_item.get("boundary_candidates") or []),
                    "avg_confidence": path.get("avg_confidence", 0.0),
                    "avg_margin": path.get("avg_margin", 0.0),
                    "root_change_count": path_root_changes(chords),
                    "distinct_root_count": len(set(root for root in roots if root)),
                    "contains_n": int(any(chord == "N" for chord in chords)),
                    "meaningful_root_change": int(bool(bar_item.get("meaningful_root_change")) and path.get("name") == bar_item.get("best_name")),
                    "chords": " | ".join(chords),
                    "roman": " | ".join(str(item) for item in path.get("roman") or []),
                    "current_roman_root_score": f"{current_metrics['roman_root']:.6f}",
                    "current_exact_score": f"{current_metrics['exact']:.6f}",
                    "path_roman_root_score": f"{metrics['roman_root']:.6f}",
                    "path_exact_score": f"{metrics['exact']:.6f}",
                    "roman_root_gain": f"{roman_gain:.6f}",
                    "exact_gain": f"{exact_gain:.6f}",
                    "target_best_roman_root": int(metrics["roman_root"] >= best_roman - 1e-6),
                    "target_best_exact": int(metrics["exact"] >= best_exact - 1e-6),
                    "target_accept_roman_root": int(roman_gain >= min_accept_delta),
                    "target_accept_exact": int(exact_gain >= min_accept_delta),
                }
            )
    return rows


def parse_pair(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--pair must be song_id=proposal_json:reference_json")
    song_id = parts[0].strip()
    if "|" in parts[1]:
        proposal, reference = parts[1].split("|", 1)
    else:
        proposal, reference = parts[1].split(":", 1)
    return song_id, resolve_path(Path(proposal.strip())), resolve_path(Path(reference.strip()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="song_id=proposal_json|reference_json. A colon is also accepted for relative paths.",
    )
    parser.add_argument("--min-accept-delta", type=float, default=0.12)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for value in args.pair:
        song_id, proposals, reference = parse_pair(value)
        rows.extend(rows_for_pair(song_id, proposals, reference, args.min_accept_delta))
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} segment path row(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
