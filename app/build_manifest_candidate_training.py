#!/usr/bin/env python
"""Build reranker candidate tables from a benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_candidate_training_set import row_for_candidate, best_reference_span
from build_topn_candidate_training_set import build_rows as build_topn_rows
from lab_to_reference_analysis import convert as lab_to_analysis
from evaluate_chords import read_reference


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = REPO_ROOT / "benchmarks"


def resolve_bench(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else BENCH_ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def spans_from_payload(payload: dict) -> list[dict]:
    spans = []
    for bar in payload.get("bars", []):
        for span in bar.get("spans") or []:
            item = dict(span)
            item["bar"] = int(bar.get("bar", item.get("bar", 0)))
            spans.append(item)
    return sorted(spans, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))


def base_candidate_rows(song_id: str, reference_payload: dict, prediction_payload: dict) -> list[dict]:
    references = spans_from_payload(reference_payload)
    rows = []
    for candidate in spans_from_payload(prediction_payload):
        reference, amount = best_reference_span(candidate, references)
        rows.append(row_for_candidate(song_id, "base_prediction", candidate, reference, amount))
    return rows


def rows_from_manifest(
    manifest: Path,
    reference_dir: Path,
    top_n: int,
    limit: int | None,
    include_topn: bool,
    include_base: bool,
) -> list[dict]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            if limit is not None and row_index > limit:
                break
            song_id = row["id"]
            audio = resolve_bench(row["audio"])
            prediction = resolve_bench(row["prediction"])
            chord_reference = resolve_bench(row["chord_reference"])
            key = str(row.get("key") or "").strip() or "C"
            if not audio.exists() or not prediction.exists() or not chord_reference.exists():
                print(f"skip missing: {song_id}")
                continue
            prediction_payload = load_json(prediction)
            reference_path = reference_dir / f"{song_id}_reference_analysis.json"
            if reference_path.exists():
                reference_payload = load_json(reference_path)
            else:
                reference_payload = lab_to_analysis(prediction_payload, read_reference(chord_reference), key)
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                reference_path.write_text(json.dumps(reference_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if include_base:
                rows.extend(base_candidate_rows(song_id, reference_payload, prediction_payload))
            if include_topn:
                rows.extend(
                    build_topn_rows(
                        song_id=song_id,
                        audio=audio,
                        reference_payload=reference_payload,
                        segment_payload=prediction_payload,
                        candidate_source="beatles_topn",
                        top_n=top_n,
                        mode="rich",
                        notation="sharps",
                        feature_mode="cqt",
                        hop_length=1024,
                        bass_weight=0.55,
                        progressions=None,
                        progression_strength=0.0,
                        simplify_short=True,
                        simplify_max_duration=0.55,
                    )
                )
            print(f"{song_id}: rows={len(rows)}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=BENCH_ROOT / "references" / "isophonics")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-topn", action="store_true")
    parser.add_argument("--no-base", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    reference_dir = args.reference_dir if args.reference_dir.is_absolute() else REPO_ROOT / args.reference_dir
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    rows = rows_from_manifest(
        manifest,
        reference_dir,
        top_n=args.top_n,
        limit=args.limit,
        include_topn=not args.no_topn,
        include_base=not args.no_base,
    )
    write_rows(output, rows)
    print(f"Wrote {len(rows)} row(s): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
