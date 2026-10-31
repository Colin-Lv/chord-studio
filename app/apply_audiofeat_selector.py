#!/usr/bin/env python
"""Apply the trained audio-evidence candidate selector to an analysis JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from augment_candidate_audio_features import add_features, interval_for_row, song_features
from build_candidate_training_set import row_for_candidate
from build_topn_candidate_training_set import build_rows, spans_from_payload
from chord_studio import SAMPLE_RATE, chord_to_roman, chord_to_roman_display, write_html
from evaluate_candidate_reranker import group_key, load_model
from refine_extensions import sync_beats_to_spans, sync_events_to_spans
from train_candidate_reranker import load_rows


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def current_candidate_rows(song_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    spans = spans_from_payload(payload)
    for span in spans:
        row = row_for_candidate(song_id, "current", span, span, float(span.get("end", 0.0)) - float(span.get("start", 0.0)))
        row["candidate_rank"] = 0
        row["candidate_score"] = span.get("confidence", 1.0)
        rows.append(row)
    return rows


def add_audio_features(rows: list[dict[str, Any]], audio: Path, feature_mode: str, hop_length: int) -> list[dict[str, Any]]:
    y, chroma, bass = song_features(str(audio), hop_length, feature_mode)
    output = []
    for row in rows:
        item = dict(row)
        interval = interval_for_row({key: str(value) for key, value in item.items()})
        from chord_studio import chroma_for_intervals

        vector = chroma_for_intervals(chroma, y, SAMPLE_RATE, hop_length, [interval])[0]
        frame_count = bass.shape[1]
        start_frame = max(0, min(frame_count - 1, int(interval.start * SAMPLE_RATE / hop_length)))
        end_frame = max(start_frame + 1, min(frame_count, int(interval.end * SAMPLE_RATE / hop_length + 0.999)))
        import numpy as np

        bass_slice = bass[:, start_frame:end_frame]
        bass_vector = np.median(bass_slice, axis=1) if bass_slice.size else np.zeros(12, dtype=float)
        peak = float(np.max(bass_vector))
        if peak > 0:
            bass_vector = bass_vector / peak
        output.append(add_features(item, vector, bass_vector))
    return output


def filter_same_root_alternatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_roots: dict[tuple[int, float, float], str] = {}
    for row in rows:
        if str(row.get("candidate_source") or "") != "current":
            continue
        current_roots[span_key(row)] = str(row.get("candidate_chord_root") or "")
    if not current_roots:
        return rows
    output = []
    for row in rows:
        if str(row.get("candidate_source") or "") == "current":
            output.append(row)
            continue
        current_root = current_roots.get(span_key(row))
        candidate_root = str(row.get("candidate_chord_root") or "")
        if current_root and candidate_root == current_root:
            output.append(row)
    return output


def select_rows(model_path: Path, candidate_csv: Path, target: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pipeline, model_target = load_model(model_path)
    target = target or model_target
    df = load_rows(candidate_csv)
    scores = pipeline.predict_proba(df)[:, 1]
    df = df.copy()
    df["selector_score"] = scores
    df["group_key"] = [group_key(row) for _, row in df.iterrows()]
    selected = []
    for _, group in df.groupby("group_key", sort=False):
        best = group.sort_values(["selector_score", "candidate_score"], ascending=False).iloc[0]
        selected.append(best.to_dict())
    report = {
        "model": str(model_path),
        "candidate_csv": str(candidate_csv),
        "target": target,
        "groups": len(selected),
    }
    return selected, report


def span_key(item: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(float(item.get("bar") or 0)),
        round(float(item.get("start") or item.get("reference_start") or 0.0), 3),
        round(float(item.get("end") or item.get("reference_end") or 0.0), 3),
    )


def apply_selected(payload: dict[str, Any], selected: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_key = {span_key(row): row for row in selected}
    changes = []
    for bar in payload.get("bars", []) or []:
        for span in bar.get("spans") or []:
            key = span_key({**span, "bar": bar.get("bar", span.get("bar", 0))})
            row = by_key.get(key)
            if row is None:
                continue
            old_chord = str(span.get("chord_display") or span.get("chord") or "N")
            new_chord = str(row.get("candidate_chord") or old_chord)
            local_key = str(span.get("local_key") or row.get("local_key") or payload.get("key") or "C")
            span["chord"] = new_chord
            span["chord_display"] = new_chord
            span["roman"] = chord_to_roman(new_chord, local_key)
            span["roman_display"] = chord_to_roman_display(new_chord, local_key)
            span["confidence"] = round(float(row.get("selector_score") or span.get("confidence") or 1.0), 6)
            span["selector_refined"] = True
            span["selector_refinement"] = {
                "from": old_chord,
                "to": new_chord,
                "source": str(row.get("candidate_source") or ""),
                "score": round(float(row.get("selector_score") or 0.0), 6),
                "candidate_score": round(float(row.get("candidate_score") or 0.0), 6),
                "audio_flat7_over_triad": round(float(row.get("audio_flat7_over_triad") or 0.0), 6),
                "audio_maj7_over_triad": round(float(row.get("audio_maj7_over_triad") or 0.0), 6),
                "audio_9_over_triad": round(float(row.get("audio_9_over_triad") or 0.0), 6),
                "audio_bass_root_energy": round(float(row.get("audio_bass_root_energy") or 0.0), 6),
            }
            if old_chord != new_chord:
                changes.append(
                    {
                        "bar": bar.get("bar"),
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "from": old_chord,
                        "to": new_chord,
                        **span["selector_refinement"],
                    }
                )
    for bar in payload.get("bars", []) or []:
        sync_beats_to_spans(bar)
    sync_events_to_spans(payload)
    payload["selector_refinements"] = changes
    return payload, changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("benchmarks/models/user5_rich_scale_degree_audiofeat_exact_rf.pkl"))
    parser.add_argument("--song-id", default="user_song")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--mode", choices=["triads", "seventh", "rich"], default="rich")
    parser.add_argument("--notation", choices=["sharps", "flats"], default="sharps")
    parser.add_argument("--feature-mode", choices=["cqt", "fast"], default="fast")
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--bass-weight", type=float, default=0.55)
    parser.add_argument("--no-current", action="store_true", help="Do not include the original analysis span as a candidate.")
    parser.add_argument(
        "--same-root-alternatives",
        action="store_true",
        help="When current is included, only let alternatives with the same chord root compete.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--candidates-output", type=Path)
    parser.add_argument("--audiofeat-output", type=Path)
    parser.add_argument("--changes-output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_path = resolve_path(args.analysis)
    audio = resolve_path(args.audio)
    payload = load_json(analysis_path)
    topn_rows = build_rows(
        song_id=args.song_id,
        audio=audio,
        reference_payload=payload,
        segment_payload=payload,
        candidate_source="audio_topn",
        top_n=args.top_n,
        mode=args.mode,
        notation=args.notation,
        feature_mode=args.feature_mode,
        hop_length=args.hop_length,
        bass_weight=args.bass_weight,
        progressions=None,
        progression_strength=0.0,
        simplify_short=True,
        simplify_max_duration=0.55,
    )
    rows = ([] if args.no_current else current_candidate_rows(args.song_id, payload)) + topn_rows
    if args.same_root_alternatives and not args.no_current:
        rows = filter_same_root_alternatives(rows)
    candidates_output = resolve_path(args.candidates_output or args.output.with_name(args.output.stem + "_candidates.csv"))
    write_csv(candidates_output, rows)
    audio_rows = add_audio_features(rows, audio, args.feature_mode, args.hop_length)
    audiofeat_output = resolve_path(args.audiofeat_output or args.output.with_name(args.output.stem + "_audiofeat_candidates.csv"))
    write_csv(audiofeat_output, audio_rows)
    selected, report = select_rows(resolve_path(args.model), audiofeat_output, None)
    refined, changes = apply_selected(payload, selected)
    refined["audiofeat_selector"] = {
        **report,
        "top_n": args.top_n,
        "mode": args.mode,
        "feature_mode": args.feature_mode,
        "include_current": not args.no_current,
        "same_root_alternatives": args.same_root_alternatives,
        "changes": len(changes),
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        write_html(resolve_path(args.html_output), refined)
    if args.changes_output:
        write_csv(resolve_path(args.changes_output), changes)
    if args.report:
        report_path = resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(refined["audiofeat_selector"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output} with {len(changes)} selector change(s)")
    if args.html_output:
        print(f"Wrote {resolve_path(args.html_output)}")
    print(f"Candidates: {audiofeat_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
