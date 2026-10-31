"""Build reference-free listening intervals from two complete chord timelines."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

import mir_eval

from evaluate_chords import harte_chord


def event_rows(payload: dict) -> list[tuple[float, float, str]]:
    duration = float(payload["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration")
    rows = sorted((float(e["start"]), float(e["end"]), str(e.get("chord_display") or e["chord"]))
                  for e in payload["events"])
    for index, (start, end, _) in enumerate(rows):
        if not math.isfinite(start + end) or start < 0 or end <= start or end > duration + .001:
            raise ValueError("Invalid event interval")
        if index and start < rows[index - 1][1] - 1e-6:
            raise ValueError("Overlapping events require manual resolution before comparison")
    return rows


def label_signature(label: str | None) -> tuple:
    if label is None:
        return ("missing",)
    normalized = harte_chord(label)
    if normalized in {"N", "X"}:
        return (normalized,) if label.strip().upper() == normalized else ("unsupported", label)
    root, bitmap, _ = mir_eval.chord.encode(normalized.partition("/")[0], reduce_extended_chords=True)
    _, _, bass = mir_eval.chord.encode(normalized, reduce_extended_chords=True)
    return ("chord", int(root), tuple(int(x) for x in bitmap), int(bass))


def difference_kind(a: str | None, b: str | None) -> str | None:
    left, right = label_signature(a), label_signature(b)
    if left == right:
        return None
    if left[0] != "chord" or right[0] != "chord":
        return "label" if "unsupported" in (left[0], right[0]) else "coverage"
    if left[1] != right[1]:
        return "root"
    if left[2] != right[2]:
        return "quality"
    return "bass"


def compare_timelines(primary: dict, comparison: dict, minimum: float = .5) -> dict:
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("Invalid minimum display duration")
    offsets = [float(item.get("audio_offset", 0)) + float(item.get("original_offset_seconds", 0))
               for item in (primary, comparison)]
    if not all(math.isfinite(offset) for offset in offsets) or abs(offsets[0] - offsets[1]) > 1e-6:
        raise ValueError("Analysis windows have different offsets")
    if abs(float(primary["duration"]) - float(comparison["duration"])) > .05:
        raise ValueError("Analysis windows have different durations")
    duration = float(primary["duration"])
    first, second = event_rows(primary), event_rows(comparison)
    starts = [[row[0] for row in rows] for rows in (first, second)]
    points = sorted({0., duration, *(min(duration, t) for row in first + second for t in row[:2])})

    def active(rows, times, moment):
        index = bisect_right(times, moment) - 1
        return rows[index][2] if index >= 0 and moment < rows[index][1] else None

    intervals = []
    for start, end in zip(points[:-1], points[1:]):
        if end - start <= 1e-9:
            continue
        a = active(first, starts[0], (start + end) / 2)
        b = active(second, starts[1], (start + end) / 2)
        kind = difference_kind(a, b)
        if kind is None:
            continue
        item = dict(start=start, end=end, primary_label=a, comparison_label=b, kind=kind)
        if intervals and all(intervals[-1][key] == item[key] for key in ("primary_label", "comparison_label", "kind")) and abs(intervals[-1]["end"] - start) < 1e-9:
            intervals[-1]["end"] = end
        else:
            intervals.append(item)
    return dict(duration=duration, all_segments=intervals,
                segments=[item for item in intervals if item["end"] - item["start"] >= minimum],
                total_count=len(intervals), display_min_duration=minimum,
                disagreement_seconds=sum(item["end"] - item["start"] for item in intervals))


def signature(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build(primary_path: Path, comparison_path: Path, minimum: float) -> dict:
    primary = json.loads(primary_path.read_text(encoding="utf-8-sig"))
    other = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
    demos = {row["id"]: row for row in other["demos"]}
    if len(demos) != len(other["demos"]) or len({row["id"] for row in primary["songs"]}) != len(primary["songs"]):
        raise ValueError("Duplicate song IDs")
    if set(demos) != {row["id"] for row in primary["songs"]}:
        raise ValueError("Both collections must have the same song IDs")
    songs = []
    for row in primary["songs"]:
        if row["status"] != "complete":
            raise ValueError("Primary collection is incomplete")
        demo = demos[row["id"]]
        source = Path(row["audio"]["path"])
        if source.resolve() != Path(demo["original_source"]).resolve():
            raise ValueError("Models analyzed different source audio")
        if signature(source)["sha256"] != row["audio"]["sha256"]:
            raise ValueError("Source audio has changed since inference")
        a_path, b_path = Path(row["prediction"]), Path(demo["native_json"])
        a, b = [json.loads(path.read_text(encoding="utf-8-sig")) for path in (a_path, b_path)]
        if signature(a_path)["sha256"] != row.get("prediction_signature", {}).get("sha256"):
            raise ValueError("Primary analysis changed since its report")
        if Path(a.get("original_source") or a["source"]).resolve() != source.resolve():
            raise ValueError("Primary payload source differs from its report")
        if Path(b.get("original_source") or b["source"]).resolve() != source.resolve():
            raise ValueError("Comparison payload original source differs from its report")
        if Path(b["source"]).resolve() != Path(demo["source"]).resolve():
            raise ValueError("Comparison playback source differs from its report")
        from build_structure_workbench import event_hash
        if event_hash(b) != demo.get("canonical_event_sha256"):
            raise ValueError("Comparison events changed since their report")
        song = compare_timelines(a, b, minimum)
        song.update(id=row["id"], title=row.get("title", row["id"]),
                    primary_analysis=signature(a_path), comparison_analysis=signature(b_path),
                    source_audio=signature(source), primary_html=row["html"], comparison_html=demo["native_html"])
        songs.append(song)
    return dict(protocol="reference_free_model_disagreement_listening", reference_hints_used=False,
                primary_model="当前默认流程", comparison_model="SheetSage2", songs=songs,
                description="两模型分歧只用于定位试听，不代表谁正确，也不是置信度。未改动任何和弦或人工标注。短于显示阈值的分歧保留在 all_segments。",
                input_reports=[signature(primary_path), signature(comparison_path)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--min-duration", type=float, default=.5)
    args = parser.parse_args()
    result = build(args.primary, args.comparison, args.min_duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps([dict(id=s["id"], displayed=len(s["segments"]), total=s["total_count"],
                           seconds=s["disagreement_seconds"]) for s in result["songs"]]))


if __name__ == "__main__":
    main()
