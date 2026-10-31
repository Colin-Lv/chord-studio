#!/usr/bin/env python
"""Apply trusted local-key sections to an existing Chord Studio analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from detect_key_timeline import apply_segments
from train_candidate_reranker import resolve_path


def active_key_for_bar(sections: list[dict[str, Any]], bar_number: int) -> str:
    active = sections[0]
    for section in sections:
        start = int(section.get("start_bar") or 1)
        end_value = section.get("end_bar")
        end = int(end_value) if end_value not in {None, ""} else 10**9
        if start <= bar_number <= end:
            active = section
    return str(active.get("key") or "C")


def annotate_bar_keys(payload: dict[str, Any], sections: list[dict[str, Any]]) -> None:
    for bar in payload.get("bars") or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        if bar_number:
            bar["local_key"] = active_key_for_bar(sections, bar_number)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def parse_section(value: str) -> dict[str, Any]:
    """Parse a section string like ``1-15:B`` or ``50-:Ab``."""
    try:
        bars, key = value.split(":", 1)
        start_text, end_text = bars.split("-", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid section {value!r}; expected start-end:key") from exc
    start = int(start_text)
    end = int(end_text) if end_text.strip() else None
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Invalid section {value!r}; key is empty")
    return {"start_bar": start, "end_bar": end, "key": key}


def sections_from_reference(reference: dict[str, Any]) -> list[dict[str, Any]]:
    if reference.get("modulations"):
        return [
            {
                "start_bar": int(item.get("start_bar") or 1),
                "end_bar": item.get("end_bar"),
                "key": str(item.get("key") or reference.get("key") or "C"),
                "mode": item.get("mode"),
                "reason": item.get("reason") or "copied from reference analysis",
            }
            for item in reference.get("modulations") or []
        ]
    timeline = reference.get("key_timeline") or {}
    if timeline.get("segments"):
        return [
            {
                "start_bar": int(item.get("start_bar") or 1),
                "end_bar": item.get("end_bar"),
                "key": str(item.get("key") or reference.get("key") or "C"),
                "reason": "copied from reference key_timeline",
            }
            for item in timeline.get("segments") or []
        ]

    sections: list[dict[str, Any]] = []
    current_key = None
    start_bar = None
    previous_bar = None
    for bar in reference.get("bars") or []:
        bar_number = int(bar.get("bar") or bar.get("number") or 0)
        key = str(bar.get("local_key") or "").strip()
        if not bar_number or not key:
            continue
        if key != current_key:
            if current_key is not None and start_bar is not None:
                sections.append({"start_bar": start_bar, "end_bar": previous_bar, "key": current_key})
            current_key = key
            start_bar = bar_number
        previous_bar = bar_number
    if current_key is not None and start_bar is not None:
        sections.append({"start_bar": start_bar, "end_bar": previous_bar, "key": current_key})
    return sections


def normalize_sections(sections: list[dict[str, Any]], total_bars: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in sorted(sections, key=lambda row: int(row.get("start_bar") or 1)):
        start = int(item.get("start_bar") or 1)
        end_value = item.get("end_bar")
        end = int(end_value) if end_value not in {None, ""} else None
        if end is not None and end < start:
            raise SystemExit(f"Invalid key section: end_bar before start_bar: {item}")
        normalized.append(
            {
                "start_bar": start,
                "end_bar": end if end is not None and end <= total_bars else None,
                "key": str(item.get("key") or "C"),
                "mode": item.get("mode"),
                "reason": item.get("reason") or "trusted key timeline",
            }
        )
    if not normalized:
        raise SystemExit("No key sections were provided or found in reference analysis.")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--from-analysis", type=Path, help="Copy modulations/key_timeline/local_key sections from this analysis.")
    parser.add_argument("--section", action="append", type=parse_section, default=[], help="Trusted section: start-end:key, e.g. 1-15:B or 50-:Ab")
    parser.add_argument("--notation", choices=["auto", "sharps", "flats"], default="auto")
    parser.add_argument("--html-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_json(args.analysis)
    sections = list(args.section)
    if args.from_analysis:
        sections.extend(sections_from_reference(load_json(args.from_analysis)))
    total_bars = len(payload.get("bars") or [])
    sections = normalize_sections(sections, total_bars)
    output_payload = apply_segments(payload, sections, args.notation)
    annotate_bar_keys(output_payload, sections)
    output_payload["key_timeline"] = {
        "method": "trusted local-key sections",
        "segments": [{key: item.get(key) for key in ("start_bar", "end_bar", "key")} for item in sections],
        "source_analysis": str(args.from_analysis) if args.from_analysis else "",
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html_output:
        from chord_studio import write_html

        write_html(resolve_path(args.html_output), output_payload)
    print(f"Wrote {output} with {len(sections)} key section(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
