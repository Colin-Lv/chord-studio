#!/usr/bin/env python
"""Export user regression cases with audio/analysis paths as a benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "benchmarks" / "audio" / "user_tests"
KNOWN_AUDIO_HINTS = {
    "pingfan_zhilu": "pingfan_zhilu_trimmed_start",
    "yorushika_dakara_boku": "BV1Qb411g7bY",
    "yorushika_dakara_boku_bar97_chain": "BV1Qb411g7bY",
    "yorushika_dakara_boku_bar97_boundary_auto": "BV1Qb411g7bY",
    "qimeidi_corrected": "BV1Sx4y157RQ",
    "starrail_heart_sorrow_key_tracking": "BV15x4y1i7m9",
}
ID_ALIASES = {
    "starrail_heart_sorrow_key_tracking": ["starrail_heart_sorrow"],
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def analysis_audio(payload: dict[str, Any]) -> str:
    audio = str(payload.get("audio") or "")
    if audio:
        return audio
    analysis_path = payload.get("analysis")
    if not analysis_path:
        return ""
    path = resolve_path(Path(str(analysis_path)))
    if not path.exists():
        return ""
    try:
        analysis = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(analysis.get("audio") or analysis.get("audio_path") or "")


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def relative_benchmark_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT / "benchmarks")).replace("\\", "/")
    except ValueError:
        return str(path)


def normalize_audio_value(value: str) -> str:
    text = str(value or "").replace("\\", "/")
    if text.startswith("benchmarks/"):
        return text[len("benchmarks/") :]
    return text


def find_audio_for_case(case: dict[str, Any]) -> str:
    audio = analysis_audio(case)
    if audio:
        return normalize_audio_value(audio)
    if not AUDIO_DIR.exists():
        return ""
    files = [path for path in AUDIO_DIR.iterdir() if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".m4s"}]
    case_id = str(case.get("id") or "")
    hints = [KNOWN_AUDIO_HINTS.get(case_id, ""), str(case.get("title") or ""), str(case.get("artist") or "")]
    for hint in hints:
        hint = compact_text(hint)
        if not hint:
            continue
        matches = [path for path in files if hint in compact_text(path.name)]
        if matches:
            matches.sort(key=lambda path: (0 if "trimmed" in path.name.lower() else 1, len(path.name)))
            return normalize_audio_value(relative_benchmark_path(matches[0]))
    return ""


def export_manifest(cases_path: Path) -> list[dict[str, str]]:
    payload = json.loads(resolve_path(cases_path).read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for case in payload.get("cases", []) or []:
        row_id = str(case.get("id") or "").strip()
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        audio = find_audio_for_case(case)
        analysis = str(case.get("analysis") or "")
        title = str(case.get("title") or row_id)
        artist = str(case.get("artist") or "")
        key = ""
        if analysis:
            path = resolve_path(Path(analysis))
            if path.exists():
                try:
                    analysis_payload = json.loads(path.read_text(encoding="utf-8"))
                    key = str(analysis_payload.get("key") or "")
                except json.JSONDecodeError:
                    pass
        base_row = {
                "id": row_id,
                "dataset": "user_regression",
                "split": "user",
                "title": title,
                "artist": artist,
                "audio": audio,
                "analysis": analysis,
                "key": key,
            }
        rows.append(base_row)
        for alias in ID_ALIASES.get(row_id, []):
            if alias in seen:
                continue
            seen.add(alias)
            rows.append({**base_row, "id": alias})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, default=Path("benchmarks/user_regression_cases.json"), nargs="?")
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmarks/manifest.user_regression.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = export_manifest(args.cases)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "dataset", "split", "title", "artist", "audio", "analysis", "key"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s): {output}")
    missing = [row["id"] for row in rows if not row["audio"]]
    if missing:
        print(f"Missing audio: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
