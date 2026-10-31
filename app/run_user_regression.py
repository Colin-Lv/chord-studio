#!/usr/bin/env python
"""Run user-verified regression checks for Chord Studio analyses."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_refine_case(case: dict[str, Any], force: bool) -> Path:
    command = case["command"]
    output_base = resolve_path(command["output"])
    assert output_base is not None
    final_json = output_base.with_name(output_base.name + "_analysis.json")
    postprocess_steps = case.get("postprocess", [])
    expected_final_json = (
        output_base.with_name(f"{output_base.name}_post{len(postprocess_steps)}_analysis.json")
        if postprocess_steps
        else final_json
    )
    if expected_final_json.exists() and not force:
        return expected_final_json
    if final_json.exists() and not force:
        return final_json

    audio = resolve_path(case.get("audio"))
    if audio is None or not audio.exists():
        raise FileNotFoundError(f"{case['id']}: missing audio: {audio}")
    external = resolve_path(case.get("external_chords"))
    output_base.parent.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(REPO_ROOT / "app" / "refine_song_structure.py"),
        str(audio),
        "-o",
        str(output_base),
        "--bpm",
        str(command["bpm"]),
        "--meter",
        str(command.get("meter", "auto")),
        "--keys",
        str(command.get("keys", "C,D,E,F,G,A,Bb,B")),
        "--progressions",
        str(command.get("progressions", "6415,6545,1564,1645")),
        "--max-entry-time",
        str(command.get("max_entry_time", 35.0)),
        "--min-modulation-bar",
        str(command.get("min_modulation_bar", 1)),
    ]
    if external is not None and external.exists():
        args += ["--external-chords", str(external)]
    if command.get("run_analysis", False):
        args.append("--run-analysis")
    run_command(args)

    current = final_json
    for index, step in enumerate(postprocess_steps, start=1):
        next_json = output_base.with_name(f"{output_base.name}_post{index}_analysis.json")
        tool = step.get("tool")
        if tool == "apply_bar_override":
            args = [
                sys.executable,
                str(REPO_ROOT / "app" / "apply_bar_override.py"),
                str(current),
                "-o",
                str(next_json),
                "--start-bar",
                str(step["start_bar"]),
                "--end-bar",
                str(step["end_bar"]),
                "--chord",
                str(step["chord"]),
                "--key",
                str(step["key"]),
            ]
            if step.get("roman"):
                args += ["--roman", str(step["roman"])]
            if step.get("roman_display"):
                args += ["--roman-display", str(step["roman_display"])]
            args.append("--html")
            run_command(args)
        elif tool == "apply_progression_sections":
            args = [
                sys.executable,
                str(REPO_ROOT / "app" / "apply_progression_sections.py"),
                str(current),
                "-o",
                str(next_json),
            ]
            for section in step.get("sections", []):
                args += ["--section", str(section)]
            args.append("--html")
            run_command(args)
        else:
            raise ValueError(f"{case['id']}: unknown postprocess tool: {tool}")
        current = next_json
    return current


def get_bar(payload: dict[str, Any], bar_number: int) -> dict[str, Any] | None:
    for bar in payload.get("bars", []):
        if int(bar.get("bar", 0)) == bar_number:
            return bar
    return None


def bar_span_values(bar: dict[str, Any], field: str) -> list[str]:
    values = []
    spans = bar.get("spans") or bar.get("beats") or []
    for item in spans:
        value = item.get(field)
        if value is not None:
            values.append(str(value))
    return values


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"  FAIL {message}")


def pass_check(message: str) -> None:
    print(f"  PASS {message}")


def check_expected(case: dict[str, Any], analysis_path: Path) -> list[str]:
    payload = load_json(analysis_path)
    expected = case.get("expected", {})
    failures: list[str] = []

    meter = expected.get("meter")
    if meter is not None:
        if payload.get("meter") == meter:
            pass_check(f"meter = {meter}")
        else:
            fail(f"meter expected {meter}, got {payload.get('meter')}", failures)

    beats_per_bar = expected.get("beats_per_bar")
    if beats_per_bar is not None:
        if int(payload.get("beats_per_bar", 0)) == int(beats_per_bar):
            pass_check(f"beats_per_bar = {beats_per_bar}")
        else:
            fail(f"beats_per_bar expected {beats_per_bar}, got {payload.get('beats_per_bar')}", failures)

    key_contains = expected.get("key_contains")
    if key_contains is not None:
        key_text = str(payload.get("key", ""))
        if str(key_contains) in key_text:
            pass_check(f"key contains {key_contains}")
        else:
            fail(f"key expected to contain {key_contains}, got {key_text}", failures)

    min_bars = expected.get("min_bars")
    if min_bars is not None:
        bars = len(payload.get("bars", []))
        if bars >= int(min_bars):
            pass_check(f"bar count {bars} >= {min_bars}")
        else:
            fail(f"bar count expected >= {min_bars}, got {bars}", failures)

    body_entry = expected.get("body_entry")
    if body_entry:
        actual = None
        bars = payload.get("bars", [])
        if bars:
            actual = float(bars[0].get("start", 0.0))
        target = float(body_entry["time"])
        tolerance = float(body_entry.get("tolerance", 0.25))
        if actual is not None and math.isfinite(actual) and abs(actual - target) <= tolerance:
            pass_check(f"body entry {actual:.3f}s ~= {target:.3f}s")
        else:
            fail(f"body entry expected {target:.3f}s +/- {tolerance:.3f}s, got {actual}", failures)

    expected_sections = expected.get("sections", [])
    actual_sections = payload.get("modulations", [])
    for expected_section in expected_sections:
        start_bar = int(expected_section["start_bar"])
        match = next((item for item in actual_sections if int(item.get("start_bar", 0)) == start_bar), None)
        if match is None:
            fail(f"missing section starting at bar {start_bar}", failures)
            continue
        mismatches = []
        for field in ("end_bar", "key", "progression"):
            if field not in expected_section:
                continue
            if match.get(field) != expected_section[field]:
                mismatches.append(f"{field} expected {expected_section[field]}, got {match.get(field)}")
        if mismatches:
            fail(f"section {start_bar}: " + "; ".join(mismatches), failures)
        else:
            pass_check(f"section {start_bar} matches")

    for expected_bar in expected.get("bar_labels", []):
        bar_number = int(expected_bar["bar"])
        bar = get_bar(payload, bar_number)
        if bar is None:
            fail(f"missing bar {bar_number}", failures)
            continue
        if "roman_sequence" in expected_bar:
            actual_sequence = bar_span_values(bar, "roman")
            target_sequence = [str(item) for item in expected_bar["roman_sequence"]]
            if actual_sequence[: len(target_sequence)] == target_sequence:
                pass_check(f"bar {bar_number} roman sequence starts with {' '.join(target_sequence)}")
            else:
                fail(f"bar {bar_number} roman sequence expected {target_sequence}, got {actual_sequence}", failures)
        for field in ("roman", "chord", "local_key"):
            if field not in expected_bar:
                continue
            actual_values = set(bar_span_values(bar, field))
            target = str(expected_bar[field])
            if actual_values == {target}:
                pass_check(f"bar {bar_number} {field} = {target}")
            else:
                fail(f"bar {bar_number} {field} expected only {target}, got {sorted(actual_values)}", failures)
    return failures


def analysis_path_for_case(case: dict[str, Any], force: bool) -> Path:
    if case.get("command", {}).get("tool") == "refine_song_structure":
        return run_refine_case(case, force)
    analysis = resolve_path(case.get("analysis"))
    if analysis is None:
        raise ValueError(f"{case['id']}: case needs either analysis or command")
    if not analysis.exists():
        raise FileNotFoundError(f"{case['id']}: missing analysis: {analysis}")
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "user_regression_cases.json",
        help="JSON file containing user-verified cases.",
    )
    parser.add_argument("--id", action="append", dest="ids", help="Run only the selected case id. Can be repeated.")
    parser.add_argument("--force", action="store_true", help="Regenerate command-based cases even if outputs already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(resolve_path(str(args.cases)) or args.cases)
    wanted = set(args.ids or [])
    total_failures: list[str] = []
    for case in config.get("cases", []):
        case_id = str(case["id"])
        if wanted and case_id not in wanted:
            continue
        print(f"\n== {case_id}: {case.get('title', '')} ==")
        try:
            analysis_path = analysis_path_for_case(case, args.force)
            print(f"analysis: {analysis_path}")
            failures = check_expected(case, analysis_path)
            total_failures.extend(f"{case_id}: {item}" for item in failures)
        except Exception as error:  # noqa: BLE001 - keep regression output actionable.
            message = f"{case_id}: ERROR {error}"
            print("  " + message)
            total_failures.append(message)

    print(f"\nSummary: {len(total_failures)} failure(s)")
    for item in total_failures:
        print(f"- {item}")
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
