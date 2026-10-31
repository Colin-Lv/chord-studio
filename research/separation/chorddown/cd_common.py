"""Shared helpers for the chord-downstream experiment (mix vs harmonic stems).

Nothing here writes outside D:\\ChordStudio\\research\\separation\\chorddown.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

WORK = Path(r"D:\ChordStudio\research\separation\chorddown")
RESULTS = WORK / "results"
COND = WORK / "cond"
SYNTH = Path(r"D:\ChordStudio\research\separation\synth6")
SESSION = Path(r"D:\ChordStudio\data\active-session.json")
STEP = 0.25
BPM = 105.0
BEATS_PER_BAR = 4
BAR = BEATS_PER_BAR * 60.0 / BPM          # 2.2857142857142856 s
# One chord per bar: the built material is C-Am-F-G x4 (see cd_verify_truth.py).
TRUTH_BARS = ["C", "Am", "F", "G"] * 4


def api():
    value = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{value['port']}"
    headers = {
        "x-chord-session": value["token"],
        "X-Harmony-Client": "studio",
        "Origin": base,
        "Referer": base + "/",
    }
    return base, headers


def upload(path: Path, timeout: int = 300) -> dict:
    base, headers = api()
    with path.open("rb") as stream:
        response = requests.post(
            f"{base}/api/upload",
            headers=headers,
            files={"file": (path.name, stream, "audio/wav")},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def analyze_project(project_id: str, profile: str = "retained10",
                    key_mode: str = "continuous", poll: int = 240) -> dict:
    """Start an analysis and wait for it. Returns the /result payload or {}."""
    base, headers = api()
    job = requests.post(
        f"{base}/api/projects/{project_id}/analyze",
        headers=headers,
        json={"profile": profile, "key_mode": key_mode},
        timeout=60,
    ).json()
    started = time.time()
    status = None
    for _ in range(poll):
        time.sleep(3)
        row = requests.get(f"{base}/api/jobs/{job['id']}", headers=headers,
                           timeout=30).json()
        if row["status"] in {"completed", "failed", "interrupted"}:
            status = row["status"]
            print(f"  job {job['id']} -> {status} in {time.time() - started:.0f}s",
                  flush=True)
            if status != "completed":
                print("   error:", row.get("error"), flush=True)
                return {}
            break
    if status != "completed":
        print("  job did not finish in time", flush=True)
        return {}
    response = requests.get(f"{base}/api/projects/{project_id}/result",
                            headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def rows_of(result: dict) -> list:
    for source in result.get("sources", []) or []:
        if source.get("id") == result.get("default_source"):
            return source.get("rows", []) or []
    sources = result.get("sources", []) or []
    return (sources[0].get("rows", []) or []) if sources else []


def timeline(result: dict, duration: float, step: float = STEP) -> dict:
    """Sample the model's label every `step` seconds, like downstream_chords.py."""
    rows = rows_of(result)
    grid = {}
    n = int(duration / step + 1e-9)
    for i in range(n):
        t = i * step
        label = "N"
        for row in rows:
            if row["start"] <= t < row["end"]:
                label = row["label"]
                break
        grid[round(t, 4)] = label
    return grid


def truth_timeline(duration: float, step: float = STEP) -> dict:
    """Bar map -> frame labels. Frames past the last full bar are 'OUT'."""
    grid = {}
    n = int(duration / step + 1e-9)
    for i in range(n):
        t = i * step
        k = int(t / BAR)
        grid[round(t, 4)] = TRUTH_BARS[k] if 0 <= k < len(TRUTH_BARS) else "OUT"
    return grid


def root_of(label: str) -> str:
    if not label:
        return "N"
    head = label.split(":")[0].strip()
    return head


def key_of(result: dict):
    for path in (("key_analysis", "key"), ("key",)):
        node = result
        for part in path:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if isinstance(node, str) and node:
            return node
    return None


def key_analysis(result: dict) -> dict:
    value = result.get("key_analysis")
    return value if isinstance(value, dict) else {}
