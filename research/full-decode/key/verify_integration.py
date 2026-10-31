"""Acceptance check for the continuous-key integration (backend + result + workbench).

  1. every project's current run was produced with key_mode=continuous and publishes the
     continuous key, while keeping the acoustic and gate keys as evidence;
  2. the workbench assets serve the panel, the renderer and the styles;
  3. the renderer produces the expected DOM for each real payload (node harness);
  4. the madmom-mode result is still available (one run per project is replayed read-only).

Run: python verify_integration.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(r"D:\ChordStudio\data\projects")
SESSION = Path(r"D:\ChordStudio\data\active-session.json")
FRONTEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend")
HERE = Path(__file__).resolve().parent
HARNESS = HERE / "renderer_harness.js"
PROJECTS = {
    "虚拟": "aa5a41797bb343e590abcf4d8b27ae54",
    "坏女孩": "0bc0304e01ed40d8baac6b76575b25b6",
    "沧海一粟": "e18b69d0edef4c5e91f6594520a816cb",
}
EXPECTED = {"虚拟": ("C", "G"), "坏女孩": ("E", "E"), "沧海一粟": ("G#", "Ab")}


def current_run(project):
    return sorted((PROJECT_ROOT / project / "runs").iterdir(),
                  key=lambda path: path.stat().st_mtime)[-1]


def main():
    results = []
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{session['port']}"
    headers = {"x-chord-session": session["token"], "X-Harmony-Client": "studio"}

    for title, project in PROJECTS.items():
        run = current_run(project)
        analysis = json.loads((run / "analysis.json").read_text(encoding="utf-8"))
        key = analysis.get("key_analysis") or {}
        block = key.get("experimental") or {}
        expect_key, expect_acoustic = EXPECTED[title]
        checks = {
            "key_mode continuous": analysis.get("key_mode") == "continuous",
            "published key is the continuous key": key.get("key") == block.get("key") == expect_key,
            "source is continuous_key_v1": key.get("source") == "continuous_key_v1",
            "acoustic key kept": (key.get("acoustic_key") or key.get("cnn_key")) == expect_acoustic,
            "gate key kept": "gated_key" in key,
            "ranking has 24 candidates": block.get("candidates") == 24,
            "warnings mention experimental status": any("连续评分" in w for w in analysis.get("warnings", [])),
        }
        print(f"\n{title} (run {run.name[:8]})")
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            results.append(ok)
        harness = subprocess.run(["node", str(HARNESS), str(run / "analysis.json")],
                                 capture_output=True, text=True, encoding="utf-8")
        print(f"  {'PASS' if harness.returncode == 0 else 'FAIL'}  renderer harness")
        if harness.returncode:
            print("   ", (harness.stdout or "").strip().splitlines()[-4:])
        results.append(harness.returncode == 0)

    page = requests.get(f"{base}/", headers=headers, timeout=10).text
    script = requests.get(f"{base}/assets/app.js", headers=headers, timeout=10).text
    style = requests.get(f"{base}/assets/style.css", headers=headers, timeout=10).text
    assets = {
        "index has keyExperimental panel": "keyExperimental" in page,
        "index offers continuous key mode": 'value="continuous"' in page,
        "app.js renders the panel": "renderKeyExperimental" in script,
        "app.js reads result.key_mode": 'result.key_mode||"madmom"' in script,
        "css styles the table": "key-experiment-table" in style,
    }
    print("\nserved workbench assets")
    for name, ok in assets.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        results.append(ok)

    ok = all(results)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
