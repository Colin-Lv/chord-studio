"""Restart the desktop app, re-analyse the projects with a chosen key mode, dump the result.

Verifies that the key mode actually changes the published key and that the experimental block
stays available as evidence.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

EXE = r"C:\Users\33573\AppData\Local\Programs\Chord Studio\chord-studio.exe"
SESSION = Path(r"D:\ChordStudio\data\active-session.json")
PROJECT_ROOT = Path(r"D:\ChordStudio\data\projects")
PROJECTS = {
    "虚拟(新项目)": "aa5a41797bb343e590abcf4d8b27ae54",
    "虚拟(旧v1项目)": "087d8b0c86414fe6b675cc622f505128",
    "坏女孩": "0bc0304e01ed40d8baac6b76575b25b6",
    "沧海一粟": "e18b69d0edef4c5e91f6594520a816cb",
}
MODE = os.environ.get("KEY_MODE", "continuous")


def wait_config(base, headers, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{base}/api/config", headers=headers, timeout=5)
            if response.ok:
                return response.json()
        except Exception:                     # noqa: BLE001
            pass
        time.sleep(2)
    raise SystemExit("app did not become ready")


def main():
    restart = os.environ.get("NO_RESTART") != "1"
    if restart:
        subprocess.run(["taskkill", "/IM", "chord-studio.exe", "/F"], capture_output=True)
        time.sleep(3)
        subprocess.Popen([EXE])
        time.sleep(26)
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{session['port']}"
    headers = {"x-chord-session": session["token"], "X-Harmony-Client": "studio",
               "Origin": base, "Referer": base + "/"}
    print("ready:", bool(wait_config(base, headers)), "| key mode:", MODE)

    for title, project in PROJECTS.items():
        response = requests.post(f"{base}/api/projects/{project}/analyze", headers=headers,
                                 json={"profile": "retained10", "key_mode": MODE}, timeout=60)
        if response.status_code >= 400:
            print(f"{title}: request rejected {response.status_code} {response.text[:200]}")
            continue
        job = response.json()["id"]
        for _ in range(240):
            time.sleep(3)
            row = requests.get(f"{base}/api/jobs/{job}", headers=headers, timeout=30).json()
            if row.get("status") in {"completed", "failed", "interrupted"}:
                if row["status"] != "completed":
                    print(f"{title}: job {row['status']} {row.get('error')}")
                break
        run = requests.get(f"{base}/api/projects/{project}", headers=headers,
                           timeout=30).json()["current_run"]
        analysis = json.loads((PROJECT_ROOT / project / "runs" / run / "analysis.json")
                              .read_text(encoding="utf-8"))
        key = analysis.get("key_analysis") or {}
        block = key.get("experimental") or {}
        warning = next((w for w in analysis.get("warnings", []) if "连续评分" in w), "")
        print(f"\n{title} run {run[:8]}  key_mode {analysis.get('key_mode')}  "
              f"published {key.get('key')}  source {key.get('source')}  "
              f"experimental {block.get('key')}")
        print(f"  acoustic {key.get('cnn_key') or key.get('acoustic_key')}  "
              f"gated {key.get('gated_key')}  gap {block.get('gap')}")
        print(f"  warning: {warning[:120]}")


if __name__ == "__main__":
    sys.exit(main())
