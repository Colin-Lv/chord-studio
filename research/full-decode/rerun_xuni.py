"""Restart the desktop app, re-analyse 虚拟, and inspect the new full-decoder source.

Needs the desktop Origin/Referer headers, otherwise the API refuses the request.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import requests

EXE = r"C:\Users\33573\AppData\Local\Programs\Chord Studio\chord-studio.exe"
SESSION = Path(r"D:\ChordStudio\data\active-session.json")
PROJECT = "aa5a41797bb343e590abcf4d8b27ae54"
APP_ROOT = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio")
MODULE = APP_ROOT / "backend" / "app" / "web" / "full_decode.py"
WORKER = APP_ROOT / "backend" / "app" / "web" / "retained_worker.py"


def wait_health(base, headers, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base}/api/config", headers=headers, timeout=5)
            if r.ok:
                return r.json()
        except Exception:                     # noqa: BLE001
            pass
        time.sleep(2)
    raise SystemExit("app did not become healthy")


def main():
    print("module mtime:", time.strftime("%H:%M:%S", time.localtime(MODULE.stat().st_mtime)),
          "worker mtime:", time.strftime("%H:%M:%S", time.localtime(WORKER.stat().st_mtime)))
    subprocess.run(["taskkill", "/IM", "chord-studio.exe", "/F"], capture_output=True)
    time.sleep(3)
    subprocess.Popen([EXE])
    time.sleep(24)
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{session['port']}"
    headers = {"x-chord-session": session["token"], "X-Harmony-Client": "studio",
               "Origin": base, "Referer": base + "/"}
    print("health:", wait_health(base, headers))

    r = requests.post(f"{base}/api/projects/{PROJECT}/analyze", headers=headers,
                      json={"profile": "retained10", "key_mode": "madmom"}, timeout=60)
    print("job:", r.status_code, r.text[:160])
    r.raise_for_status()
    job = r.json()["id"]
    for _ in range(240):
        time.sleep(3)
        row = requests.get(f"{base}/api/jobs/{job}", headers=headers, timeout=30).json()
        if row.get("status") in {"completed", "failed", "interrupted"}:
            print("status:", row["status"], row.get("error"))
            if row.get("status") != "completed":
                raise SystemExit(1)
            break
    run = requests.get(f"{base}/api/projects/{PROJECT}", headers=headers,
                       timeout=30).json()["current_run"]
    path = Path(r"D:\ChordStudio\data\projects") / PROJECT / "runs" / run / "analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    print("run:", run)
    print("default:", data["default_source"], "| key:", (data.get("key_analysis") or {}).get("key"))
    print("sources:", [(x["id"], len(x["rows"])) for x in data["sources"]])
    prov = data.get("provenance", {})
    print("full_decode provenance:", json.dumps(prov.get("full_decode"), ensure_ascii=False))
    print("lab files:", sorted(p.name for p in path.parent.glob("*.lab")))
    (Path(__file__).resolve().parent / "prod_run.json").write_text(
        json.dumps(dict(run=run, analysis=str(path), default=data["default_source"],
                        sources=[(x["id"], len(x["rows"])) for x in data["sources"]],
                        full_decode=prov.get("full_decode")), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

