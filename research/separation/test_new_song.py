"""End-to-end check on songs the app has never seen: upload -> separate -> manifest.

Creates a short stereo clip and a very short mono clip, uploads them as new projects,
runs separation, and reports the manifest shape (bars, steps, duration).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

SESSION = Path(r"D:\ChordStudio\data\active-session.json")
FFMPEG = r"C:\Users\33573\AppData\Roaming\Python\Python313\Scripts\ffmpeg.EXE"
SOURCE = Path(r"D:\ChordStudio\data\projects\aa5a41797bb343e590abcf4d8b27ae54\runs\11fa80454ad94da3a81efa19c1cab64c\source\audio.m4a")


def make_clip(target: Path, start: int, seconds: int, channels: int):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                    "-t", str(seconds), "-i", str(SOURCE), "-ac", str(channels), "-ar", "44100",
                    str(target)], check=True)


def main():
    value = json.loads(SESSION.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{value['port']}"
    headers = {"x-chord-session": value["token"], "X-Harmony-Client": "studio",
               "Origin": base, "Referer": base + "/"}
    work = Path(tempfile.mkdtemp(prefix="stems-upload-"))
    cases = [("自检 25s", 25, 2), ("自检 6s 单声道", 6, 1)]
    results = []
    for name, seconds, channels in cases:
        clip = work / f"{name}.wav"
        make_clip(clip, 100, seconds, channels)
        with clip.open("rb") as stream:
            response = requests.post(f"{base}/api/upload", headers=headers,
                                     files={"file": (clip.name, stream, "audio/wav")}, timeout=120)
        print(f"[{name}] upload {response.status_code}")
        response.raise_for_status()
        project = response.json()
        job = requests.post(f"{base}/api/projects/{project['id']}/separate", headers=headers,
                            json={"force": False, "model": "htdemucs"}, timeout=60).json()
        started = time.time()
        for _ in range(120):
            time.sleep(2)
            row = requests.get(f"{base}/api/jobs/{job['id']}", headers=headers, timeout=30).json()
            if row["status"] in {"completed", "failed", "interrupted"}:
                print(f"  [{time.time() - started:4.0f}s] {row['status']} {row.get('error') or ''}")
                break
        manifest = requests.get(f"{base}/api/projects/{project['id']}/stems", headers=headers,
                                timeout=30)
        print(f"  manifest {manifest.status_code}")
        if manifest.status_code == 200:
            data = manifest.json()
            print(f"  duration={data['duration']} bpm={data['bpm']} bars={len(data['bars'])} "
                  f"tracks={len(data['tracks'])} steps={len(data['tracks'][0]['steps'])}")
            results.append((name, project["id"], len(data["bars"])))
    print("\nprojects:", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
