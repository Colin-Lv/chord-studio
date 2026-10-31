"""Screenshot a gated page of the desktop app through the Chrome DevTools Protocol.

The app only accepts requests carrying its session cookie, which is set when the browser
opens `/?session=<token>`. A plain `--screenshot` run has no cookie, so this script drives a
headless Chrome with a persistent profile: bootstrap first, then navigate to the target.

Usage:  python cdp_shot.py <path-with-query> <out.png> [width] [height] [settle-ms]
"""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import requests
import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9333


def session():
    value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{value['port']}", value["token"]


async def shoot(target: str, out: Path, width: int, height: int, settle: int):
    base, token = session()
    profile = Path(tempfile.mkdtemp(prefix="cdp-profile-"))
    chrome = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
        f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
        f"--window-size={width},{height}", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    page = None
    try:
        for _ in range(60):
            time.sleep(0.5)
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
                page = next(item for item in targets if item["type"] == "page")
                break
            except Exception:                          # noqa: BLE001
                continue
        if page is None:
            raise SystemExit("chrome did not expose a page target")

        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024) as ws:
            counter = 0

            async def call(method, **params):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps(dict(id=counter, method=method, params=params)))
                while True:
                    message = json.loads(await ws.recv())
                    if message.get("id") == counter:
                        return message.get("result", {})

            await call("Page.enable")
            await call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                       deviceScaleFactor=1, mobile=False)
            # 1) bootstrap: sets the session cookie on /
            await call("Page.navigate", url=f"{base}/?session={token}")
            await asyncio.sleep(2.5)
            # 2) the real page
            await call("Page.navigate", url=f"{base}{target}")
            await asyncio.sleep(settle / 1000)
            result = await call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(result["data"]))
            print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:                              # noqa: BLE001
            chrome.kill()


if __name__ == "__main__":
    target = sys.argv[1]
    out = Path(sys.argv[2])
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    settle = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
    asyncio.run(shoot(target, out, width, height, settle))
