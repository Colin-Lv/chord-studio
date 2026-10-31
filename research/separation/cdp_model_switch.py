"""Verify switching between the two checkpoints in the page."""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9338
TARGET = "/stems?project=aa5a41797bb343e590abcf4d8b27ae54"


def session():
    value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{value['port']}", value["token"]


async def main():
    base, token = session()
    profile = Path(tempfile.mkdtemp(prefix="cdp-model-"))
    chrome = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
        "--autoplay-policy=no-user-gesture-required",
        f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
        "--window-size=1500,1000", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = None
        for _ in range(60):
            time.sleep(0.5)
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
                page = next(item for item in targets if item["type"] == "page")
                break
            except Exception:                          # noqa: BLE001
                continue
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

            async def evaluate(expression):
                result = await call("Runtime.evaluate", expression=expression,
                                    returnByValue=True, awaitPromise=True)
                return result.get("result", {}).get("value")

            await call("Page.enable")
            await call("Runtime.enable")
            await call("Emulation.setDeviceMetricsOverride", width=1500, height=1000,
                       deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=f"{base}/?session={token}")
            await asyncio.sleep(2.0)
            await call("Page.navigate", url=f"{base}{TARGET}")
            await asyncio.sleep(5.0)
            print("options:", await evaluate("""[...document.getElementById('modelSelect').options].map(o => o.textContent)"""))
            print("current:", await evaluate("document.getElementById('modelInfo').textContent"))
            print("audio src:", await evaluate("document.querySelector('audio.stem-audio').src.split('?')[1] || 'no-query'"))

            await evaluate("""(() => { const s = document.getElementById('modelSelect'); s.value = 'htdemucs_ft'; s.dispatchEvent(new Event('change')); })()""")
            await asyncio.sleep(4.0)
            print("after switch:", await evaluate("document.getElementById('modelInfo').textContent"))
            print("audio src:", await evaluate("document.querySelector('audio.stem-audio').src.split('?')[1] || 'no-query'"))
            shot = await call("Page.captureScreenshot", format="png")
            Path(r"D:\ChordStudio\research\separation\shots\model_switch.png").write_bytes(
                base64.b64decode(shot["data"]))
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:                              # noqa: BLE001
            chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
