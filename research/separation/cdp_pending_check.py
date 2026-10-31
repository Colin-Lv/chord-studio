"""End-to-end check of the pending -> separate -> load flow for a project without stems."""
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
PORT = 9335
PROJECT = "0bc0304e01ed40d8baac6b76575b25b6"          # 坏女孩, no stems yet
OUT = Path(r"D:\ChordStudio\research\separation\shots")


def session():
    value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{value['port']}", value["token"]


async def main():
    base, token = session()
    profile = Path(tempfile.mkdtemp(prefix="cdp-pending-"))
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

            async def shot(name):
                data = await call("Page.captureScreenshot", format="png")
                path = OUT / name
                path.write_bytes(base64.b64decode(data["data"]))
                print("shot:", path.name)

            await call("Page.enable")
            await call("Runtime.enable")
            await call("Emulation.setDeviceMetricsOverride", width=1500, height=1000,
                       deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=f"{base}/?session={token}")
            await asyncio.sleep(2.0)
            await call("Page.navigate", url=f"{base}/stems?project={PROJECT}")
            await asyncio.sleep(4.0)
            print("pending visible:", await evaluate("!document.getElementById('pending').hidden"))
            print("status text:", await evaluate("document.getElementById('songMeta').textContent"))
            await shot("pending.png")

            await evaluate("document.getElementById('startSeparate').click()")
            started = time.time()
            for _ in range(180):
                await asyncio.sleep(3)
                status = await evaluate("document.getElementById('separateStatus').textContent")
                ready = await evaluate("!document.getElementById('workspace').hidden")
                print(f"  [{time.time() - started:5.0f}s] {status}")
                if ready:
                    break
                if "失败" in (status or "") or "重试" in (status or ""):
                    print("FAILED")
                    break
            print("workspace visible:", await evaluate("!document.getElementById('workspace').hidden"))
            print("tracks:", await evaluate("document.querySelectorAll('audio.stem-audio').length"))
            print("bpm:", await evaluate("document.getElementById('bpm').textContent"))
            await shot("badgirl_stems.png")
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:                              # noqa: BLE001
            chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
