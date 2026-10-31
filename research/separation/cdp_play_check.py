"""Drive the stems page in headless Chrome and verify playback synchronisation.

Bootstraps the session cookie, opens /stems, clicks play, waits, then reads the four audio
elements' currentTime to confirm the tracks advance together, and captures a screenshot.
"""
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
PORT = 9334
TARGET = "/stems?project=aa5a41797bb343e590abcf4d8b27ae54"


def session():
    value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{value['port']}", value["token"]


async def main():
    base, token = session()
    profile = Path(tempfile.mkdtemp(prefix="cdp-play-"))
    chrome = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
        "--autoplay-policy=no-user-gesture-required",
        f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
        "--window-size=1500,1080", "about:blank",
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
            await call("Emulation.setDeviceMetricsOverride", width=1500, height=1080,
                       deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=f"{base}/?session={token}")
            await asyncio.sleep(2.0)
            await call("Page.navigate", url=f"{base}{TARGET}")
            await asyncio.sleep(6.0)

            print("page:", await evaluate("document.querySelector('.stems-title h1')?.textContent"))
            print("tracks:", await evaluate("document.querySelectorAll('audio.stem-audio').length"))
            print("bars:", await evaluate("document.querySelectorAll('#ruler span').length"))

            await evaluate("document.getElementById('play').click()")
            await asyncio.sleep(3.0)
            state = await evaluate("""(() => {
                const audios = [...document.querySelectorAll('audio.stem-audio')];
                return {
                  times: audios.map(a => Number(a.currentTime.toFixed(3))),
                  paused: audios.map(a => a.paused),
                  volume: audios.map(a => a.volume),
                  muted: audios.map(a => a.muted),
                  clock: document.getElementById('clock').textContent,
                  playLabel: document.getElementById('play').textContent,
                };
            })()""")
            print("after 3s:", json.dumps(state, ensure_ascii=False))
            times = state["times"]
            spread = max(times) - min(times) if times else 0
            print(f"sync spread: {spread:.3f}s")

            # mute / solo behaviour
            await evaluate("document.querySelector('.track-head button.solo').click()")
            await asyncio.sleep(0.4)
            solo = await evaluate("""(() => ({
                muted: [...document.querySelectorAll('audio.stem-audio')].map(a => a.muted),
                pressed: [...document.querySelectorAll('.track-head button.solo')].map(b => b.getAttribute('aria-pressed')),
            }))()""")
            print("solo:", json.dumps(solo, ensure_ascii=False))

            shot = await call("Page.captureScreenshot", format="png")
            Path(r"D:\ChordStudio\research\separation\shots\stems_playing.png").write_bytes(
                base64.b64decode(shot["data"]))
            await evaluate("document.getElementById('play').click()")
            print("screenshot saved")
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:                              # noqa: BLE001
            chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
