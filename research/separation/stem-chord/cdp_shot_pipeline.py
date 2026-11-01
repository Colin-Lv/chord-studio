"""Screenshot the workbench: pipeline panel open, then a project with the fused source."""
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
PORT = 9341
OUT = Path(r"D:\ChordStudio\research\separation\stem-chord")


def session():
    value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{value['port']}", value["token"]


async def main():
    base, token = session()
    profile = Path(tempfile.mkdtemp(prefix="cdp-pipe-"))
    chrome = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
        "--autoplay-policy=no-user-gesture-required",
        f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
        "--window-size=1500,1050", "about:blank",
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
            await call("Emulation.setDeviceMetricsOverride", width=1500, height=1050,
                       deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=f"{base}/?session={token}")
            await asyncio.sleep(4.0)

            await evaluate("document.getElementById('separate').click()")
            await asyncio.sleep(1.5)
            print("panel hidden:", await evaluate("document.getElementById('pipelinePanel').hidden"))
            print("models:", await evaluate("[...document.getElementById('pipeModel').options].map(o => o.textContent)"))
            await shot("panel.png")

            # open the project that has the fused timeline
            await evaluate("""(() => {
              const items = [...document.querySelectorAll('.project-item')];
              const hit = items.find(b => b.textContent.includes('虚拟'));
              if (hit) hit.click();
              return items.length;
            })()""")
            await asyncio.sleep(6.0)
            print("sources:", await evaluate("[...document.querySelectorAll('#sources button')].map(b => b.textContent.trim())"))
            print("default source:", await evaluate("document.querySelector('#sources button[aria-pressed=\"true\"]')?.textContent.trim()"))
            await shot("fused_result.png")
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:                              # noqa: BLE001
            chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
