"""Find which request fails when a project with a stem_fusion source is opened."""
from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9342


def session():
    value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{value['port']}", value["token"]


async def main():
    base, token = session()
    chrome = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--mute-audio",
        f"--remote-debugging-port={PORT}", f"--user-data-dir={tempfile.mkdtemp(prefix='cdp-err-')}",
        "--window-size=1400,900", "about:blank",
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
            pending = {}

            async def call(method, **params):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps(dict(id=counter, method=method, params=params)))
                while True:
                    message = json.loads(await ws.recv())
                    if message.get("id") == counter:
                        return message.get("result", {})

            await call("Page.enable")
            await call("Runtime.enable")
            await call("Network.enable")
            await call("Page.navigate", url=f"{base}/?session={token}")
            await asyncio.sleep(4.0)

            async def click_project():
                await call("Runtime.evaluate", expression="""
                  [...document.querySelectorAll('.project-item')].find(b => b.textContent.includes('虚拟'))?.click()
                """)

            task = asyncio.create_task(click_project())
            failures = []
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                except asyncio.TimeoutError:
                    continue
                if message.get("method") == "Network.responseReceived":
                    params = message["params"]
                    status = params["response"]["status"]
                    if status >= 400:
                        failures.append((status, params["response"]["url"]))
            await task
            print("failures:")
            for status, url in failures:
                print(" ", status, url)
            print("toast:", await (lambda: None)() if False else "")
            result = await call("Runtime.evaluate", expression="document.getElementById('toast').textContent",
                                returnByValue=True)
            print("toast text:", result.get("result", {}).get("value"))
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:                              # noqa: BLE001
            chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
