"""Read the workbench select states in headless Chrome to explain the rendered default."""
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
PORT = 9336


def session():
    value = json.loads(Path(r"D:\ChordStudio\data\active-session.json").read_text(encoding="utf-8"))
    return f"http://127.0.0.1:{value['port']}", value["token"]


async def main():
    base, token = session()
    profile = Path(tempfile.mkdtemp(prefix="cdp-form-"))
    chrome = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}",
        f"--user-data-dir={profile}", "--window-size=1400,900", "about:blank",
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
        async with websockets.connect(page["webSocketDebuggerUrl"]) as ws:
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
            await call("Page.navigate", url=f"{base}/?session={token}")
            await asyncio.sleep(4.0)
            print("initial value:", await evaluate("document.getElementById('keyMode').value"))
            print("options:", await evaluate("""[...document.getElementById('keyMode').options].map(o => o.value + ':' + (o.selected ? 'sel' : ''))"""))
            print("result panel hidden:", await evaluate("document.getElementById('resultPanel').hidden"))
            print("state.project:", await evaluate("window.__dsh ? 'n/a' : typeof state === 'object' ? (state && state.project ? state.project.id : null) : 'no state'"))
            print("separate button:", await evaluate("document.getElementById('separate')?.textContent"))
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except Exception:                              # noqa: BLE001
            chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
