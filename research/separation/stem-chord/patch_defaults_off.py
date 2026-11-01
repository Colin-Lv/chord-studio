"""Fusion is a negative result: restore the mix-only default and default the option off."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
sys.path.insert(0, str(BACKEND))

from app.web.storage import Store, atomic_json  # noqa: E402

PROJECT = "aa5a41797bb343e590abcf4d8b27ae54"


def restore_default():
    store = Store(Path(r"D:\ChordStudio\data\projects"))
    project = store.project(PROJECT)
    run_dir = store.run_dir(PROJECT, project["current_run"])
    result = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    ids = [source["id"] for source in result["sources"]]
    if "retained10_full" in ids:
        result["default_source"] = "retained10_full"
    elif "retained10_stable" in ids:
        result["default_source"] = "retained10_stable"
    atomic_json(run_dir / "analysis.json", result)
    print("default_source:", result["default_source"])


def main():
    restore_default()
    server = BACKEND / "app/web/server.py"
    text = server.read_text(encoding="utf-8")
    OLD = '''    harmony: bool = True
    harmony_default: bool = True'''
    NEW = '''    harmony: bool = False
    harmony_default: bool = False'''
    if text.count(OLD) == 1:
        server.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print("server.py: harmony off by default")

    js = BACKEND / "frontend/pipeline.js"
    text = js.read_text(encoding="utf-8")
    OLD_JS = 'harmony_default: document.getElementById("pipeHarmonyDefault")?.checked !== false,'
    NEW_JS = 'harmony_default: Boolean(document.getElementById("pipeHarmonyDefault")?.checked),'
    if text.count(OLD_JS) == 1:
        js.write_text(text.replace(OLD_JS, NEW_JS), encoding="utf-8")
        print("pipeline.js: fused default only when checked")

    html = BACKEND / "frontend/index.html"
    text = html.read_text(encoding="utf-8")
    OLD_HTML = '<label class="pipe-check"><input type="checkbox" id="pipeHarmony" checked> 同时识别和声（分轨融合）</label>'
    NEW_HTML = '<label class="pipe-check"><input type="checkbox" id="pipeHarmony"> 同时识别和声（分轨融合·实验）</label>'
    if text.count(OLD_HTML) == 1:
        text = text.replace(OLD_HTML, NEW_HTML)
    OLD_HTML2 = '<label class="pipe-check"><input type="checkbox" id="pipeHarmonyDefault" checked> 融合结果设为默认时间轴</label>'
    NEW_HTML2 = '<label class="pipe-check"><input type="checkbox" id="pipeHarmonyDefault"> 融合结果设为默认时间轴（目前未超过整混音）</label>'
    if text.count(OLD_HTML2) == 1:
        text = text.replace(OLD_HTML2, NEW_HTML2)
    html.write_text(text, encoding="utf-8")
    print("index.html: experimental labels")


if __name__ == "__main__":
    main()
