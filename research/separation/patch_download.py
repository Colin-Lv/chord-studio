"""Per-track WAV download in the multitrack view."""
from __future__ import annotations

from pathlib import Path

JS = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\stems.js")
text = JS.read_text(encoding="utf-8")

OLD = """    const buttons = document.createElement("span");
    buttons.className = "track-buttons";
    const mute = document.createElement("button");
    mute.textContent = "M";"""
NEW = """    const buttons = document.createElement("span");
    buttons.className = "track-buttons";
    const download = document.createElement("a");
    download.className = "track-download";
    download.textContent = "↓";
    download.title = `下载 ${track.name} WAV`;
    download.href = state.manifest.wav_urls?.[track.id] || state.drumParts?.wav_urls?.[track.id]
      || `/api/projects/${PROJECT}/stems/${track.wav}`;
    download.download = `${track.id}.wav`;
    const mute = document.createElement("button");
    mute.textContent = "M";"""
assert text.count(OLD) == 1
text = text.replace(OLD, NEW)

OLD_APPEND = """    buttons.append(mute, solo);"""
NEW_APPEND = """    buttons.append(download, mute, solo);"""
assert text.count(OLD_APPEND) == 1
text = text.replace(OLD_APPEND, NEW_APPEND)
JS.write_text(text, encoding="utf-8")
print("stems.js: per-track download")

CSS = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\stems.css")
text = CSS.read_text(encoding="utf-8")
text += """
.track-buttons .track-download { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 5px; border: 1px solid #3d4f53; background: #1b2629; color: var(--muted); font-size: 12px; text-decoration: none; }
.track-buttons .track-download:hover { color: var(--mint); border-color: var(--mint); }
"""
CSS.write_text(text, encoding="utf-8")
print("stems.css: download style")
