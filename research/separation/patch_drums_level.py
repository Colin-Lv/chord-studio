"""Make quiet drum parts look quiet, and keep the track count honest when expanding."""
from __future__ import annotations

from pathlib import Path

BACKEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
WORKER = BACKEND / "app/web/drums_worker.py"
text = WORKER.read_text(encoding="utf-8")
OLD = """        _annotate(wav_dir / f"{part['id']}.wav", entry, bars, length)
        tracks.append(entry)
"""
NEW = """        _annotate(wav_dir / f"{part['id']}.wav", entry, bars, length)
        tracks.append(entry)
    loudest = max(track["rms_db"] for track in tracks)
    for track in tracks:
        # The grid normalises each row by itself, so a -70 dB cymbal would look as dense as
        # the kick. Carry an explicit level factor so the UI can draw quiet parts faintly.
        track["level_factor"] = round(max(0.05, min(1.0, 10 ** ((track["rms_db"] - loudest) / 20))), 3)
"""
assert text.count(OLD) == 1
WORKER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("drums_worker: level_factor")

JS = BACKEND / "frontend/stems.js"
text = JS.read_text(encoding="utf-8")

OLD_ALPHA = """        ctx.fillStyle = hexToRgba(track.color, 0.22 + 0.72 * value);"""
NEW_ALPHA = """        ctx.fillStyle = hexToRgba(track.color, (0.22 + 0.72 * value) * (track.level_factor ?? 1));"""
assert text.count(OLD_ALPHA) == 1
text = text.replace(OLD_ALPHA, NEW_ALPHA)

OLD_META = """  $("songMeta").textContent = `${fmt(manifest.duration)} · ${state.tracks.length} 轨 · ${sourceText}`;"""
NEW_META = """  $("songMeta").textContent = `${fmt(manifest.duration)} · ${state.tracks.length} 轨 · ${sourceText}`;
  state.metaSuffix = sourceText;"""
assert text.count(OLD_META) == 1
text = text.replace(OLD_META, NEW_META)

OLD_REBUILD = """  seek(time);
  renderDrumButton();"""
NEW_REBUILD = """  seek(time);
  $("songMeta").textContent = `${fmt(state.manifest.duration)} · ${state.tracks.length} 轨 · ${state.metaSuffix || ""}`;
  renderDrumButton();"""
assert text.count(OLD_REBUILD) == 1
text = text.replace(OLD_REBUILD, NEW_REBUILD)
JS.write_text(text, encoding="utf-8")
print("stems.js: level factor + track count")
