"""Show the reconstruction residual next to the model name."""
from __future__ import annotations

from pathlib import Path

JS = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\stems.js")
text = JS.read_text(encoding="utf-8")
OLD = '''  $("modelInfo").textContent = `${manifest.model?.label || manifest.model?.name || "htdemucs"} · ${manifest.model?.seconds ?? "—"} s`;'''
NEW = '''  const residual = manifest.reconstruction?.relative_db;
  $("modelInfo").textContent = `${manifest.model?.label || manifest.model?.name || "htdemucs"} · ${manifest.model?.seconds ?? "—"} s`
    + (residual == null ? "" : ` · 重构残差 ${Number(residual).toFixed(1)} dB`);'''
assert text.count(OLD) == 1, text.count(OLD)
JS.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("stems.js: reconstruction shown")
