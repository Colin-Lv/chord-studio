"""Respect ?model= on the stems page and show the real track count in the stage label."""
from __future__ import annotations

from pathlib import Path

JS = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\stems.js")
text = JS.read_text(encoding="utf-8")

OLD_CONST = 'const PROJECT = params.get("project");'
NEW_CONST = 'const PROJECT = params.get("project");\nconst MODEL = params.get("model");'
assert text.count(OLD_CONST) == 1
text = text.replace(OLD_CONST, NEW_CONST)

OLD_FETCH = '''  try {
    const manifest = await api(`/api/projects/${PROJECT}/stems`);
    await start(manifest);
  } catch (error) {
    if (error.status === 404) showPending();
    else showError(error.message);
  }'''
NEW_FETCH = '''  try {
    const query = MODEL ? `?model=${encodeURIComponent(MODEL)}` : "";
    const manifest = await api(`/api/projects/${PROJECT}/stems${query}`);
    await start(manifest);
  } catch (error) {
    if (error.status === 404) showPending();
    else showError(error.message);
  }'''
assert text.count(OLD_FETCH) == 1
text = text.replace(OLD_FETCH, NEW_FETCH)
JS.write_text(text, encoding="utf-8")
print("stems.js: ?model= respected")

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")
OLD = '''    stage("separate", f"正在分离音轨（{MODELS[model]['label']} · 4 轨）", separate)'''
NEW = '''    stage("separate", f"正在分离音轨（{MODELS[model]['label']} · {len(stems_for(model))} 轨）", separate)'''
assert text.count(OLD) == 1, text.count(OLD)
WORKER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("stems_worker: stage label uses the real track count")
