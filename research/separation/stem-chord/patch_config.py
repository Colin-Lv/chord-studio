"""Expose the separation models through /api/config so the pipeline panel can list them."""
from __future__ import annotations

from pathlib import Path

SERVER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\server.py")
text = SERVER.read_text(encoding="utf-8")

OLD = '''    @application.get("/api/config")
    def config():
        return dict(profiles=[dict(id=k, name=v) for k, v in PROFILES.items()], max_upload_bytes=MAX_UPLOAD,
                    default_profile="retained10", desktop_version="1.0.0", retained10_max_seconds=600,
                    editor_preview=os.environ.get("HARMONY_EDITOR_PREVIEW") == "1")'''
NEW = '''    @application.get("/api/config")
    def config():
        from .stems_worker import MODELS
        return dict(profiles=[dict(id=k, name=v) for k, v in PROFILES.items()], max_upload_bytes=MAX_UPLOAD,
                    default_profile="retained10", desktop_version="1.0.0", retained10_max_seconds=600,
                    stems_models=[dict(id=m["id"], label=m["label"], note=m.get("note", ""),
                                       stems=m.get("stems", 4)) for m in MODELS.values()],
                    editor_preview=os.environ.get("HARMONY_EDITOR_PREVIEW") == "1")'''
assert text.count(OLD) == 1, text.count(OLD)
SERVER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("config exposes stems models")
