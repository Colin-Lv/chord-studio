"""Per-model stem lists: htdemucs_6s adds guitar and piano (six rows in the grid)."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_STEMS = '''STEMS = (
    dict(id="vocals", name="人声", short="VOX", instrument="主唱 / 和声", color="#b6e7ce"),
    dict(id="drums", name="鼓", short="DRM", instrument="鼓组与打击乐", color="#e6c07b"),
    dict(id="bass", name="贝斯", short="BAS", instrument="贝斯", color="#7fb3e6"),
    dict(id="other", name="其他", short="OTH", instrument="钢琴 / 吉他 / 铺底", color="#c9a0e6"),
)'''
NEW_STEMS = '''STEMS = (
    dict(id="vocals", name="人声", short="VOX", instrument="主唱 / 和声", color="#b6e7ce"),
    dict(id="drums", name="鼓", short="DRM", instrument="鼓组与打击乐", color="#e6c07b"),
    dict(id="bass", name="贝斯", short="BAS", instrument="贝斯", color="#7fb3e6"),
    dict(id="other", name="其他", short="OTH", instrument="钢琴 / 吉他 / 铺底", color="#c9a0e6"),
)
STEMS_6 = STEMS + (
    dict(id="guitar", name="吉他", short="GTR", instrument="吉他", color="#d98bd9"),
    dict(id="piano", name="钢琴", short="KEY", instrument="钢琴", color="#9fe08a"),
)'''
assert text.count(OLD_STEMS) == 1
text = text.replace(OLD_STEMS, NEW_STEMS)

OLD_MODELS = '''    "htdemucs_ft": dict(id="htdemucs_ft", label="精细", note="4 个微调模型集成，GPU 约 45 秒，质量更好"),
}'''
NEW_MODELS = '''    "htdemucs_ft": dict(id="htdemucs_ft", label="精细", note="4 个微调模型集成，GPU 约 45 秒，质量更好"),
    "htdemucs_6s": dict(id="htdemucs_6s", label="六轨", note="额外分出吉他/钢琴，整体质量弱于四轨模型", stems=6),
}'''
assert text.count(OLD_MODELS) == 1
text = text.replace(OLD_MODELS, NEW_MODELS)

# stems_for helper after available_models
OLD_AVAIL = '''def available_models(sha: str) -> list[dict]:'''
NEW_AVAIL = '''def stems_for(model: str):
    return STEMS_6 if MODELS.get(model, {}).get("stems") == 6 else STEMS


def available_models(sha: str) -> list[dict]:'''
assert text.count(OLD_AVAIL) == 1
text = text.replace(OLD_AVAIL, NEW_AVAIL)

# collect(): use the model's stems
OLD_COLLECT = '''    def collect():
        wanted = {stem["id"] for stem in STEMS}'''
NEW_COLLECT = '''    def collect():
        wanted = {stem["id"] for stem in stems_for(model)}'''
assert text.count(OLD_COLLECT) == 1
text = text.replace(OLD_COLLECT, NEW_COLLECT)

# annotate(): iterate the model's stems
OLD_ANNOTATE = '''        tracks = []
        for stem in STEMS:'''
NEW_ANNOTATE = '''        tracks = []
        for stem in stems_for(model):'''
assert text.count(OLD_ANNOTATE) == 1
text = text.replace(OLD_ANNOTATE, NEW_ANNOTATE)

WORKER.write_text(text, encoding="utf-8")
print("stems_worker: htdemucs_6s with six tracks")
