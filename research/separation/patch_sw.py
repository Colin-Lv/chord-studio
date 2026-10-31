"""Add the six-stem BS-Roformer-SW checkpoint as a third audio-separator option."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD_MODELS = '''    "bs_roformer": dict(id="bs_roformer", label="人声优先", note="BS-RoFormer 两轨，GPU 约 3 分钟，人声更强", stems=2),
}'''
NEW_MODELS = '''    "bs_roformer": dict(id="bs_roformer", label="人声优先", note="BS-RoFormer 两轨，GPU 约 3 分钟，人声更强", stems=2),
    "bs_roformer_sw": dict(id="bs_roformer_sw", label="六轨精细",
                           note="BS-Roformer-SW 六轨，GPU 约 1 分钟；吉他/钢琴为部分估计",
                           stems=6, runner="registry", model_file="BS-Roformer-SW.ckpt"),
}'''
assert text.count(OLD_MODELS) == 1
text = text.replace(OLD_MODELS, NEW_MODELS)

OLD_BRANCH = '''        if model == "bs_roformer":
            outdir = work / "roformer"'''
NEW_BRANCH = '''        spec = MODELS[model]
        if spec.get("runner") == "registry":
            outdir = work / "registry"
            result = subprocess.run([python, "-B", str(ROOT / "app/web/separator_runner.py"),
                                     "--input", str(source_wav), "--outdir", str(outdir),
                                     "--model", spec["model_file"]],
                                    cwd=str(ROOT), env=env, capture_output=True, text=True,
                                    timeout=3600,
                                    creationflags=subprocess.CREATE_NO_WINDOW
                                    if os.name == "nt" else 0)
            if result.returncode:
                raise RuntimeError("分离失败：" + (result.stderr or result.stdout or "")[-1500:])
            wanted = {stem["id"] for stem in stems_for(model)}
            for produced in outdir.glob("*.wav"):
                name = produced.name.lower()
                for stem_id in wanted:
                    if f"({stem_id})" in name:
                        shutil.copyfile(produced, work / f"{stem_id}.wav")
            publish(stage_label="正在标注音轨信息")
            return
        if model == "bs_roformer":
            outdir = work / "roformer"'''
assert text.count(OLD_BRANCH) == 1
text = text.replace(OLD_BRANCH, NEW_BRANCH)
WORKER.write_text(text, encoding="utf-8")
print("stems_worker: bs_roformer_sw")

SERVER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\server.py")
text = SERVER.read_text(encoding="utf-8")
print("server unchanged" if "MODELS" in text else "server: no MODELS reference")
