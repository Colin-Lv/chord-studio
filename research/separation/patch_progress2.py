"""Make the progress label monotonic across htdemucs_ft's four sub-models."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD = '''        # demucs draws a carriage-return progress bar, so read chunks and split on CR/LF.
        last = -10
        buffer = ""
        while True:
            chunk = process.stdout.read(512) if process.stdout else b""
            if not chunk:
                break
            buffer += chunk.decode("utf-8", "replace")
            parts = re.split(r"[\\r\\n]+", buffer)
            buffer = parts.pop() if parts else ""
            for part in parts:
                match = re.search(r"(\\d{1,3})%\\|", part)
                if not match:
                    continue
                percent = int(match.group(1))
                if percent >= last + 5:
                    last = percent
                    publish(stage_label=f"正在分离音轨（{percent}%）")
'''
NEW = '''        # demucs draws a carriage-return progress bar, so read chunks and split on CR/LF.
        # htdemucs_ft runs four checkpoints: detect a restarted bar and fold it into one number.
        submodels = 4 if model == "htdemucs_ft" else 1
        cycle = 0
        last = -10
        shown = -10
        buffer = ""
        while True:
            chunk = process.stdout.read(512) if process.stdout else b""
            if not chunk:
                break
            buffer += chunk.decode("utf-8", "replace")
            parts = re.split(r"[\\r\\n]+", buffer)
            buffer = parts.pop() if parts else ""
            for part in parts:
                match = re.search(r"(\\d{1,3})%\\|", part)
                if not match:
                    continue
                percent = int(match.group(1))
                if percent + 5 < last:
                    cycle = min(submodels - 1, cycle + 1)
                last = percent
                overall = int((cycle + percent / 100) * 100 / submodels) if submodels > 1 else percent
                if overall >= shown + 3:
                    shown = overall
                    publish(stage_label=f"正在分离音轨（{overall}%）")
'''
assert text.count(OLD) == 1, text.count(OLD)
WORKER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("progress is now monotonic across sub-models")
