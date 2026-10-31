"""Parse demucs' carriage-return progress bars by reading byte chunks, not lines."""
from __future__ import annotations

from pathlib import Path

WORKER = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\app\web\stems_worker.py")
text = WORKER.read_text(encoding="utf-8")

OLD = """        process = subprocess.Popen([str(part) for part in command], cwd=str(ROOT), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   errors="replace", bufsize=1,
                                   creationflags=subprocess.CREATE_NO_WINDOW
                                   if os.name == "nt" else 0)
        last = -10
        for line in process.stdout:                       # type: ignore[union-attr]
            match = re.search(r"(\\d{1,3})%\\|", line)
            if not match:
                continue
            percent = int(match.group(1))
            if percent >= last + 5:
                last = percent
                publish(stage_label=f"正在分离音轨（{percent}%）")
        if process.wait():
            raise RuntimeError("demucs 分离失败，请查看任务日志。")
"""

NEW = """        process = subprocess.Popen([str(part) for part in command], cwd=str(ROOT), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW
                                   if os.name == "nt" else 0)
        # demucs draws a carriage-return progress bar, so read chunks and split on CR/LF.
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
        if process.wait():
            raise RuntimeError("demucs 分离失败，请查看任务日志。")
"""

count = text.count(OLD)
print(f"occurrences: {count}")
if count != 1:
    raise SystemExit("expected exactly one")
WORKER.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("progress reader patched")
