"""Apply the small, exact app.js edits for the key-mode switch (avoids shell quoting)."""
from pathlib import Path

PATH = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\app.js")

EDITS = [
    ('$("keyMode").value="madmom";',
     '$("keyMode").value=result.key_mode||"madmom";'),
]

text = PATH.read_text(encoding="utf-8")
for old, new in EDITS:
    count = text.count(old)
    print(f"{count} x {old[:60]}")
    if count != 1:
        raise SystemExit(f"expected exactly one occurrence, found {count}")
    text = text.replace(old, new)
PATH.write_text(text, encoding="utf-8")
print("patched", PATH)
