"""Print a slice of a JS file around a marker (avoids shell quoting problems)."""
import sys
from pathlib import Path

path = Path(sys.argv[1])
marker = sys.argv[2]
before = int(sys.argv[3]) if len(sys.argv) > 3 else 200
after = int(sys.argv[4]) if len(sys.argv) > 4 else 800
text = path.read_text(encoding="utf-8")
index = text.find(marker)
if index < 0:
    raise SystemExit(f"marker not found: {marker}")
print(text[max(0, index - before):index + after])
