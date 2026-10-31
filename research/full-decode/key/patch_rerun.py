"""Add the older 虚拟 project to the re-analysis list."""
from pathlib import Path

PATH = Path(r"D:\ChordStudio\research\full-decode\key\rerun_songs.py")

OLD = '''PROJECTS = {
    "虚拟": "aa5a41797bb343e590abcf4d8b27ae54",
    "坏女孩": "0bc0304e01ed40d8baac6b76575b25b6",
    "沧海一粟": "e18b69d0edef4c5e91f6594520a816cb",
}'''

NEW = '''PROJECTS = {
    "虚拟(新项目)": "aa5a41797bb343e590abcf4d8b27ae54",
    "虚拟(旧v1项目)": "087d8b0c86414fe6b675cc622f505128",
    "坏女孩": "0bc0304e01ed40d8baac6b76575b25b6",
    "沧海一粟": "e18b69d0edef4c5e91f6594520a816cb",
}'''

text = PATH.read_text(encoding="utf-8")
count = text.count(OLD)
print(f"occurrences: {count}")
if count != 1:
    raise SystemExit("expected exactly one")
PATH.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("patched")
