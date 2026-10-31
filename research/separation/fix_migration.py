"""Finish migrating <data>/stems/<sha>/... into <data>/stems/<sha>/htdemucs/..."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(r"D:\ChordStudio\data\stems")

for sha_dir in ROOT.iterdir():
    if not sha_dir.is_dir():
        continue
    target = sha_dir / "htdemucs"
    pending = [sha_dir / "manifest.json", sha_dir / "wav", *sha_dir.glob("*.mp3")]
    pending = [path for path in pending if path.exists()]
    if not pending:
        continue
    target.mkdir(exist_ok=True)
    for path in pending:
        destination = target / path.name
        if destination.exists():
            print(f"skip existing {destination}")
            continue
        path.rename(destination)
    print(f"migrated {sha_dir.name[:10]}: {[p.name for p in pending]}")

print("\nlayout:")
for sha_dir in ROOT.iterdir():
    if sha_dir.is_dir():
        children = sorted(child.name for child in sha_dir.iterdir())
        print(f"  {sha_dir.name[:10]}: {children}")
