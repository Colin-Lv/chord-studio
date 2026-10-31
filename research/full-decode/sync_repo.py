"""Sync the latest implementation into the git repo tree (repo = D:/ChordStudio/Dialogue).

Rules:
  * copy files that are missing in the repo, or newer at the source (mtime);
  * never delete; never overwrite a repo file that is newer than the source;
  * skip caches, compiled files and anything above MAX_BYTES.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(r"D:\ChordStudio\Dialogue")
INSTALLED = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend")
MAX_BYTES = 8 * 1024 * 1024
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".tmp", "node_modules"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".tmp"}

PAIRS = [
    (INSTALLED / "app", REPO / "app"),
    (INSTALLED / "frontend", REPO / "frontend"),
    (INSTALLED / "inference", REPO / "inference"),
]

ROOT_FILES = ["desktop_runtime.json", "desktop_backend.py", "inference_windowless.py"]


def main(dry: bool):
    copied, kept, skipped = [], [], []
    for source_root, target_root in PAIRS:
        if not source_root.is_dir():
            print("missing source:", source_root)
            continue
        for path in source_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source_root)
            if any(part in SKIP_DIRS for part in rel.parts) or path.suffix in SKIP_SUFFIX:
                skipped.append(str(rel))
                continue
            if path.stat().st_size > MAX_BYTES:
                skipped.append(f"{rel} ({path.stat().st_size} bytes)")
                continue
            target = target_root / rel
            if target.exists():
                if target.stat().st_mtime >= path.stat().st_mtime:
                    kept.append(str(rel))
                    continue
            copied.append(str(target_root.name / rel))
            if not dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ROOT_FILES:
        source = INSTALLED / name
        target = REPO / name
        if not source.is_file():
            continue
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            kept.append(name)
            continue
        copied.append(name)
        if not dry:
            shutil.copy2(source, target)
    print(f"{'DRY RUN' if dry else 'SYNCED'}: copy {len(copied)}, keep repo version {len(kept)}, "
          f"skipped {len(skipped)}")
    print("\ncopied (first 40):")
    for name in copied[:40]:
        print("  +", name)
    if len(copied) > 40:
        print(f"  ... and {len(copied) - 40} more")
    print("\nkept repo-newer (first 20):")
    for name in kept[:20]:
        print("  =", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry" in sys.argv))
