"""Remove the throwaway test projects created during verification, plus their stem caches."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

DATA = Path(r"D:\ChordStudio\data")
PROJECTS = DATA / "projects"
STEMS = DATA / "stems"

TEST_IDS = {
    "df63d264c1934c91ad4a128266ed59b2",   # BKS-Bulldozer_harmonic.wav
    "b79cfb613ee94a9a9919a4a918f3fcbe",   # mixture.wav
    "f49b13e297444a31bbe0239985007bd3",   # 虚拟_harmonic.wav
    "45dcafe48d7f486cab7676086dc684b0",   # virtual.wav
    "92ff08da1d6a4d27836eb8d49c70d2f8",   # 自检 6s 单声道
    "fb0851fc076845f280f77450a0a3e316",   # 自检 25s
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    removed = []
    for project_id in TEST_IDS:
        folder = PROJECTS / project_id
        if folder.is_dir():
            shutil.rmtree(folder)
            removed.append(project_id)
    print(f"removed {len(removed)} test projects")

    keep = set()
    for folder in PROJECTS.iterdir():
        project_file = folder / "project.json"
        if not project_file.is_file():
            continue
        project = json.loads(project_file.read_text(encoding="utf-8"))
        if not project.get("audio"):
            continue
        audio = folder / project["audio"]
        if audio.is_file():
            keep.add(sha256(audio))
    print(f"{len(keep)} live audio hashes")

    dropped = []
    for folder in STEMS.iterdir():
        if folder.is_dir() and folder.name not in keep:
            size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            shutil.rmtree(folder)
            dropped.append((folder.name[:10], round(size / 1024**2)))
    print("dropped stem caches:", dropped)


if __name__ == "__main__":
    main()
