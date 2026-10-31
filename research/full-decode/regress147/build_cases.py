"""Build cases.json for the 147-case regression set (audio from the archive)."""
from __future__ import annotations

import json
from pathlib import Path

CASES = Path(r"D:\ChordStudio\Dialogue\benchmarks\reports\harmonic_key_v2_20261006\acoustics")
AUDIO = Path(r"D:\ChordStudio\_archive\2026-10-30\benchmarks\external_key_data\audio")
HERE = Path(__file__).resolve().parent


def main():
    manifest = {}
    missing = []
    for case in sorted(CASES.glob("*/case.json")):
        data = json.loads(case.read_text(encoding="utf-8"))
        case_id = data["id"]
        audio = AUDIO / f"{case_id}.mp3"
        if not audio.is_file():
            missing.append(case_id)
            continue
        manifest[case_id] = dict(audio=str(audio), reference=data.get("reference"),
                                 split=data.get("split"), dataset=data.get("dataset"),
                                 genre=data.get("genre"))
    (HERE / "cases.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"{len(manifest)} cases, {len(missing)} missing audio")
    if missing:
        print("missing:", missing[:10])


if __name__ == "__main__":
    main()
