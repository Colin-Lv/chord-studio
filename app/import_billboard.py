#!/usr/bin/env python
"""Import McGill Billboard salami_chords annotations into benchmark layout."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from chord_studio import parse_chord_name


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BillboardItem:
    item_id: str
    source: Path
    artist: str
    title: str
    key: str
    meter: str
    chord_count: int


def safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._-") or "item"


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def normalize_key(value: str) -> str:
    text = value.strip()
    if not text or text == "?":
        return ""
    text = text.replace("♭", "b").replace("♯", "#")
    if ":" in text:
        root, mode = text.split(":", 1)
        if mode.lower().startswith("min"):
            return f"{root}m"
        if mode.lower().startswith("maj"):
            return root
    return text


def normalize_meter(value: str) -> str:
    text = value.strip()
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    return f"{match.group(1)}/{match.group(2)}" if match else ""


def normalize_chord(value: str) -> str:
    text = value.strip()
    if not text or text in {"N", "X", "&pause", "*", "?"}:
        return "N"
    text = text.replace("♭", "b").replace("♯", "#")
    text = text.replace(":min", "m").replace(":maj", "")
    text = text.replace(":hdim7", "m7b5").replace(":dim", "dim").replace(":aug", "aug")
    text = text.replace(":maj7", "maj7").replace(":min7", "m7").replace(":7", "7")
    root, _ = parse_chord_name(text.split("/", 1)[0])
    return text if root is not None else "N"


def parse_salami_chords(path: Path) -> tuple[dict[str, str], list[tuple[float, float, str]]]:
    metadata: dict[str, str] = {}
    chords: list[tuple[float, float, str]] = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.match(r"#\s*([^:]+):\s*(.*)$", line)
            if match:
                metadata[match.group(1).strip().lower()] = match.group(2).strip()
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        start = parse_float(parts[0])
        end = parse_float(parts[1])
        if start is None or end is None or end <= start:
            continue
        chord = normalize_chord(parts[2])
        chords.append((start, end, chord))
    return metadata, chords


def import_item(source: Path, root: Path, output_root: Path) -> BillboardItem | None:
    metadata, chords = parse_salami_chords(source)
    if not chords:
        return None
    relative_parent = source.parent.relative_to(root)
    source_id = safe_id(str(relative_parent))
    item_id = f"billboard_{source_id}"
    target_dir = output_root / "billboard" / item_id
    target_dir.mkdir(parents=True, exist_ok=True)
    chord_target = target_dir / "chords.lab"
    chord_target.write_text(
        "\n".join(f"{start:.3f} {end:.3f} {chord}" for start, end, chord in chords) + "\n",
        encoding="utf-8",
    )
    return BillboardItem(
        item_id=item_id,
        source=source,
        artist=metadata.get("artist", ""),
        title=metadata.get("title", source.parent.name),
        key=normalize_key(metadata.get("tonic", "")),
        meter=normalize_meter(metadata.get("metre", metadata.get("meter", ""))),
        chord_count=len(chords),
    )


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_manifest(path: Path, items: list[BillboardItem], output_root: Path) -> None:
    base = path.parent.resolve()
    fields = [
        "id",
        "dataset",
        "split",
        "artist",
        "title",
        "audio",
        "prediction",
        "chord_reference",
        "beat_reference",
        "key",
        "meter",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            chord_reference = output_root / "billboard" / item.item_id / "chords.lab"
            writer.writerow(
                {
                    "id": item.item_id,
                    "dataset": "billboard",
                    "split": "dev",
                    "artist": item.artist,
                    "title": item.title,
                    "audio": "",
                    "prediction": f"predictions/{item.item_id}_analysis.json",
                    "chord_reference": relpath(chord_reference, base),
                    "beat_reference": "",
                    "key": item.key,
                    "meter": item.meter,
                    "notes": f"McGill Billboard salami_chords; chords={item.chord_count}; source={item.source}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing Billboard salami_chords.txt files.")
    parser.add_argument("-o", "--manifest", type=Path, default=REPO_ROOT / "benchmarks" / "manifest.billboard.csv")
    parser.add_argument("--import-dir", type=Path, default=REPO_ROOT / "benchmarks" / "imported")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Billboard annotation root not found: {root}")
    output_root = args.import_dir.resolve()
    files = sorted(root.rglob("salami_chords.txt"))
    if not files:
        files = sorted(path for path in root.rglob("*.txt") if path.name.lower().endswith("chords.txt"))
    items = []
    for source in files:
        item = import_item(source, root, output_root)
        if item:
            items.append(item)
    write_manifest(args.manifest.resolve(), items, output_root)
    print(f"Imported items: {len(items)}")
    print(f"Wrote: {args.manifest}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
