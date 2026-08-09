#!/usr/bin/env python
"""Import public MIR annotations into the local benchmark layout."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


CHORD_EXTENSIONS = {".lab", ".chords"}
BEAT_EXTENSIONS = {".beats", ".beat", ".txt"}
TTL_EXTENSIONS = {".ttl"}


@dataclass(frozen=True)
class ImportedItem:
    item_id: str
    dataset: str
    artist: str
    title: str
    chord_reference: Path | None
    beat_reference: Path | None
    key: str
    meter: str
    notes: str


def safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "item"


def parse_timed_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            rows.append(parts)
    return rows


def parse_iso_duration(value: str) -> float:
    text = value.strip().strip('"')
    match = re.fullmatch(
        r"PT(?:(?P<hours>[0-9.]+)H)?(?:(?P<minutes>[0-9.]+)M)?(?:(?P<seconds>[0-9.]+)S)?",
        text,
    )
    if not match:
        return float(text)
    return (
        float(match.group("hours") or 0.0) * 3600.0
        + float(match.group("minutes") or 0.0) * 60.0
        + float(match.group("seconds") or 0.0)
    )


def ttl_blocks(text: str, prefix: str) -> list[str]:
    pattern = re.compile(rf"(?ms)^:{re.escape(prefix)}_[^\n]*\n.*?(?=^:\w+_|^:maker_|^:record_|\Z)")
    return [match.group(0) for match in pattern.finditer(text)]


def parse_ttl_annotations(path: Path) -> tuple[list[tuple[float, float, str]], list[tuple[float, int]], str]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    chords: list[tuple[float, float, str]] = []
    beats: list[tuple[float, int]] = []
    keys: list[tuple[float, float, str]] = []

    for block in ttl_blocks(text, "beat"):
        time_match = re.search(r'tl:at\s+"([^"]+)"', block)
        label_match = re.search(r'rdfs:label\s+"([^"]+)"', block)
        if not time_match:
            continue
        beat = int(float(label_match.group(1))) if label_match and is_float(label_match.group(1)) else 0
        beats.append((parse_iso_duration(time_match.group(1)), beat))

    for prefix, target in (("chord", chords), ("key", keys)):
        for block in ttl_blocks(text, prefix):
            begin_match = re.search(r'tl:beginsAt\s+"([^"]+)"', block)
            duration_match = re.search(r'tl:duration\s+"([^"]+)"', block)
            label_match = re.search(r'rdfs:label\s+"([^"]+)"', block)
            if not begin_match or not duration_match or not label_match:
                continue
            start = parse_iso_duration(begin_match.group(1))
            end = start + parse_iso_duration(duration_match.group(1))
            target.append((start, end, label_match.group(1)))

    key = ""
    tonal_keys = [
        (end - start, label)
        for start, end, label in keys
        if label and label.lower() not in {"silence", "n", "none"}
    ]
    if tonal_keys:
        key = normalize_key_label(max(tonal_keys, key=lambda item: item[0])[1])
    return chords, beats, key


def write_ttl_import(path: Path, item_dir: Path) -> tuple[Path | None, Path | None, str]:
    chords, beats, key = parse_ttl_annotations(path)
    chord_target = item_dir / "chords.lab" if chords else None
    beat_target = item_dir / "beats.beats" if beats else None
    item_dir.mkdir(parents=True, exist_ok=True)
    if chord_target:
        chord_target.write_text(
            "\n".join(
                f"{start:.3f} {end:.3f} {normalize_chord_label(label)}"
                for start, end, label in chords
            )
            + "\n",
            encoding="utf-8",
        )
    if beat_target:
        beat_target.write_text(
            "\n".join(
                f"{time:.3f} {beat}" if beat > 0 else f"{time:.3f}"
                for time, beat in beats
            )
            + "\n",
            encoding="utf-8",
        )
    return chord_target, beat_target, key


def looks_like_chord_file(path: Path) -> bool:
    if path.suffix.lower() not in CHORD_EXTENSIONS:
        return False
    rows = parse_timed_rows(path)
    return any(len(row) >= 3 and is_float(row[0]) and is_float(row[1]) for row in rows[:20])


def looks_like_beat_file(path: Path) -> bool:
    if path.suffix.lower() not in BEAT_EXTENSIONS:
        return False
    rows = parse_timed_rows(path)
    if not rows:
        return False
    numeric_rows = [
        row for row in rows[:30]
        if len(row) >= 1 and is_float(row[0]) and (len(row) == 1 or is_float(row[1]))
    ]
    return len(numeric_rows) >= max(1, min(5, len(rows[:30])))


def is_float(value: object) -> bool:
    try:
        float(str(value))
        return True
    except ValueError:
        return False


def normalize_chord_label(label: str) -> str:
    text = str(label).strip()
    if not text or text in {"N", "X", "n", "x", "&pause", "*"}:
        return "N"
    return text


def normalize_key_label(label: str) -> str:
    text = str(label or "").strip()
    if not text or text.lower() in {"silence", "n", "none"}:
        return ""
    if ":" in text:
        root, quality = text.split(":", 1)
        if quality.lower().startswith("min"):
            return f"{root}m"
        if quality.lower().startswith("maj"):
            return root
    return text


def normalize_chord_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for row in parse_timed_rows(source):
        if len(row) < 3 or not is_float(row[0]) or not is_float(row[1]):
            continue
        output.append(f"{float(row[0]):.3f} {float(row[1]):.3f} {normalize_chord_label(row[2])}")
    target.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")


def normalize_beat_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for row in parse_timed_rows(source):
        if not row or not is_float(row[0]):
            continue
        beat = int(float(row[1])) if len(row) > 1 and is_float(row[1]) else 0
        if beat > 0:
            output.append(f"{float(row[0]):.3f} {beat}")
        else:
            output.append(f"{float(row[0]):.3f}")
    target.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")


def infer_metadata(path: Path, root: Path, dataset: str) -> tuple[str, str, str]:
    relative = path.relative_to(root)
    parts = relative.parts
    stem = path.stem
    artist = ""
    title = stem
    if len(parts) >= 3:
        artist = parts[-3]
        title = parts[-2] if parts[-2].lower() not in {"chords", "beat", "beats", "annotations"} else stem
    elif len(parts) >= 2:
        title = parts[-2] if parts[-2].lower() not in {"chords", "beat", "beats", "annotations"} else stem
    item_id = safe_id(f"{dataset}_{relative.with_suffix('')}")
    return item_id, artist, title


def pair_annotations(root: Path, dataset: str, chord_glob: str, beat_glob: str) -> list[tuple[Path | None, Path | None]]:
    chord_files = [path for path in root.glob(chord_glob) if path.is_file() and looks_like_chord_file(path)]
    beat_files = [path for path in root.glob(beat_glob) if path.is_file() and looks_like_beat_file(path)]
    chord_by_dir = {safe_id(str(path.parent.relative_to(root))).lower(): path for path in chord_files}
    beat_by_dir = {safe_id(str(path.parent.relative_to(root))).lower(): path for path in beat_files}
    beat_by_stem = {safe_id(path.stem).lower(): path for path in beat_files}
    beat_by_parent = {safe_id(path.parent.name).lower(): path for path in beat_files}
    pairs: list[tuple[Path | None, Path | None]] = []
    used_chords: set[Path] = set()
    used_beats: set[Path] = set()

    for key, chord in chord_by_dir.items():
        beat = beat_by_dir.get(key)
        if beat is not None:
            pairs.append((chord, beat))
            used_chords.add(chord)
            used_beats.add(beat)

    for chord in chord_files:
        if chord in used_chords:
            continue
        keys = [safe_id(chord.stem).lower(), safe_id(chord.parent.name).lower()]
        beat = next((beat_by_stem.get(key) or beat_by_parent.get(key) for key in keys if key), None)
        pairs.append((chord, beat))
        used_chords.add(chord)
        if beat is not None:
            used_beats.add(beat)

    for beat in beat_files:
        if beat not in used_beats:
            pairs.append((None, beat))
    return pairs


def import_dataset(
    root: Path,
    dataset: str,
    output_root: Path,
    split: str,
    chord_glob: str,
    beat_glob: str,
    ttl_glob: str,
) -> list[ImportedItem]:
    items = []
    ttl_files = [path for path in root.glob(ttl_glob) if path.is_file() and path.suffix.lower() in TTL_EXTENSIONS]
    if ttl_files:
        for path in ttl_files:
            item_id, artist, title = infer_metadata(path, root, dataset)
            item_dir = output_root / dataset / item_id
            chord_target, beat_target, key = write_ttl_import(path, item_dir)
            if chord_target is None and beat_target is None:
                continue
            items.append(
                ImportedItem(
                    item_id=item_id,
                    dataset=dataset,
                    artist=artist,
                    title=title,
                    chord_reference=chord_target,
                    beat_reference=beat_target,
                    key=key,
                    meter="",
                    notes=f"imported from {path}",
                )
            )
        return items

    for chord_path, beat_path in pair_annotations(root, dataset, chord_glob, beat_glob):
        source = chord_path or beat_path
        if source is None:
            continue
        item_id, artist, title = infer_metadata(source, root, dataset)
        item_dir = output_root / dataset / item_id
        chord_target = item_dir / "chords.lab" if chord_path else None
        beat_target = item_dir / "beats.beats" if beat_path else None
        if chord_path and chord_target:
            normalize_chord_file(chord_path, chord_target)
        if beat_path and beat_target:
            normalize_beat_file(beat_path, beat_target)
        items.append(
            ImportedItem(
                item_id=item_id,
                dataset=dataset,
                artist=artist,
                title=title,
                chord_reference=chord_target,
                beat_reference=beat_target,
                key="",
                meter="",
                notes=f"imported from {source}",
            )
        )
    return items


def write_manifest(path: Path, items: list[ImportedItem], split: str, base: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "id": item.item_id,
                    "dataset": item.dataset,
                    "split": split,
                    "artist": item.artist,
                    "title": item.title,
                    "audio": "",
                    "prediction": f"predictions/{item.item_id}_analysis.json",
                    "chord_reference": relpath(item.chord_reference, base),
                    "beat_reference": relpath(item.beat_reference, base),
                    "key": item.key,
                    "meter": item.meter,
                    "notes": item.notes,
                }
            )


def relpath(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import public annotation files into benchmark layout.")
    parser.add_argument("root", type=Path, help="Raw annotation root.")
    parser.add_argument("--dataset", required=True, choices=["billboard", "isophonics", "ballroom", "salami", "rwc_pop", "local"])
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/imported"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--chord-glob", default="**/*.lab")
    parser.add_argument("--beat-glob", default="**/*.beats")
    parser.add_argument("--ttl-glob", default="**/*.ttl")
    parser.add_argument("--copy-raw", action="store_true", help="Reserved for future use; raw audio is not copied.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.copy_raw:
        print("Note: raw audio is not copied by this importer.")
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Annotation root not found: {root}")
    output_root = args.output_root.resolve()
    manifest = args.manifest or Path("benchmarks") / f"manifest.{args.dataset}.csv"
    items = import_dataset(
        root,
        args.dataset,
        output_root,
        args.split,
        args.chord_glob,
        args.beat_glob,
        args.ttl_glob,
    )
    write_manifest(manifest.resolve(), items, args.split, manifest.resolve().parent)
    print(f"Imported items: {len(items)}")
    print(f"Wrote: {manifest}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
