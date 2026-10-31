#!/usr/bin/env python
"""Import local Guitar Pro files as symbolic chord/beat references.

This importer is intentionally local-file only. It expects `.gp3`, `.gp4`,
`.gp5`, or `.gpx` files that the user is allowed to use, then exports the same
benchmark layout used by the MIR annotation importers:

    benchmarks/raw/<dataset>/<song_id>/chords.lab
    benchmarks/raw/<dataset>/<song_id>/beats.beats
    benchmarks/manifest.<dataset>.csv

The resulting manifest can be passed to `build_harmonic_knowledge_base.py` to
learn progression priors from symbolic scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parent.parent
GP_EXTENSIONS = {".gp", ".gp3", ".gp4", ".gp5", ".gpx"}
QUARTER_TICKS = 960.0
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_BY_ACCIDENTALS_MAJOR = {
    -7: "Cb",
    -6: "Gb",
    -5: "Db",
    -4: "Ab",
    -3: "Eb",
    -2: "Bb",
    -1: "F",
    0: "C",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
    5: "B",
    6: "F#",
    7: "C#",
}
KEY_BY_ACCIDENTALS_MINOR = {
    -7: "Abm",
    -6: "Ebm",
    -5: "Bbm",
    -4: "Fm",
    -3: "Cm",
    -2: "Gm",
    -1: "Dm",
    0: "Am",
    1: "Em",
    2: "Bm",
    3: "F#m",
    4: "C#m",
    5: "G#m",
    6: "D#m",
    7: "A#m",
}


@dataclass(frozen=True)
class TimedChord:
    start: float
    end: float
    chord: str


@dataclass(frozen=True)
class ImportResult:
    item_id: str
    title: str
    artist: str
    chords_lab: Path | None
    beats_file: Path | None
    metadata_file: Path
    key: str
    meter: str
    notes: str


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._-") or "guitarpro_item"


def load_guitarpro_module():
    try:
        import guitarpro  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyguitarpro. Install it with:\n"
            "  python -m pip install -r requirements_chord_studio.txt"
        ) from exc
    return guitarpro


def get_value(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def duration_value(value: Any, default: int = 4) -> int:
    if value is None:
        return default
    raw = get_value(value, "value", value)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    return parsed or default


def measure_quarters(header: Any) -> float:
    signature = get_value(header, "timeSignature", None)
    numerator = int(get_value(signature, "numerator", 4) or 4)
    denominator = duration_value(get_value(signature, "denominator", None), 4)
    return numerator * (4.0 / denominator)


def measure_meter(header: Any) -> str:
    signature = get_value(header, "timeSignature", None)
    numerator = int(get_value(signature, "numerator", 4) or 4)
    denominator = duration_value(get_value(signature, "denominator", None), 4)
    return f"{numerator}/{denominator}"


def beat_offset_quarters(beat: Any, measure_start_tick: float) -> float:
    start_tick = float(get_value(beat, "start", measure_start_tick) or measure_start_tick)
    offset = max(0.0, start_tick - measure_start_tick)
    return offset / QUARTER_TICKS


def normalize_root(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return NOTE_NAMES[value % 12]
    text = str(value).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return {
        "CSharp": "C#",
        "DSharp": "D#",
        "FSharp": "F#",
        "GSharp": "G#",
        "ASharp": "A#",
        "BFlat": "Bb",
        "EFlat": "Eb",
        "AFlat": "Ab",
        "DFlat": "Db",
        "GFlat": "Gb",
    }.get(text, text)


def normalize_chord_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    text = text.replace(" ", "")
    text = text.replace("maj", "maj").replace("min", "m")
    text = text.replace("°", "dim").replace("o", "dim")
    text = text.replace("Δ", "maj")
    return text


def gpif_text(parent: ET.Element | None, path: str, default: str = "") -> str:
    if parent is None:
        return default
    found = parent.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def gpif_tempo(root: ET.Element) -> float:
    for automation in root.findall("./MasterTrack/Automations/Automation"):
        if gpif_text(automation, "Type") != "Tempo":
            continue
        value = gpif_text(automation, "Value")
        parts = value.split()
        if parts:
            try:
                return float(parts[0])
            except ValueError:
                pass
    return 120.0


def gpif_key(master_bar: ET.Element | None) -> str:
    key = master_bar.find("Key") if master_bar is not None else None
    if key is None:
        return ""
    try:
        accidental_count = int(gpif_text(key, "AccidentalCount", "0"))
    except ValueError:
        accidental_count = 0
    mode = gpif_text(key, "Mode", "Major")
    if mode.lower().startswith("min"):
        return KEY_BY_ACCIDENTALS_MINOR.get(accidental_count, "")
    return KEY_BY_ACCIDENTALS_MAJOR.get(accidental_count, "")


def gpif_note_name(note: ET.Element | None) -> str:
    if note is None:
        return ""
    step = (note.attrib.get("step") or "").strip()
    accidental = (note.attrib.get("accidental") or "Natural").strip()
    suffix = {
        "Sharp": "#",
        "DoubleSharp": "##",
        "Flat": "b",
        "DoubleFlat": "bb",
        "Natural": "",
    }.get(accidental, "")
    return f"{step}{suffix}" if step else ""


def gpif_chord_label(chord: ET.Element) -> str:
    root = gpif_note_name(chord.find("KeyNote"))
    bass = gpif_note_name(chord.find("BassNote"))
    if not root:
        return ""
    degrees = []
    for degree in chord.findall("Degree"):
        if degree.attrib.get("omitted", "false").lower() == "true":
            continue
        degrees.append((degree.attrib.get("interval", ""), degree.attrib.get("alteration", "")))
    has_minor_third = ("Third", "Minor") in degrees
    has_major_third = ("Third", "Major") in degrees
    has_dim_fifth = ("Fifth", "Diminished") in degrees
    has_aug_fifth = ("Fifth", "Augmented") in degrees
    has_minor_seventh = ("Seventh", "Minor") in degrees
    has_major_seventh = ("Seventh", "Major") in degrees
    has_major_sixth = ("Sixth", "Major") in degrees
    has_ninth = any(interval == "Ninth" for interval, _ in degrees)
    has_fourth = ("Fourth", "Perfect") in degrees

    suffix = ""
    if has_minor_third and has_dim_fifth and has_minor_seventh:
        suffix = "m7b5"
    elif has_minor_third:
        suffix = "m"
    elif has_dim_fifth:
        suffix = "dim"
    elif has_aug_fifth:
        suffix = "aug"
    elif has_fourth and not (has_minor_third or has_major_third):
        suffix = "sus4"

    if has_major_seventh:
        suffix = f"{suffix}maj7" if suffix == "m" else "maj7"
    elif has_minor_seventh:
        suffix = f"{suffix}7" if suffix and not suffix.endswith("7") else "7"
    elif has_major_sixth:
        suffix = f"{suffix}6" if suffix else "6"
    if has_ninth:
        suffix = f"{suffix}9" if "7" in suffix else f"{suffix}add9"

    label = f"{root}{suffix}"
    if bass and bass != root:
        label = f"{label}/{bass}"
    return label


def gpif_rhythm_quarters(rhythm: ET.Element | None) -> float:
    if rhythm is None:
        return 1.0
    note_value = gpif_text(rhythm, "NoteValue", "Quarter")
    quarters = {
        "Long": 16.0,
        "DoubleWhole": 8.0,
        "Whole": 4.0,
        "Half": 2.0,
        "Quarter": 1.0,
        "Eighth": 0.5,
        "16th": 0.25,
        "32nd": 0.125,
        "64th": 0.0625,
    }.get(note_value, 1.0)
    dot = rhythm.find("AugmentationDot")
    if dot is not None:
        try:
            count = int(dot.attrib.get("count", "1"))
        except ValueError:
            count = 1
        addition = quarters
        for _ in range(count):
            addition *= 0.5
            quarters += addition
    tuplet = rhythm.find("PrimaryTuplet")
    if tuplet is not None:
        try:
            num = float(tuplet.attrib.get("num", "1"))
            den = float(tuplet.attrib.get("den", "1"))
            if num > 0:
                quarters *= den / num
        except ValueError:
            pass
    return quarters


def gpif_time_quarters(value: str) -> float:
    if "/" not in value:
        return 4.0
    numerator, denominator = value.split("/", 1)
    try:
        return float(numerator) * (4.0 / float(denominator))
    except ValueError:
        return 4.0


def gpif_beat_count(value: str) -> int:
    if "/" not in value:
        return 4
    numerator, _ = value.split("/", 1)
    try:
        return max(1, int(numerator))
    except ValueError:
        return 4


def gpif_read_score(path: Path) -> ET.Element | None:
    if not zipfile.is_zipfile(path):
        return None
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "Content/score.gpif" not in names:
            return None
        with archive.open("Content/score.gpif") as handle:
            return ET.parse(handle).getroot()


def gpif_id_map(root: ET.Element, collection: str, item: str) -> dict[str, ET.Element]:
    parent = root.find(collection)
    if parent is None:
        return {}
    return {child.attrib.get("id", str(index)): child for index, child in enumerate(parent.findall(item))}


def gpif_best_track_index(root: ET.Element, bars: dict[str, ET.Element], voices: dict[str, ET.Element], beats: dict[str, ET.Element]) -> int:
    track_count = len(root.findall("./Tracks/Track"))
    scores = [0] * max(1, track_count)
    for master_bar in root.findall("./MasterBars/MasterBar"):
        bar_ids = gpif_text(master_bar, "Bars").split()
        for track_index, bar_id in enumerate(bar_ids):
            bar = bars.get(bar_id)
            if bar is None:
                continue
            for voice_id in gpif_text(bar, "Voices").split():
                voice = voices.get(voice_id)
                if voice is None:
                    continue
                for beat_id in gpif_text(voice, "Beats").split():
                    beat = beats.get(beat_id)
                    if beat is not None and beat.find("Chord") is not None:
                        scores[track_index] += 1
    return max(range(len(scores)), key=lambda index: scores[index])


def extract_gpif_symbolic(root: ET.Element) -> tuple[list[TimedChord], list[tuple[float, int]], str, dict[str, Any]]:
    chord_defs = [element for element in root.iter("Chord") if element.find("KeyNote") is not None]
    bars = gpif_id_map(root, "./Bars", "Bar")
    voices = gpif_id_map(root, "./Voices", "Voice")
    beats_by_id = gpif_id_map(root, "./Beats", "Beat")
    rhythms = gpif_id_map(root, "./Rhythms", "Rhythm")
    track_index = gpif_best_track_index(root, bars, voices, beats_by_id)
    tempo = gpif_tempo(root)
    seconds_per_quarter = 60.0 / tempo
    chord_events: list[tuple[float, str]] = []
    beat_grid: list[tuple[float, int]] = []
    meters: list[str] = []
    keys: list[str] = []
    elapsed = 0.0

    for master_bar in root.findall("./MasterBars/MasterBar"):
        meter = gpif_text(master_bar, "Time", "4/4")
        meters.append(meter)
        key = gpif_key(master_bar)
        if key:
            keys.append(key)
        bar_quarters = gpif_time_quarters(meter)
        bar_seconds = bar_quarters * seconds_per_quarter
        beat_count = gpif_beat_count(meter)
        for beat_index in range(beat_count):
            beat_grid.append((elapsed + (bar_seconds * beat_index / beat_count), beat_index + 1))

        bar_ids = gpif_text(master_bar, "Bars").split()
        if track_index < len(bar_ids):
            bar = bars.get(bar_ids[track_index])
            offset_quarters = 0.0
            if bar is not None:
                voice_ids = [voice_id for voice_id in gpif_text(bar, "Voices").split() if voice_id != "-1"]
                voice = voices.get(voice_ids[0]) if voice_ids else None
                if voice is not None:
                    for beat_id in gpif_text(voice, "Beats").split():
                        beat = beats_by_id.get(beat_id)
                        if beat is None:
                            continue
                        chord_ref = beat.find("Chord")
                        if chord_ref is not None and chord_ref.text is not None:
                            try:
                                chord_index = int(chord_ref.text.strip())
                            except ValueError:
                                chord_index = -1
                            if 0 <= chord_index < len(chord_defs):
                                label = gpif_chord_label(chord_defs[chord_index])
                                if label:
                                    chord_events.append((elapsed + offset_quarters * seconds_per_quarter, label))
                        rhythm_ref = beat.find("Rhythm")
                        rhythm = rhythms.get(rhythm_ref.attrib.get("ref", "")) if rhythm_ref is not None else None
                        offset_quarters += gpif_rhythm_quarters(rhythm)
        elapsed += bar_seconds

    timed: list[TimedChord] = []
    compact_events: list[tuple[float, str]] = []
    for start, label in sorted(chord_events, key=lambda item: item[0]):
        if compact_events and abs(compact_events[-1][0] - start) < 0.025 and compact_events[-1][1] == label:
            continue
        compact_events.append((start, label))
    for index, (start, label) in enumerate(compact_events):
        end = compact_events[index + 1][0] if index + 1 < len(compact_events) else elapsed
        if end - start > 0.05:
            timed.append(TimedChord(start, end, label))
    timed = merge_adjacent_chords(timed)

    tracks = root.findall("./Tracks/Track")
    track_name = gpif_text(tracks[track_index], "Name") if track_index < len(tracks) else ""
    metadata = {
        "title": gpif_text(root, "./Score/Title"),
        "artist": gpif_text(root, "./Score/Artist") or gpif_text(root, "./Score/Copyright"),
        "tempo": tempo,
        "track": track_name,
        "track_index": track_index,
        "track_count": len(tracks),
        "measure_count": len(root.findall("./MasterBars/MasterBar")),
        "format": "gpif",
    }
    dominant_meter = max(set(meters), key=meters.count) if meters else "4/4"
    metadata["key"] = max(set(keys), key=keys.count) if keys else ""
    return timed, beat_grid, dominant_meter, metadata


def merge_adjacent_chords(chords: list[TimedChord], tolerance: float = 0.035) -> list[TimedChord]:
    merged: list[TimedChord] = []
    for item in sorted(chords, key=lambda chord: (chord.start, chord.end)):
        if (
            merged
            and merged[-1].chord == item.chord
            and abs(merged[-1].end - item.start) <= tolerance
        ):
            previous = merged[-1]
            merged[-1] = TimedChord(previous.start, max(previous.end, item.end), previous.chord)
        else:
            merged.append(item)
    return merged


def chord_from_effect(beat: Any) -> str:
    effect = get_value(beat, "effect", None)
    chord = get_value(effect, "chord", None)
    if chord is None:
        return ""

    for field in ("name", "text"):
        label = normalize_chord_name(str(get_value(chord, field, "") or ""))
        if label and label not in {"None", "N"}:
            return label

    root = normalize_root(get_value(chord, "root", None))
    chord_type = str(get_value(chord, "type", "") or "").lower()
    suffix = ""
    if "minor" in chord_type or chord_type in {"m", "min"}:
        suffix = "m"
    elif "majorseventh" in chord_type or "maj7" in chord_type:
        suffix = "maj7"
    elif "seventh" in chord_type or chord_type == "7":
        suffix = "7"
    elif "diminished" in chord_type or "dim" in chord_type:
        suffix = "dim"
    elif "augmented" in chord_type or "aug" in chord_type:
        suffix = "aug"
    elif "sus" in chord_type:
        suffix = "sus4"

    bass = normalize_root(get_value(chord, "bass", None))
    if root and bass and bass != root:
        return f"{root}{suffix}/{bass}"
    return f"{root}{suffix}" if root else ""


def track_score(track: Any) -> tuple[int, int]:
    percussion = int(bool(get_value(track, "isPercussionTrack", False)))
    chord_count = 0
    note_count = 0
    for measure in get_value(track, "measures", []) or []:
        for voice in get_value(measure, "voices", []) or []:
            for beat in get_value(voice, "beats", []) or []:
                if chord_from_effect(beat):
                    chord_count += 1
                note_count += len(get_value(beat, "notes", []) or [])
    return (0 if percussion else 1, chord_count * 10 + note_count)


def best_track(song: Any) -> Any:
    tracks = list(get_value(song, "tracks", []) or [])
    if not tracks:
        raise ValueError("No tracks found in Guitar Pro file.")
    return max(tracks, key=track_score)


def song_tempo(song: Any) -> float:
    tempo = get_value(song, "tempo", None)
    try:
        return float(tempo or 120.0)
    except (TypeError, ValueError):
        return 120.0


def first_key(song: Any) -> str:
    # Guitar Pro files do not always store a clean tonal key. Keep this optional
    # rather than inventing a key from accidentals.
    key = str(get_value(song, "key", "") or "").strip()
    if key in {"None", "0"}:
        return ""
    if key.startswith("KeySignature."):
        key = key.split(".", 1)[1]
    key = key.replace("Major", "").replace("Minor", "m")
    return key


def extract_symbolic(song: Any) -> tuple[list[TimedChord], list[tuple[float, int]], str, dict[str, Any]]:
    track = best_track(song)
    tempo = song_tempo(song)
    seconds_per_quarter = 60.0 / tempo
    chords: list[tuple[float, str]] = []
    beats: list[tuple[float, int]] = []
    metadata = {
        "title": get_value(song, "title", ""),
        "artist": get_value(song, "artist", ""),
        "tempo": tempo,
        "track": get_value(track, "name", ""),
        "track_count": len(get_value(song, "tracks", []) or []),
        "measure_count": len(get_value(song, "measureHeaders", []) or []),
    }

    elapsed = 0.0
    headers = list(get_value(song, "measureHeaders", []) or [])
    measures = list(get_value(track, "measures", []) or [])
    meters: list[str] = []
    for index, header in enumerate(headers):
        meter = measure_meter(header)
        meters.append(meter)
        quarters = measure_quarters(header)
        denominator = int(meter.split("/", 1)[1])
        denominator_quarters = 4.0 / denominator
        denominator_count = max(1, int(round(quarters / denominator_quarters)))
        for beat_index in range(denominator_count):
            beats.append((elapsed + beat_index * denominator_quarters * seconds_per_quarter, beat_index + 1))

        if index < len(measures):
            measure = measures[index]
            measure_start = float(get_value(header, "start", 0.0) or 0.0)
            for voice in get_value(measure, "voices", []) or []:
                for beat in get_value(voice, "beats", []) or []:
                    label = chord_from_effect(beat)
                    if not label:
                        continue
                    offset = beat_offset_quarters(beat, measure_start) * seconds_per_quarter
                    chords.append((elapsed + offset, label))
        elapsed += quarters * seconds_per_quarter

    end_time = elapsed
    timed: list[TimedChord] = []
    unique_events = []
    for start, chord in sorted(chords, key=lambda item: item[0]):
        if unique_events and abs(unique_events[-1][0] - start) < 0.025 and unique_events[-1][1] == chord:
            continue
        unique_events.append((start, chord))
    for index, (start, chord) in enumerate(unique_events):
        end = unique_events[index + 1][0] if index + 1 < len(unique_events) else end_time
        if end - start > 0.05:
            timed.append(TimedChord(start, end, chord))
    timed = merge_adjacent_chords(timed)

    dominant_meter = max(set(meters), key=meters.count) if meters else "4/4"
    return timed, beats, dominant_meter, metadata


def write_import(path: Path, dataset: str, output_root: Path) -> ImportResult:
    gpif_root = gpif_read_score(path)
    if gpif_root is not None:
        song = None
        chords, beats, meter, metadata = extract_gpif_symbolic(gpif_root)
    else:
        guitarpro = load_guitarpro_module()
        song = guitarpro.parse(str(path))
        chords, beats, meter, metadata = extract_symbolic(song)
    title = str(metadata.get("title") or path.stem).strip() or path.stem
    artist = str(metadata.get("artist") or "").strip()
    item_id = safe_id(f"{dataset}_{artist}_{title}_{path.stem}")
    item_dir = output_root / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    chords_lab = item_dir / "chords.lab" if chords else None
    if chords_lab:
        chords_lab.write_text(
            "\n".join(f"{item.start:.3f} {item.end:.3f} {item.chord}" for item in chords) + "\n",
            encoding="utf-8",
        )
    beats_file = item_dir / "beats.beats" if beats else None
    if beats_file:
        beats_file.write_text(
            "\n".join(f"{time:.3f} {beat}" for time, beat in beats) + "\n",
            encoding="utf-8",
        )
    metadata_file = item_dir / "metadata.json"
    metadata_file.write_text(
        json.dumps({**metadata, "source_file": str(path), "chord_events": len(chords)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    notes = "symbolic Guitar Pro import"
    if not chords:
        notes += "; no explicit chord names found"
    key = str(metadata.get("key") or "").strip()
    if not key and song is not None:
        key = first_key(song)
    return ImportResult(item_id, title, artist, chords_lab, beats_file, metadata_file, key, meter, notes)


def find_gp_files(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in GP_EXTENSIONS:
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in GP_EXTENSIONS)


def write_manifest(results: list[ImportResult], manifest_path: Path, dataset: str) -> None:
    fieldnames = [
        "id",
        "dataset",
        "split",
        "artist",
        "title",
        "key",
        "meter",
        "audio",
        "prediction",
        "chord_reference",
        "beat_reference",
        "notes",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "id": item.item_id,
                    "dataset": dataset,
                    "split": "symbolic",
                    "artist": item.artist,
                    "title": item.title,
                    "key": item.key,
                    "meter": item.meter,
                    "audio": "",
                    "prediction": "",
                    "chord_reference": (
                        str(item.chords_lab.relative_to(REPO_ROOT / "benchmarks")).replace("\\", "/")
                        if item.chords_lab
                        else ""
                    ),
                    "beat_reference": (
                        str(item.beats_file.relative_to(REPO_ROOT / "benchmarks")).replace("\\", "/")
                        if item.beats_file
                        else ""
                    ),
                    "notes": item.notes,
                }
            )


def write_error_report(errors: list[dict[str, str]], path: Path) -> None:
    if not errors:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_file", "error_type", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A GP file or a directory containing .gp/.gp3/.gp4/.gp5/.gpx files.")
    parser.add_argument("--dataset", default="guitarpro")
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "raw" / "guitarpro",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "manifest.guitarpro.csv",
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "guitarpro_import_errors.csv",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip unreadable files and write an error report instead of aborting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_root = resolve_path(args.output_root)
    files = find_gp_files(input_path)
    if not files:
        print(f"No Guitar Pro files found: {input_path}")
        return 1
    results = []
    errors: list[dict[str, str]] = []
    for path in files:
        try:
            results.append(write_import(path, args.dataset, output_root))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            errors.append(
                {
                    "source_file": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc).replace("\r", " ").replace("\n", " "),
                }
            )
    manifest_path = resolve_path(args.manifest)
    write_manifest(results, manifest_path, args.dataset)
    write_error_report(errors, resolve_path(args.error_report))
    print(
        json.dumps(
            {
                "files": len(files),
                "imported": len(results),
                "errors": len(errors),
                "with_chords": sum(1 for item in results if item.chords_lab),
                "manifest": str(manifest_path),
                "output_root": str(output_root),
                "error_report": str(resolve_path(args.error_report)) if errors else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
