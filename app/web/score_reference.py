"""Bounded local score parsing. References remain unverified and never edit audio.

MusicXML harmony/degree semantics follow the W3C MusicXML 4.0 reference:
https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/degree-alter/
No note-to-chord inference, OCR, repeat expansion, or concert-pitch conversion.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import re
import zipfile
import xml.etree.ElementTree as ET

from app.chord_semantics_v2 import parse_chord

MAX_BYTES = 8 * 1024 * 1024
MAX_ELEMENTS = 100_000
MAX_EVENTS = 20_000
MAX_MEASURES = 10_000
MAX_ZIP_MEMBERS = 128
_KIND = {
    "major": ("maj", ["1", "3", "5"]), "minor": ("min", ["1", "b3", "5"]),
    "augmented": ("aug", ["1", "3", "#5"]), "diminished": ("dim", ["1", "b3", "b5"]),
    "dominant": ("7", ["1", "3", "5", "b7"]), "major-seventh": ("maj7", ["1", "3", "5", "7"]),
    "minor-seventh": ("min7", ["1", "b3", "5", "b7"]),
    "diminished-seventh": ("dim7", ["1", "b3", "b5", "bb7"]),
    "half-diminished": ("hdim7", ["1", "b3", "b5", "b7"]),
    "major-minor": ("minmaj7", ["1", "b3", "5", "7"]),
    "major-sixth": ("maj6", ["1", "3", "5", "6"]), "minor-sixth": ("min6", ["1", "b3", "5", "6"]),
    "dominant-ninth": ("9", ["1", "3", "5", "b7", "9"]),
    "major-ninth": ("maj9", ["1", "3", "5", "7", "9"]),
    "minor-ninth": ("min9", ["1", "b3", "5", "b7", "9"]),
    "dominant-11th": (None, ["1", "3", "5", "b7", "9", "11"]),
    "major-11th": (None, ["1", "3", "5", "7", "9", "11"]),
    "minor-11th": ("min11", ["1", "b3", "5", "b7", "9", "11"]),
    "dominant-13th": ("13", ["1", "3", "5", "b7", "9", "11", "13"]),
    "major-13th": ("maj13", ["1", "3", "5", "7", "9", "11", "13"]),
    "minor-13th": ("min13", ["1", "b3", "5", "b7", "9", "11", "13"]),
    "suspended-second": ("sus2", ["1", "2", "5"]), "suspended-fourth": ("sus4", ["1", "4", "5"]),
    "power": (None, ["1", "5"]), "pedal": (None, ["1"]),
}


def _tag(node):
    return node.tag.rsplit("}", 1)[-1]


def _children(node, name):
    return [child for child in node if _tag(child) == name]


def _child(node, name):
    if node is None:
        return None
    return next((child for child in node if _tag(child) == name), None)


def _text(node, name, default=""):
    item = _child(node, name)
    return default if item is None else (item.text or "").strip()


def _number(value, name, low=-1_000_000, high=1_000_000):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"Invalid {name}")
    return result


def _integer(value, name, low=-100, high=100):
    result = _number(value, name, low, high)
    if not result.is_integer():
        raise ValueError(f"Unsupported fractional {name}")
    return int(result)


def _safe_xml(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Score XML exceeds 8 MiB")
    # Also catch UTF-16/32 declarations. XML entities never need network access.
    scan = data.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in scan or b"<!ENTITY" in scan:
        raise ValueError("DTD and ENTITY declarations are not supported")
    try:
        root = ET.fromstring(data)
    except (ET.ParseError, ValueError) as exc:
        raise ValueError("Invalid MusicXML") from exc
    count = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_ELEMENTS or depth > 64:
            raise ValueError("XML structure exceeds parsing limit")
        stack.extend((child, depth + 1) for child in node)
    return root


def _mxl_xml(data):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        with archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_MEMBERS or sum(i.file_size for i in members) > MAX_BYTES:
                raise ValueError("MXL archive exceeds parsing limits")
            names = set()
            for item in members:
                path = PurePosixPath(item.filename)
                if ("\\" in item.filename or path.is_absolute() or ".." in path.parts
                        or ":" in item.filename or item.flag_bits & 1):
                    raise ValueError("Unsafe MXL member path or encryption")
                if item.filename in names or item.file_size > MAX_BYTES:
                    raise ValueError("Duplicate or oversized MXL member")
                names.add(item.filename)
                if item.file_size / max(1, item.compress_size) > 200:
                    raise ValueError("MXL compression ratio exceeds limit")
            if "META-INF/container.xml" not in names:
                raise ValueError("MXL requires META-INF/container.xml")
            container = _safe_xml(archive.read("META-INF/container.xml"))
            roots = [node for node in container.iter() if _tag(node) == "rootfile"]
            roots = [node for node in roots if node.attrib.get("media-type") in
                     (None, "application/vnd.recordare.musicxml+xml")]
            if len(roots) != 1:
                raise ValueError("MXL requires one MusicXML rootfile")
            name = roots[0].attrib.get("full-path", "")
            root_path = PurePosixPath(name)
            if "\\" in name or ":" in name or root_path.is_absolute() or ".." in root_path.parts:
                raise ValueError("Unsafe MXL rootfile path")
            if name not in names:
                raise ValueError("MXL rootfile is not present")
            return archive.read(name), name
    except (zipfile.BadZipFile, RuntimeError, KeyError, OSError) as exc:
        raise ValueError("Invalid MXL archive") from exc


def _pitch(node, prefix):
    step = _text(node, prefix + "-step")
    if step not in "ABCDEFG" or len(step) != 1:
        raise ValueError("Unsupported harmony pitch")
    alteration = _integer(_text(node, prefix + "-alter", "0"), "pitch alteration", -2, 2)
    return step + ("#" * alteration if alteration > 0 else "b" * -alteration)


def _degree_token(number, alter):
    return ("#" * alter if alter > 0 else "b" * -alter) + str(number)


def _harmony(node):
    """Return raw, normalized and details; retain unsupported functions/poly-chords."""
    raw = ET.tostring(node, encoding="unicode")
    details = {"harmony_type": node.attrib.get("type", "explicit"),
               "print_object": node.attrib.get("print-object", "yes")}
    try:
        roots = _children(node, "root")
        kinds = _children(node, "kind")
        if len(kinds) != 1 or len(roots) > 1:
            raise ValueError("Stacked harmony requires manual interpretation")
        kind = (kinds[0].text or "").strip()
        details["musicxml_kind"] = kind
        if kind == "none":
            return kinds[0].attrib.get("text", "N.C."), "N", details
        if len(roots) != 1 or kind not in _KIND:
            raise ValueError("Functional or unsupported harmony kind requires review")
        root = _pitch(roots[0], "root")
        quality, degrees = _KIND[kind]
        degrees = list(degrees)
        changes = []
        for degree in _children(node, "degree"):
            number = _integer(_text(degree, "degree-value"), "degree", 1, 21)
            delta = _integer(_text(degree, "degree-alter", "0"), "degree alteration", -2, 2)
            operation = _text(degree, "degree-type")
            existing = next((d for d in degrees if int(re.sub("[b#]", "", d)) == number), None)
            if operation == "add":
                # MusicXML add is relative to dominant, including the minor seventh.
                token = _degree_token(number, delta + (-1 if number % 7 == 0 else 0))
                if token not in degrees:
                    degrees.append(token)
            elif operation in {"alter", "subtract"}:
                if existing is None:
                    raise ValueError("Degree alteration references absent chord tone")
                base = existing.count("#") - existing.count("b")
                target = _degree_token(number, base + delta)
                if operation == "subtract":
                    if target not in degrees:
                        raise ValueError("Degree subtraction references absent chord tone")
                    degrees.remove(target)
                else:
                    degrees.remove(existing)
                    degrees.append(target)
            else:
                raise ValueError("Unsupported degree operation")
            changes.append({"number": number, "alter": delta, "type": operation})
        label = root + ":" + (quality if quality and not changes else "(" + ",".join(degrees) + ")")
        bass = _child(node, "bass")
        inversion = _text(node, "inversion")
        if bass is not None:
            label += "/" + _pitch(bass, "bass")
        elif inversion:
            index = _integer(inversion, "inversion", 0, len(degrees) - 1)
            label += "/" + degrees[index]
        sem = parse_chord(label)
        if not sem.valid:
            raise ValueError(sem.parse_error or "Unrecognized chord")
        details["degrees"] = changes
        display = root + kinds[0].attrib.get("text", ":" + kind)
        if bass is not None:
            display += "/" + _pitch(bass, "bass")
        details["raw_musicxml"] = raw
        return display, sem.harte, details
    except ValueError as exc:
        details["normalization_warning"] = str(exc)
        return raw, None, details


def _meter(node):
    times = _children(node, "time")
    if not times:
        return None
    time = times[0]
    beats = _text(time, "beats")
    beat_type = _text(time, "beat-type")
    if len(_children(time, "beats")) != 1 or not re.fullmatch(r"[0-9]+(?:\+[0-9]+)*", beats):
        return None  # senza-misura / composite denominator require manual interpretation.
    numerator = sum(_integer(n, "meter numerator", 1, 64) for n in beats.split("+"))
    denominator = _integer(beat_type, "meter denominator", 1, 64)
    if denominator & (denominator - 1):
        raise ValueError("Unsupported meter denominator")
    return {"numerator": numerator, "denominator": denominator, "display": beats + "/" + beat_type}


def _musicxml(data, filename):
    root = _safe_xml(data)
    if _tag(root) != "score-partwise":
        raise ValueError("Only score-partwise MusicXML is supported")
    result = {"format": "musicxml", "events": [], "measures": [], "warnings": []}
    for part in _children(root, "part"):
        part_id = part.attrib.get("id", "part")
        divisions, meter, transpose, page = 1.0, None, 0, 1
        for ordinal, measure in enumerate(_children(part, "measure"), 1):
            if len(result["measures"]) >= MAX_MEASURES:
                raise ValueError("Score exceeds measure limit")
            number = measure.attrib.get("number", str(ordinal))
            measure_id = f"{part_id}:m{ordinal}"
            cursor, extent, pending = 0.0, 0.0, []
            repeat_context = []
            for item in measure:
                tag = _tag(item)
                if tag == "attributes":
                    if _text(item, "divisions"):
                        divisions = _number(_text(item, "divisions"), "divisions", .000001)
                    if _children(item, "time"):
                        meter = _meter(item)
                        if meter is None:
                            result["warnings"].append(f"{measure_id}: free/composite meter needs review")
                    tr = _child(item, "transpose")
                    if tr is not None:
                        transpose = _integer(_text(tr, "chromatic", "0"), "chromatic transposition")
                        transpose += 12 * _integer(_text(tr, "octave-change", "0"), "octave transposition", -8, 8)
                elif tag == "print" and item.attrib.get("new-page") == "yes" and ordinal != 1:
                    page += 1
                elif tag == "harmony":
                    raw, normalized, details = _harmony(item)
                    offset = cursor + _number(_text(item, "offset", "0"), "harmony offset") / divisions
                    event = {"id": f"event-{len(result['events']) + len(pending) + 1}", "kind": "chord_symbol",
                             "source_file": filename, "page": page, "part_id": part_id,
                             "measure_id": measure_id, "measure_number": number, "offset_quarter": offset,
                             "raw_chord": raw, "normalized_harte": normalized, "alternatives": [],
                             "confidence": {"kind": "explicit_symbol_parse", "value": None, "calibrated": False},
                             "review_status": "unreviewed", "transpose_chromatic": transpose,
                             "pitch_domain": "written", **details}
                    pending.append(event)
                elif tag in {"note", "forward", "backup"}:
                    duration = _number(_text(item, "duration", "0"), "duration", 0) / divisions
                    if tag == "backup":
                        cursor -= duration
                        if cursor < -.000001:
                            raise ValueError("MusicXML backup precedes measure start")
                    elif tag == "forward" or (_child(item, "chord") is None and _child(item, "grace") is None):
                        cursor += duration
                    extent = max(extent, cursor)
                elif tag == "barline":
                    for item2 in item:
                        if _tag(item2) in {"repeat", "ending"}:
                            repeat_context.append({"kind": _tag(item2), **item2.attrib})
            expected = meter["numerator"] * 4 / meter["denominator"] if meter else None
            implicit = measure.attrib.get("implicit") == "yes"
            length = extent if implicit and extent > 0 else max(extent, expected or 0)
            if length <= 0:
                length = None
                result["warnings"].append(f"{measure_id}: measure duration unavailable")
            record = {"id": measure_id, "part_id": part_id, "measure_number": number,
                      "ordinal": ordinal, "page": page, "measure_quarters": length, "meter": meter,
                      "pickup": bool(implicit and expected and extent < expected), "implicit": implicit,
                      "repeat_context": repeat_context, "transpose_chromatic": transpose}
            result["measures"].append(record)
            for event in pending:
                event.update(measure_quarters=length, meter=meter, repeat_context=repeat_context)
                event["alignment_eligible"] = bool(length and 0 <= event["offset_quarter"] < length)
                if not event["alignment_eligible"]:
                    result["warnings"].append(f"{event['id']}: offset or measure duration requires review")
                if event.get("normalization_warning"):
                    result["warnings"].append(f"{event['id']}: {event['normalization_warning']}")
                result["events"].append(event)
            if len(result["events"]) > MAX_EVENTS:
                raise ValueError("Score exceeds event limit")
    result["warnings"].extend([
        "Written-order measures only; repeats/endings/jumps are not expanded into audio playback.",
        "Written harmony is not converted to concert pitch; verify transposition, capo and arrangement before alignment.",
    ])
    return result


def _midi(data, filename):
    try:
        import mido
        midi = mido.MidiFile(file=io.BytesIO(data))
    except (ImportError, ValueError, EOFError, OSError) as exc:
        raise ValueError("MIDI cannot be parsed in this environment") from exc
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("Asynchronous/SMPTE MIDI requires separate timing review")
    events, metadata = [], []
    total_messages = 0
    for track_index, track in enumerate(midi.tracks):
        tick, active = 0, {}
        for message in track:
            total_messages += 1
            if total_messages > MAX_ELEMENTS:
                raise ValueError("MIDI exceeds message limit")
            tick += message.time
            if message.is_meta:
                if message.type in {"time_signature", "set_tempo", "key_signature", "marker", "text", "track_name"}:
                    metadata.append({"track": track_index, "offset_quarter": tick / midi.ticks_per_beat,
                                     "type": message.type, "data": message.dict()})
                continue
            if message.type not in {"note_on", "note_off"}:
                continue
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                active.setdefault(key, []).append((tick, message.velocity))
            elif active.get(key):
                start, velocity = active[key].pop(0)
                if tick <= start:
                    continue
                events.append({"id": f"event-{len(events)+1}", "kind": "note_evidence",
                               "source_file": filename, "page": None, "part_id": f"track-{track_index}",
                               "measure_id": None, "measure_number": None,
                               "offset_quarter": start / midi.ticks_per_beat,
                               "duration_quarter": (tick-start) / midi.ticks_per_beat,
                               "measure_quarters": None, "meter": None, "raw_chord": None,
                               "normalized_harte": None, "midi_note": message.note,
                               "channel": message.channel, "percussion": message.channel == 9,
                               "velocity": velocity, "confidence": {"kind": "midi_note_event", "value": None,
                                                                       "calibrated": False},
                               "review_status": "unreviewed", "transpose_chromatic": None,
                               "repeat_context": [], "alignment_eligible": False})
                if len(events) > MAX_EVENTS:
                    raise ValueError("MIDI exceeds note-event limit")
        if any(active.values()):
            metadata.append({"track": track_index, "type": "unclosed_notes", "count": sum(map(len, active.values()))})
    return {"format": "midi", "events": events, "measures": [], "metadata": metadata,
            "warnings": ["MIDI notes are evidence, not printed chord symbols or verified audio labels.",
                         "MIDI pickup, bar phase and audio alignment require explicit review; no measure mapping was guessed.",
                         "Sustain-pedal acoustic note duration is not inferred from MIDI note-off events."]}


def _gp_xml(data):
    """Read a small GP7/8 ZIP only, without extracting styles/audio/assets."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ZIP_MEMBERS or sum(i.file_size for i in entries) > MAX_BYTES:
                raise ValueError("GP archive exceeds parsing limits")
            names = set()
            for item in entries:
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or ".." in path.parts or "\\" in item.filename
                        or ":" in item.filename or item.flag_bits & 1):
                    raise ValueError("Unsafe GP member path or encryption")
                if item.filename in names or item.file_size > MAX_BYTES:
                    raise ValueError("Duplicate or oversized GP member")
                if item.file_size / max(1, item.compress_size) > 200:
                    raise ValueError("GP compression ratio exceeds limit")
                names.add(item.filename)
            if "Content/score.gpif" not in names:
                raise ValueError("Only GP7/8 ZIP with Content/score.gpif is supported")
            return archive.read("Content/score.gpif")
    except (zipfile.BadZipFile, RuntimeError, KeyError, OSError) as exc:
        raise ValueError("Invalid GP7/8 ZIP archive") from exc


def _id_table(root, collection, item_name):
    container = _child(root, collection)
    if container is None:
        raise ValueError(f"GPIF requires {collection}")
    result = {}
    for node in _children(container, item_name):
        identity = node.attrib.get("id")
        if not identity or identity in result:
            raise ValueError(f"Missing or duplicate GPIF {item_name} identity")
        result[identity] = node
    return result


def _gpif(data, filename):
    """Resolve MasterBars -> Bars -> Voices -> Beats -> Rhythm and staff chord IDs.

    GPIF identity/order conventions verified against alphaTab's own importer:
    https://github.com/CoderLine/alphaTab/blob/develop/packages/alphatab/src/importer/GpifParser.ts
    Deliberately supports explicit names only, not a full Guitar Pro renderer.
    """
    root = _safe_xml(data)
    if _tag(root) != "GPIF":
        raise ValueError("Expected GPIF root")
    track_table = _id_table(root, "Tracks", "Track")
    bars = _id_table(root, "Bars", "Bar")
    voices = _id_table(root, "Voices", "Voice")
    beats = _id_table(root, "Beats", "Beat")
    rhythms = _id_table(root, "Rhythms", "Rhythm")
    master_track = _child(root, "MasterTrack")
    master_bars = _child(root, "MasterBars")
    if master_track is None or master_bars is None:
        raise ValueError("GPIF requires MasterTrack/MasterBars")
    track_order = _text(master_track, "Tracks").split()
    if len(track_order) != len(set(track_order)) or set(track_order) != set(track_table):
        raise ValueError("GPIF MasterTrack must identify every track uniquely")
    score = _child(root, "Score")
    context = {key.lower(): _text(score, key) if score is not None else ""
               for key in ("Title", "Artist", "SubTitle", "Album", "Words", "Music", "Copyright", "Tabber")}
    context["arrangement_version"] = context["subtitle"] or context["title"]
    staves, tracks, libraries = [], [], []
    for track_id in track_order:
        track = track_table[track_id]
        staff_root = _child(track, "Staves")
        if staff_root is None or not _children(staff_root, "Staff"):
            raise ValueError("GPIF staff mapping is unavailable; export MusicXML for this version")
        transpose = _child(track, "Transpose")
        track_context = {"track_id": track_id, "track_name": _text(track, "Name"),
                         "transpose_chromatic": _integer(_text(transpose, "Chromatic", "0"), "GP transposition")
                         + 12 * _integer(_text(transpose, "Octave", "0"), "GP octave", -8, 8)
                         if transpose is not None else None}
        tracks.append(track_context)
        for staff_index, staff in enumerate(_children(staff_root, "Staff")):
            properties = _child(staff, "Properties")
            property_map = {}
            for prop in _children(properties, "Property") if properties is not None else []:
                name = prop.attrib.get("name")
                if name in property_map:
                    raise ValueError("Duplicate GPIF staff property")
                property_map[name] = prop
            capo_node = property_map.get("CapoFret")
            partial_capo = property_map.get("PartialCapoFret")
            partial_flags = property_map.get("PartialCapoStringFlags")
            tuning_node = property_map.get("Tuning")
            tuning_text = _text(tuning_node, "Pitches") if tuning_node is not None else ""
            tuning = [_integer(n, "MIDI tuning pitch", 0, 127) for n in tuning_text.split()]
            staff_context = {**track_context, "staff_index": staff_index,
                             "part_id": f"track-{track_id}:staff-{staff_index}",
                             "capo": _integer(_text(capo_node, "Fret"), "capo", 0, 24) if capo_node is not None else None,
                             "partial_capo": _integer(_text(partial_capo, "Fret"), "partial capo", 0, 24) if partial_capo is not None else None,
                             "partial_capo_string_flags": _text(partial_flags, "Bitset") if partial_flags is not None else None,
                             "tuning_midi_low_to_high": tuning,
                             "fingering_key": None, "sounding_key": None,
                             "pitch_domain": "named_diagram_unknown_concert"}
            chord_names = {}
            for collection in ("ChordCollection", "DiagramCollection"):
                library = property_map.get(collection)
                if library is None:
                    continue
                for item in library.iter():
                    if _tag(item) != "Item":
                        continue
                    chord_id = item.attrib.get("id")
                    if chord_id is None:
                        raise ValueError("GPIF chord library item lacks identity")
                    name = item.attrib.get("name", "")
                    if len(name) > 256:
                        raise ValueError("Oversized GPIF chord name")
                    chord_names.setdefault(chord_id, [])
                    if name not in chord_names[chord_id]:
                        chord_names[chord_id].append(name)
                    libraries.append({**staff_context, "chord_id": chord_id,
                                      "raw_chord": name, "collection": collection, "used": False})
            staves.append((staff_context, chord_names))
    durations = {}
    base_values = {"DoubleWhole": 8, "Whole": 4, "Half": 2, "Quarter": 1, "Eighth": .5,
                   "16th": .25, "32nd": .125, "64th": .0625, "128th": .03125, "256th": .015625}
    warnings = []
    for identity, rhythm in rhythms.items():
        value = base_values.get(_text(rhythm, "NoteValue"))
        if value is None:
            durations[identity] = None
            warnings.append(f"rhythm-{identity}: unsupported rhythm; subsequent beat offsets need review")
            continue
        dot = _child(rhythm, "AugmentationDot")
        if dot is not None:
            count = _integer(dot.attrib.get("count", "0"), "dots", 0, 4)
            value *= 2 - 2 ** -count
        tuplet = _child(rhythm, "PrimaryTuplet")
        if tuplet is not None:
            value *= (_integer(tuplet.attrib.get("den"), "tuplet denominator", 1, 64)
                      / _integer(tuplet.attrib.get("num"), "tuplet numerator", 1, 64))
        durations[identity] = value
    events, measures, used_libraries = [], [], set()
    traversed_beats = 0
    anacrusis = _child(master_track, "Anacrusis") is not None
    for ordinal, master in enumerate(_children(master_bars, "MasterBar"), 1):
        if ordinal > MAX_MEASURES:
            raise ValueError("GPIF exceeds measure limit")
        bar_ids = _text(master, "Bars").split()
        if len(bar_ids) != len(staves):
            raise ValueError("GPIF bar/staff mapping is incomplete or unsupported")
        time = _text(master, "Time")
        match = re.fullmatch(r"([0-9]+)\/([0-9]+)", time)
        meter = None
        expected = None
        if match:
            numerator = _integer(match[1], "GP numerator", 1, 64)
            denominator = _integer(match[2], "GP denominator", 1, 64)
            if denominator & (denominator - 1):
                raise ValueError("Unsupported GPIF denominator")
            meter = {"numerator": numerator, "denominator": denominator}
            expected = numerator * 4 / denominator
        free_time = _child(master, "FreeTime") is not None
        repeats = [{"kind": _tag(node), **node.attrib, "text": (node.text or "").strip()}
                   for node in master if _tag(node) in {"Repeat", "AlternateEndings", "Directions"}]
        key = _child(master, "Key")
        key_hint = {"accidental_count": _text(key, "AccidentalCount"), "mode": _text(key, "Mode")}
        section = _child(master, "Section")
        section_hint = {"letter": _text(section, "Letter"), "text": _text(section, "Text")} if section is not None else None
        for bar_id, (staff_context, chord_names) in zip(bar_ids, staves):
            if bar_id == "-1":
                continue
            if bar_id not in bars:
                raise ValueError("GPIF MasterBar references missing Bar")
            measure_id = f"{staff_context['part_id']}:m{ordinal}"
            pending, extents = [], []
            for voice_id in _text(bars[bar_id], "Voices").split():
                if voice_id == "-1":
                    continue
                if voice_id not in voices:
                    raise ValueError("GPIF Bar references missing Voice")
                cursor = 0.0
                for beat_id in _text(voices[voice_id], "Beats").split():
                    traversed_beats += 1
                    if traversed_beats > MAX_ELEMENTS:
                        raise ValueError("GPIF referenced beat expansion exceeds limit")
                    if beat_id not in beats:
                        raise ValueError("GPIF Voice references missing Beat")
                    beat = beats[beat_id]
                    rhythm_node = _child(beat, "Rhythm")
                    rhythm_id = rhythm_node.attrib.get("ref") if rhythm_node is not None else None
                    if rhythm_id not in rhythms:
                        raise ValueError("GPIF Beat references missing Rhythm")
                    duration = durations[rhythm_id]
                    grace = _child(beat, "GraceNotes") is not None
                    chord_id = _text(beat, "Chord")
                    if chord_id:
                        names = chord_names.get(chord_id, [])
                        raw = names[0] if names else None
                        sem = parse_chord(raw)
                        event = {"id": f"event-{len(events)+len(pending)+1}", "kind": "chord_symbol",
                                 "source_file": filename, "page": None, "measure_id": measure_id,
                                 "measure_number": str(ordinal), "offset_quarter": cursor,
                                 "duration_quarter": duration, "raw_chord": raw,
                                 "normalized_harte": sem.harte if sem.valid and len(names) == 1 else None,
                                 "alternatives": names[1:], "review_status": "unreviewed",
                                 "confidence": {"kind": "explicit_gpif_chord_reference", "value": None, "calibrated": False},
                                 "repeat_context": repeats, "key_signature_hint": key_hint, "section_hint": section_hint,
                                 "grace": grace, "gpif_refs": {"bar": bar_id, "voice": voice_id, "beat": beat_id,
                                                                  "rhythm": rhythm_id, "chord": chord_id},
                                 **context, **staff_context}
                        if not names or not raw or len(names) != 1:
                            event["normalization_warning"] = "Chord reference has missing/ambiguous explicit name; no note inference"
                            warnings.append(f"{event['id']}: {event['normalization_warning']}")
                        elif not sem.valid:
                            event["normalization_warning"] = "Explicit chord spelling needs review: " + (sem.parse_error or "unknown chord")
                            warnings.append(f"{event['id']}: {event['normalization_warning']}")
                        pending.append(event)
                        used_libraries.add((staff_context["part_id"], chord_id))
                    if not grace:
                        cursor = cursor + duration if cursor is not None and duration is not None else None
                extents.append(cursor)
            pickup = anacrusis and ordinal == 1
            length = max(extents) if pickup and extents and all(n is not None for n in extents) else expected
            # Partial voices can still carry exact early offsets. Unknown rhythms never do.
            timing_safe = bool(length and not free_time and all(n is not None and n <= length+.000001 for n in extents))
            if not timing_safe and pending:
                warnings.append(f"{measure_id}: incomplete/free/overfull timing requires review")
            measures.append({"id": measure_id, "ordinal": ordinal, "measure_number": str(ordinal),
                             "page": None, "meter": meter, "measure_quarters": length, "pickup": pickup,
                             "repeat_context": repeats, "key_signature_hint": key_hint, "section_hint": section_hint,
                             "timing_review_required": not timing_safe, **staff_context})
            for event in pending:
                event.update(meter=meter, measure_quarters=length,
                             alignment_eligible=bool(timing_safe and not event["grace"] and
                                                     event["offset_quarter"] is not None and 0 <= event["offset_quarter"] < length))
                events.append(event)
            if len(events) > MAX_EVENTS or len(measures) > MAX_MEASURES:
                raise ValueError("GPIF exceeds event/measure limit")
    for item in libraries:
        item["used"] = (item["part_id"], item["chord_id"]) in used_libraries
    warnings.extend(["Only beat-referenced explicit chord names are candidates; unused diagram libraries are not harmony progressions.",
                     "GPIF written-order measures only; repeats, endings, capo, tuning and arrangement need review before audio alignment.",
                     "Named diagram pitch domain is not assumed to be concert pitch. No note-to-chord inference was performed.",
                     "GPIF pages/layout are not rendered; page positions are unavailable."])
    return {"format": "gpif", "events": events, "measures": measures, "tracks": tracks,
            "score_context": context, "chord_libraries": libraries, "warnings": warnings}


def _printed_json(data, filename):
    """Human/OCR exchange format; original page evidence and ambiguity survive."""
    try:
        document = json.loads(data.decode("utf-8-sig"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid printed-score review JSON") from exc
    if not isinstance(document, dict) or document.get("schema") != "score_printed_review_v1":
        raise ValueError("Expected schema score_printed_review_v1; generic analysis JSON is not a score")
    rows = document.get("events", [])
    if not isinstance(rows, list) or len(rows) > MAX_EVENTS:
        raise ValueError("Invalid score events")
    context = {key: document.get(key) for key in
               ("source_library_id", "capo", "fingering_key", "sounding_key", "arrangement_version", "title")}
    def checked_context(values):
        values = dict(values)
        for name, value in values.items():
            if name == "capo":
                if value is not None:
                    values[name] = _integer(value, "capo", 0, 12)
            elif value is not None and (not isinstance(value, str) or len(value) > 1024):
                raise ValueError(f"Invalid printed score {name}")
        return values

    context = checked_context(context)
    events, measures = [], {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Invalid printed score event")
        raw = row.get("raw_chord")
        if raw is not None and (not isinstance(raw, str) or len(raw) > 256):
            raise ValueError("Invalid chord text")
        alternatives = row.get("alternatives", [])
        if not isinstance(alternatives, list) or len(alternatives) > 20 or any(
                not isinstance(a, str) or len(a) > 256 for a in alternatives):
            raise ValueError("Invalid chord alternatives")
        page = _integer(row.get("page", 1), "page", 1, 10_000)
        measure_number = str(row.get("measure_number", row.get("bar", "")))
        if not measure_number or len(measure_number) > 64:
            raise ValueError("Printed score events require a written bar/measure number")
        part_id = str(row.get("part_id", "printed"))
        measure_id = str(row.get("measure_id", f"{part_id}:p{page}:m{measure_number}"))
        if len(measure_id) > 256 or len(part_id) > 128:
            raise ValueError("Invalid measure/part identity")
        meter = row.get("meter")
        if meter is not None:
            if not isinstance(meter, dict):
                raise ValueError("Invalid printed meter")
            numerator = _integer(meter.get("numerator"), "meter numerator", 1, 64)
            denominator = _integer(meter.get("denominator"), "meter denominator", 1, 64)
            if denominator & (denominator-1):
                raise ValueError("Invalid printed meter denominator")
            meter = {"numerator": numerator, "denominator": denominator}
        length = row.get("measure_quarters")
        if length is not None:
            length = _number(length, "measure duration", .000001, 256)
        elif meter:
            length = meter["numerator"] * 4 / meter["denominator"]
        offset = row.get("offset_quarter")
        if offset is None and row.get("beat") is not None and meter:
            offset = (_number(row["beat"], "beat", 1, 65) - 1) * 4 / meter["denominator"]
        # No invisible first-beat assumption for glyphs whose positions are unknown.
        if offset is not None:
            offset = _number(offset, "offset", 0, 256)
        bbox = row.get("bbox")
        if bbox is not None:
            if row.get("bbox_format", "xywh") != "xywh" or row.get("bbox_units", "pixels") != "pixels":
                raise ValueError("Printed review bbox requires xywh in pixels; convert other coordinate formats explicitly")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bbox must be [x,y,width,height] in source pixels")
            bbox = [_number(v, "bbox coordinate", 0, 1_000_000) for v in bbox]
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError("Invalid bbox size")
        review = row.get("review_status", "unreviewed")
        if review not in {"unreviewed", "needs_review", "human_confirmed", "rejected"}:
            raise ValueError("Invalid review status")
        sem = parse_chord(raw)
        event_context = checked_context({key: row.get(key, value) for key, value in context.items()})
        repeat_context = row.get("repeat_context", [])
        if not isinstance(repeat_context, list) or len(repeat_context) > 64 or any(
                not isinstance(item, dict) for item in repeat_context):
            raise ValueError("Invalid printed repeat context")
        source_image_file = row.get("source_image_file")
        if source_image_file is not None and (not isinstance(source_image_file, str) or len(source_image_file) > 256):
            raise ValueError("Invalid source image filename")
        # Ambiguity never disappears merely because the primary text is parseable.
        event = {"id": f"event-{index+1}", "kind": "chord_symbol", "source_file": filename,
                 "page": page, "part_id": part_id, "measure_id": measure_id,
                 "measure_number": measure_number, "offset_quarter": offset, "measure_quarters": length,
                 "meter": meter, "raw_chord": raw, "normalized_harte": sem.harte if sem.valid else None,
                 "alternatives": alternatives, "bbox": bbox, "bbox_format": "xywh", "bbox_units": "pixels",
                 "review_status": "unreviewed", "source_review_status": review,
                 "confidence": {"kind": "printed_score_review_exchange", "value": None, "calibrated": False},
                 "transpose_chromatic": None, "repeat_context": repeat_context,
                 "source_image_file": source_image_file,
                 "pitch_domain": "fingering_unknown_concert", **event_context,
                 "alignment_eligible": bool(length and offset is not None and offset < length)}
        events.append(event)
        measure = {"id": measure_id, "part_id": part_id, "page": page, "measure_number": measure_number,
                   "measure_quarters": length, "meter": meter, "pickup": bool(row.get("pickup", False)),
                   "repeat_context": event["repeat_context"]}
        if measure_id in measures and measures[measure_id] != measure:
            raise ValueError("Conflicting identity/timing for printed measure")
        measures[measure_id] = measure
    return {"format": "printed_review_json", "events": events, "measures": list(measures.values()),
            "score_context": context,
            "warnings": ["Printed-score exchange is page-linked review evidence; it is not automatic OCR/OMR.",
                         "Verify capo, fingering key, sounding key, arrangement and repeats before audio alignment.",
                         "Review status does not make a score an audio training label."]}


def parse_score(data: bytes, filename: str) -> dict:
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError("Score must contain 1 byte to 8 MiB")
    filename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or len(filename) > 256:
        raise ValueError("Invalid score filename")
    extension = filename.rsplit(".", 1)[-1].lower()
    if extension in {"xml", "musicxml"}:
        result = _musicxml(data, filename)
    elif extension == "mxl":
        xml, member = _mxl_xml(data)
        result = _musicxml(xml, filename)
        result.update(format="mxl", archive_rootfile=member)
    elif extension in {"mid", "midi"}:
        result = _midi(data, filename)
    elif extension == "gpif":
        result = _gpif(data, filename)
    elif extension == "gp":
        result = _gpif(_gp_xml(data), filename)
        result.update(format="gp7_zip", archive_rootfile="Content/score.gpif")
    elif extension == "json":
        result = _printed_json(data, filename)
    else:
        raise ValueError("Use GP7 ZIP/GPIF, MusicXML/MXL/MIDI or page-linked score_printed_review_v1 JSON; PDF/images require visual review")
    result.update(schema="score_reference_candidates_v1", source_file=filename,
                  source_sha256=hashlib.sha256(data).hexdigest(), source_bytes=len(data),
                  manual_audio_labels=False, training_gold=False, alignment_status="unreviewed")
    return result
