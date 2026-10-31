"""Independent v2 chord semantics; no v1 parser or evaluator is imported.

Full pitch sets use ``mir_eval.chord.encode(..., reduce_extended_chords=True)``.
Its implicit root bass is deliberately NOT exposed as observed bass. Chord
tones and slash/pedal tones have separate bitmaps. Raw labels and degree
spelling survive canonicalization, including explicit extension omissions.

Reference: https://mir-eval.readthedocs.io/latest/api/chord.html
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Literal

import mir_eval

SCHEMA = "chord_semantics_v2"
STATES = ("chord", "N", "X", "missing")
EXTENSIONS = (9, 11, 13)
_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_DEGREES = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
_EMPTY_BITMAP = (0,) * 12


@dataclass(frozen=True, order=True)
class Degree:
    number: int
    alteration: int = 0
    omitted: bool = False
    token: str = ""


def parse_degree(token: str) -> Degree:
    match = re.fullmatch(r"(\*?)([b#]*)([1-9][0-9]*)", token)
    if match is None:
        raise ValueError(f"Invalid scale degree {token!r}")
    omitted, accidental, number = match.groups()
    return Degree(int(number), accidental.count("#") - accidental.count("b"), bool(omitted), token)


def _degree_token(degree: Degree) -> str:
    accidental = "#" * degree.alteration if degree.alteration > 0 else "b" * -degree.alteration
    return ("*" if degree.omitted else "") + accidental + str(degree.number)


@dataclass(frozen=True)
class ChordSemantics:
    raw_label: str | None
    state: Literal["chord", "N", "X", "missing"]
    harte: str | None = None
    canonical_label: str | None = None
    root_pc: int | None = None
    quality: str | None = None
    chord_bitmap: tuple[int, ...] = _EMPTY_BITMAP
    full_bitmap: tuple[int, ...] = _EMPTY_BITMAP
    explicit_bass: bool = False
    bass_pc: int | None = None
    bass_relative: int | None = None
    bass_degree: str | None = None
    degrees: tuple[Degree, ...] = ()
    expanded_degrees: tuple[Degree, ...] = ()
    extension_degrees: tuple[tuple[int, tuple[int, ...]], ...] = ()
    mentioned_extensions: tuple[int, ...] = ()
    triad_bitmap: tuple[int, ...] = ()
    tetrad_bitmap: tuple[int, ...] = ()
    parse_error: str | None = None

    @property
    def identity(self) -> tuple:
        if self.state != "chord":
            return (self.state,)
        # Degree spelling distinguishes upper-voice extensions from equivalent
        # pitch-class lower tones (e.g. an added 13 versus a sixth).
        return ("chord", self.root_pc, self.chord_bitmap, self.extension_degrees,
                self.explicit_bass, self.bass_relative)

    @property
    def harmonic_harte(self) -> str | None:
        return self.harte.split("/", 1)[0] if self.harte else self.harte

    @property
    def valid(self) -> bool:
        return self.state in {"chord", "N"}

    def extension(self, degree: int) -> tuple[int, ...]:
        return dict(self.extension_degrees).get(degree, ())


def _normalize_body(body: str) -> str:
    """Convert display shorthand into Harte without depending on v1 aliases."""
    match = re.fullmatch(r"([A-G][b#]*)(?::?)(.*)", body)
    if match is None:
        raise ValueError("Invalid chord root")
    root, suffix = match.groups()
    if ":" in suffix:
        raise ValueError("Multiple quality separators")
    quality, separator, modifiers = suffix.partition("(")
    aliases = {"": "maj", "m": "min", "-": "min", "major": "maj", "minor": "min",
               "M": "maj", "M7": "maj7", "M9": "maj9", "M11": "maj11", "M13": "maj13",
               "m6": "min6", "6": "maj6", "m7": "min7", "m9": "min9",
               "m11": "min11", "m13": "min13", "-7": "min7", "dom7": "7",
               "m7b5": "hdim7", "min7b5": "hdim7", "ø": "hdim7", "ø7": "hdim7",
               "o": "dim", "o7": "dim7", "+": "aug", "sus": "sus4",
               "mMaj7": "minmaj7", "mmaj7": "minmaj7", "mMaj9": "minmaj9",
               "mmaj9": "minmaj9"}
    quality = aliases.get(quality, quality)
    extra = []
    if quality in {"add9", "madd9", "add11", "madd11", "add13", "madd13"}:
        extra = [quality.split("add", 1)[1]]
        quality = "min" if quality.startswith("m") else "maj"
    if quality == "maj11":
        quality, extra = "maj7", ["9", "11"]
    if quality == "minmaj9":
        quality, extra = "minmaj7", ["9"]
    if separator:
        if not modifiers.endswith(")") or "(" in modifiers or ")" in modifiers[:-1]:
            raise ValueError("Malformed scale-degree parentheses")
        extra.extend(token.strip() for token in modifiers[:-1].split(","))
        # Harte explicit degree sets C:(...) have no assumed major quality.
        if suffix.startswith("("):
            quality = ""
    body = root + ":" + quality
    if extra:
        body += "(" + ",".join(extra) + ")"
    return body


@lru_cache(maxsize=8192)
def parse_chord(label: str | None) -> ChordSemantics:
    """Parse one label, keeping invalid/unknown distinct from N and missing.

    None/empty string are missing. Unknown spellings retain raw_label and an
    error reason. No malformed chord is silently changed to silence.
    """
    if label is not None and not isinstance(label, str):
        raise TypeError("Chord labels must be strings or None")
    if label is None or not label.strip():
        return ChordSemantics(label, "missing")
    text = label.strip().strip('"').strip("'")
    if text.lower() in {"n", "nc", "n.c.", "no_chord", "no-chord", "silence"}:
        return ChordSemantics(label, "N", "N", "N")
    if text.lower() in {"x", "unknown", "unrecognized"}:
        return ChordSemantics(label, "X", "X", "X")
    try:
        text = text.replace("♭", "b").replace("♯", "#").replace("Δ", "maj").replace("−", "-")
        if any(char.isspace() for char in text):
            raise ValueError("Whitespace inside chord label is unsupported")
        body, slash, bass = text.partition("/")
        normalized_body = _normalize_body(body)
        root_spelling = normalized_body.split(":", 1)[0]
        root = mir_eval.chord.pitch_class_to_semitone(root_spelling) % 12
        if slash:
            if re.fullmatch(r"[A-G][b#]*", bass):
                relative = (mir_eval.chord.pitch_class_to_semitone(bass) - root) % 12
                bass = _DEGREES[relative]
            else:
                parse_degree(bass)
                if bass.startswith("*"):
                    raise ValueError("An omitted bass is not a bass observation")
            normalized = normalized_body + "/" + bass
        else:
            normalized = normalized_body
        _, quality, written, _ = mir_eval.chord.split(normalized, reduce_extended_chords=False)
        _, reduced_quality, expanded, _ = mir_eval.chord.split(normalized, reduce_extended_chords=True)
        degrees = tuple(sorted(parse_degree(token) for token in written))
        expanded_degrees = tuple(sorted(parse_degree(token) for token in expanded))
        root, chord_bitmap, _ = mir_eval.chord.encode(normalized_body, reduce_extended_chords=True)
        _, full_bitmap, bass_relative = mir_eval.chord.encode(normalized, reduce_extended_chords=True)
        _, tetrad_bitmap, _ = mir_eval.chord.encode(normalized_body, reduce_extended_chords=False)
        extensions = []
        for number in EXTENSIONS:
            tokens = [degree for degree in expanded_degrees if degree.number == number]
            present = {degree.alteration for degree in tokens if not degree.omitted}
            present -= {degree.alteration for degree in tokens if degree.omitted}
            if present:
                extensions.append((number, tuple(sorted(present))))
        canonical = _NOTES[int(root)] + ":" + reduced_quality
        if expanded_degrees:
            canonical += "(" + ",".join(_degree_token(degree) for degree in expanded_degrees) + ")"
        if slash:
            canonical += "/" + _DEGREES[int(bass_relative)]
        return ChordSemantics(
            raw_label=label, state="chord", harte=normalized, canonical_label=canonical,
            root_pc=int(root), quality=quality,
            chord_bitmap=tuple(int(v) for v in chord_bitmap), full_bitmap=tuple(int(v) for v in full_bitmap),
            explicit_bass=bool(slash), bass_pc=(int(root) + int(bass_relative)) % 12 if slash else None,
            bass_relative=int(bass_relative) if slash else None, bass_degree=bass if slash else None,
            degrees=degrees, expanded_degrees=expanded_degrees,
            extension_degrees=tuple(extensions),
            mentioned_extensions=tuple(number for number in EXTENSIONS
                                       if any(d.number == number for d in expanded_degrees)),
            triad_bitmap=tuple(int(v) for v in tetrad_bitmap[:8]),
            tetrad_bitmap=tuple(int(v) for v in tetrad_bitmap))
    except (ValueError, KeyError, mir_eval.chord.InvalidChordException) as exc:
        return ChordSemantics(label, "X", "X", "X", parse_error=str(exc))


def same_chord(left: ChordSemantics, right: ChordSemantics, level: str = "full") -> bool:
    if level not in {"root", "triad", "tetrad", "full"}:
        raise ValueError(f"Unknown semantic comparison level {level!r}")
    if left.state != "chord" or right.state != "chord":
        return left.state == right.state
    if left.root_pc != right.root_pc:
        return False
    if level == "root":
        return True
    if level == "triad":
        return left.triad_bitmap == right.triad_bitmap
    if level == "tetrad":
        return left.tetrad_bitmap == right.tetrad_bitmap
    return left.identity == right.identity
