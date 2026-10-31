"""Independent aliases for repaired reference semantics; frozen v2 stays intact.

This version is NOT a drop-in feature encoder for a v1/v2 trained model.
"""
from dataclasses import dataclass, fields
from functools import lru_cache
import re

if __package__:
    from .chord_semantics_v2 import ChordSemantics, parse_chord as parse_v2
else:
    from chord_semantics_v2 import ChordSemantics, parse_chord as parse_v2

SCHEMA = "chord_semantics_v3"
ALIASES = {"7sus4": ("sus4", ("b7",)), "9sus4": ("sus4", ("b7", "9")),
           "7sus2": ("sus2", ("b7",)), "9sus2": ("sus2", ("b7", "9"))}


@dataclass(frozen=True)
class ChordSemanticsV3(ChordSemantics):
    schema: str = SCHEMA
    alias_applied: str | None = None


@lru_cache(maxsize=8192)
def parse_chord(label: str | None) -> ChordSemanticsV3:
    alias = None
    mapped = label
    if isinstance(label, str):
        text = label.strip().strip('"').strip("'").replace("♭", "b").replace("♯", "#")
        body, slash, bass = text.partition("/")
        match = re.fullmatch(r"([A-G][b#]*):?(7sus[24]|9sus[24])(?:\(([^()]*)\))?", body)
        if match:
            root, alias, modifiers = match.groups()
            quality, degrees = ALIASES[alias]
            tokens = list(degrees) + (modifiers.split(",") if modifiers else [])
            mapped = root + ":" + quality + "(" + ",".join(tokens) + ")" + ("/" + bass if slash else "")
    value = parse_v2(mapped)
    data = {f.name: getattr(value, f.name) for f in fields(ChordSemantics)}
    data["raw_label"] = label
    return ChordSemanticsV3(**data, alias_applied=alias)


def to_harte(label: str | None) -> str | None:
    value = parse_chord(label)
    if value.parse_error:
        raise ValueError(value.parse_error)
    return value.harte


def require_new_contract(contract: dict):
    if contract.get("semantics_schema") != SCHEMA or not contract.get("retrained_for_semantics"):
        raise ValueError("New semantics require a newly trained model and a v3 feature contract")
