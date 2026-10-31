"""Explicit cached-input loading for v2 experiments; no extraction fallback."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from apply_multisource_candidates import NNLS_CONTRACT_KEYS
from evaluate_chords import TimedChord, read_reference, validate_timeline, parse_seconds
from experiment_protocol_v2 import signature
from harmonic_note_features import NoteEvidence, _validate_evidence
from multisource_candidates_v2 import SOURCE_NAMES, interval_values


@dataclass(frozen=True)
class ReferenceInterval:
    start: float
    end: float
    chord: str | None
    valid_fields: tuple[str,...] | None = None
    field_mask: dict | None = None


def read_source(path, *, anchor=False):
    """Retain unknown/missing labels and overlapping source observations.

    Only raw anchors need a disjoint timeline; external supports are unions.
    Annotation-specific metadata filtering is deliberately absent here.
    """
    path=Path(path)
    if path.suffix.lower()==".json":
        data=json.loads(path.read_text(encoding="utf-8"))
        data=data.get("events",[]) if isinstance(data,dict) else data
        rows=[TimedChord(float(r["start"]),float(r["end"]),r.get("chord")) for r in data]
    elif path.suffix.lower()==".csv":
        with path.open(encoding="utf-8-sig",newline="") as stream:
            data=list(csv.DictReader(stream))
        rows=[TimedChord(parse_seconds(r["start"]),parse_seconds(r["end"]),
            r.get("chord",r.get("label",r.get("name")))) for r in data]
    else:
        rows=[]
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields=line.split(maxsplit=2)
            if len(fields)<2:
                raise ValueError("Source LAB row requires start/end")
            rows.append(TimedChord(parse_seconds(fields[0]),parse_seconds(fields[1]),
                                   fields[2].strip() if len(fields)==3 else None))
    rows.sort(key=lambda r:(r.start,r.end))
    for row in rows:
        start,_,_=interval_values(row)
        if start<0:
            raise ValueError("Negative source time")
    if anchor and any(a.end>b.start for a,b in zip(rows,rows[1:])):
        raise ValueError("Raw anchors must be disjoint; overlapping external sources are allowed")
    return rows,dict(rounding_overlap_repairs=0,policy="Exact source intervals and original labels retained")


def read_annotation(path):
    """JSON supports explicit field masks; LAB/CSV use conservative policy."""
    path=Path(path)
    if path.suffix.lower()==".json":
        data=json.loads(path.read_text(encoding="utf-8"))
        data=data.get("events",data.get("annotations",[])) if isinstance(data,dict) else data
        rows=[ReferenceInterval(float(r["start"]),float(r["end"]),r.get("chord"),
            tuple(r["valid_fields"]) if r.get("valid_fields") is not None else None,r.get("field_mask")) for r in data]
        rows.sort(key=lambda r:(r.start,r.end))
        if any(not np.isfinite([r.start,r.end]).all() or r.start<0 or r.end<=r.start for r in rows):
            raise ValueError("Invalid annotated interval")
        if any(a.end>b.start for a,b in zip(rows,rows[1:])):
            raise ValueError("Masked JSON references must have nonoverlapping intervals")
        return rows,dict(rounding_overlap_repairs=0)
    # Isophonics LAB may carry long-lived "Point" editor markers. Discard
    # only that explicit metadata token, retaining all other unknown labels
    # and missing rows so annotation coverage keeps their full time support.
    original=[row for row in read_source(path)[0] if row.chord!="Point"]
    rows=validate_timeline(original,str(path))
    return rows,dict(rounding_overlap_repairs=sum(a.end!=b.end for a,b in zip(original,rows)),
                    policy="Existing standard timeline repair only: <=2.001ms annotation rounding overlap; no new alignment")


def load_inputs(row, *, references=False, base=None, expected_nnls=None):
    base=Path(base or ".").resolve()
    names=["audio",*SOURCE_NAMES,"notes"]+(["reference"] if references else [])
    paths={name:(base/row[name]).resolve() for name in names}
    missing=[str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required cached inputs missing; acoustic inference forbidden: "+", ".join(missing))
    identities={name:signature(path) for name,path in paths.items()}
    for name,identity in row.get("input_identities",{}).items():
        if name in identities and identities[name]!=identity:
            raise ValueError("Input identity changed: "+name)
    sources={}
    repairs={}
    for name in SOURCE_NAMES:
        sources[name],repairs[name]=read_source(paths[name],anchor=name=="raw_madmom")
    if not sources["raw_madmom"]:
        raise ValueError("Raw source must contain intervals")
    with np.load(paths["notes"],allow_pickle=False) as data:
        evidence=NoteEvidence(data["times"],data["activations"],json.loads(str(data["metadata"])))
    _validate_evidence(evidence)
    identity=evidence.metadata["identity"]
    if identity.get("source_sha256")!=identities["audio"]["sha256"]:
        raise ValueError("NNLS evidence is for different audio")
    if identity.get("offset_seconds")!=0 or identity.get("requested_duration_seconds") is not None:
        raise ValueError("NNLS evidence must cover the whole source audio at offset zero")
    nnls={key:identity[key] for key in NNLS_CONTRACT_KEYS}
    if expected_nnls is not None and nnls!=expected_nnls:
        raise ValueError("NNLS extraction semantics differ from the model contract")
    result=dict(id=row["id"],raw=sources["raw_madmom"],sources=sources,evidence=evidence,
                input_identities=identities,nnls_contract=nnls,timeline_repairs=repairs)
    if references:
        result["reference"],repairs["reference"]=read_annotation(paths["reference"])
    return result


def standard_reference(rows):
    # The unchanged evaluator treats explicit X and reference gaps as unknown.
    return [TimedChord(r.start,r.end,r.chord if r.chord is not None and r.chord.strip() else "X") for r in rows]
