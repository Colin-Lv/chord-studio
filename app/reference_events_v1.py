"""Canonical reference events, conservative masks and explicit revision views.

Legacy source policy is selected by the caller, never inferred from field order.
No acoustic evidence or gold labels are inferred from confidence or edited flags.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import re
import json
import os
from pathlib import Path
import uuid

import mir_eval

from chord_semantics_v3 import parse_chord, SCHEMA as SEMANTICS_SCHEMA

SCHEMA = "canonical_reference_events_v1"
FIELD_SEMANTICS_SCHEMA = "reference_field_semantics_v2"
FIELDS = ("state", "root", "quality", "seventh", "ext9", "ext11", "ext13", "bass", "tonic", "mode")
REVIEW_STATUSES = {"unreviewed", "pending_conflict", "listened_confirmed", "unresolved"}
NOTE_PC = {"C":0, "D":2, "E":4, "F":5, "G":7, "A":9, "B":11}


def key_definition(display, confirmed=None):
    result = {"display_key": display, "notation_key": display, "tonic_pc": None, "mode": None,
              "known_mask": {"tonic":False, "mode":False}, "task_status":"legacy_display_coordinate_only"}
    if confirmed is not None:
        pc, mode = confirmed.get("tonic_pc"), confirmed.get("mode")
        if isinstance(pc, bool) or not isinstance(pc, int) or pc not in range(12) or mode not in {"major", "minor"}:
            raise ValueError("Explicit tonic-mode confirmation requires pc 0..11 and major/minor")
        result.update(tonic_pc=pc, mode=mode, known_mask={"tonic":True,"mode":True}, task_status="explicit_tonic_mode_annotation")
    return result


def structural_mask(sem, key):
    chord = sem.state == "chord" and not sem.parse_error
    seventh_quality = sem.quality in {"7","maj7","min7","dim7","hdim7","minmaj7","9","maj9","min9","11","min11","13","min13"}
    return {"state": sem.state in {"chord","N"} and not sem.parse_error, "root":chord, "quality":chord,
            "seventh": chord and (seventh_quality or any(d.number == 7 for d in sem.degrees)),
            **{f"ext{n}": chord and n in sem.mentioned_extensions for n in (9,11,13)},
            "bass":chord and sem.explicit_bass, **key["known_mask"]}


def base_quality_family(sem):
    """Classify the written lower body; upper degrees never supply thirds/fifths.

    Reduced qualities preserve their base tones. Written lower-degree additions
    and omissions still apply, so 7(4,*3) and sus4(b7) agree. An upper #9 is
    not a minor third, #11 is not a diminished fifth, and 11 is not a sus4.
    Full pitch, extension and bass targets remain in their separate fields.
    """
    root, reduced, _, _ = mir_eval.chord.split(sem.harmonic_harte, reduce_extended_chords=True)
    lower = [d.token for d in sem.expanded_degrees if d.number <= 5]
    body = root + ":" + reduced + ("(" + ",".join(lower) + ")" if lower else "")
    _, bitmap, _ = mir_eval.chord.encode(body, reduce_extended_chords=True)
    tones = {i for i, present in enumerate(bitmap) if present}
    return ("augmented" if {4, 8} <= tones and 7 not in tones else "major" if 4 in tones else
            "diminished" if {3, 6} <= tones else "minor" if 3 in tones else
            "sus4" if 5 in tones else "sus2" if 2 in tones else
            "power" if 7 in tones else "other")


def require_current_field_semantics(record):
    if record.get("field_semantics_schema") != FIELD_SEMANTICS_SCHEMA:
        raise ValueError("Old field semantics: normalize into a new independent reference first")


def reference_binding(record):
    """Immutable source/clock/event origins; editable labels and reviews excluded."""
    fields = ("schema", "semantics_schema", "field_semantics_schema", "source_policy",
              "source_identity", "sample_rate", "bar_grid", "quantization_artifacts")
    event_fields = ("event_id", "source_index", "bar", "start_sample", "end_sample",
                    "raw_start", "raw_end", "raw_span")
    return {**{name: record.get(name) for name in fields},
            "event_origins": [{name: event.get(name) for name in event_fields}
                              for event in record["canonical_events"]]}


def field_values(sem, key, known):
    quality = sem.quality or ""
    family = base_quality_family(sem) if known["quality"] else None
    sevenths = {9} if quality=="dim7" else {11} if quality in {"maj7","maj9","minmaj7"} else {10} if quality in {"7","min7","hdim7","9","min9","11","min11","13","min13"} else set()
    for d in sem.degrees:
        if d.number == 7:
            if d.omitted:sevenths.discard((11+d.alteration)%12)
            else:sevenths.add((11+d.alteration)%12)
    return {"state":sem.state if known["state"] else None,"root":sem.root_pc if known["root"] else None,
            "quality":family if known["quality"] else None,"seventh":sorted(sevenths) if known["seventh"] else None,
            **{f"ext{n}":list(sem.extension(n)) if known[f"ext{n}"] else None for n in (9,11,13)},
            "bass":sem.bass_pc if known["bass"] else None,
            "tonic":key["tonic_pc"] if known["tonic"] else None,"mode":key["mode"] if known["mode"] else None}


def legacy_spans(data, source_policy):
    if source_policy == "legacy_studio_bars_spans":
        if "bars" not in data:
            raise ValueError("Declared legacy source bars.spans is absent")
        return [dict(s, bar=b["bar"]) for b in data["bars"] for s in b.get("spans", [])]
    if source_policy == "declared_top_level_events":
        if "events" not in data:
            raise ValueError("Declared top-level events source is absent")
        return deepcopy(data["events"])
    raise ValueError("An explicit supported legacy source policy is required")


def legacy_consistency(data):
    ss = legacy_spans(data, "legacy_studio_bars_spans")
    signature = lambda s: (s.get("start"), s.get("end"), s.get("chord_display") or s.get("chord"), s.get("roman_display") or s.get("roman"), s.get("local_key"))
    events = data.get("events")
    stale = events is not None and [signature(s) for s in ss] != [signature(s) for s in events]
    beats_diff = []
    for b in data.get("bars", []):
        for beat in b.get("beats", []):
            hits = [s for s in b.get("spans", []) if float(s["end"]) > float(beat["start"]) and float(s["start"]) < float(beat["end"])]
            labels = list(dict.fromkeys(s.get("chord_display") or s.get("chord") for s in hits))
            existing = beat.get("chord_display") or beat.get("chord")
            if labels and (len(labels) > 1 or existing != labels[0]):
                beats_diff.append({"bar":b["bar"], "beat":beat.get("beat"), "stored_label":existing,
                                   "canonical_labels":labels, "start":beat["start"], "end":beat["end"]})
    return {"spans_count":len(ss), "events_count":len(events) if events is not None else None,
            "events_stale":stale, "beat_conflicts":beats_diff,
            "legacy_derived_fields_status": {"events":"stale" if stale else "same_or_absent", "bars.beats":"stale_or_lossy" if beats_diff else "same"}}


def local_display_key(data, span):
    if span.get("local_key"):
        return span["local_key"]
    bar = span.get("bar")
    for section in data.get("modulations", []):
        if bar is not None and section.get("start_bar",0) <= bar and (section.get("end_bar") is None or bar <= section["end_bar"]):
            return section.get("key")
    return data.get("key")


def normalize_reference(data, *, source_policy, sample_rate, source_identity, origin_policy="legacy_export", conflicts=None, confirmed_key=None):
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("A declared positive integer analysis sample rate is required")
    raw = legacy_spans(data, source_policy)
    raw.sort(key=lambda s:(s["start"],s["end"]))
    events, artifacts = [], []
    prior_end = -1
    copy_range = (data.get("auto_completion") or {}).get("target_bars")
    for i, span in enumerate(raw):
        a, b = float(span["start"]), float(span["end"])
        if not math.isfinite(a+b) or a < 0 or b <= a:
            raise ValueError("Invalid reference interval")
        left, right = round(a * sample_rate), round(b * sample_rate)
        if right <= left:
            artifacts.append({"source_index":i, "reason":"sub_sample_interval", "raw_span":span})
            continue
        if left < prior_end:
            raise ValueError("Reference intervals overlap at the declared sample precision")
        prior_end = right
        label = span.get("chord_display") or span.get("chord")
        sem = parse_chord(label)
        key = key_definition(local_display_key(data,span), confirmed_key)
        known = structural_mask(sem,key)
        conflict_fields = (conflicts or {}).get(i, [])
        for field in conflict_fields:
            if field not in FIELDS:
                raise ValueError("Unknown conflicting field")
            known[field] = False
        in_copy = bool(copy_range and copy_range[0] <= int(span.get("bar",0)) <= copy_range[1])
        inherited = {k:v for k,v in span.items() if "inferred" in k or "refined" in k}
        origins = []
        if span.get("edited"):origins.append("legacy_edited_unverified")
        if in_copy:origins.append("auto_copy_possible_later_human_edits")
        if origin_policy == "user_rule_assisted" or data.get("progression_override") or data.get("bar_overrides"):origins.append("user_rule_assisted")
        if inherited:origins.append("inherited_automatic_inference")
        if not origins:origins.append("inherited_automatic_unreviewed")
        status = "pending_conflict" if conflict_fields else "unreviewed"
        events.append({"event_id":f"e{i:05d}", "source_index":i, "bar":span.get("bar"),
            "start_sample":left,"end_sample":right,"start":left/sample_rate,"end":right/sample_rate,
            "raw_start":a,"raw_end":b,"raw_label":label,"chord":sem.harte if not sem.parse_error else label,
            "semantics":asdict(sem),"known_mask":known,"field_values":field_values(sem,key,known),"training_mask":dict.fromkeys(FIELDS,False),
            "key_definition":key,"roman_stored":span.get("roman_display") or span.get("roman"),
            "provenance":{"origins":origins,"edited_flag":bool(span.get("edited")),"inherited_flags":inherited,
                          "confidence_stored":span.get("confidence"),"prediction_exposure":"legacy_export_prediction_visible"},
            "review":{"status":status,"confirmed_fields":[],"conflict_fields":conflict_fields},
            "raw_span":deepcopy(span)})
    skeleton = [{"bar":b["bar"],"start":b["start"],"end":b["end"],
                 "beat_grid":[{"beat":x.get("beat"),"start":x["start"],"end":x["end"]} for x in b.get("beats",[])]} for b in data.get("bars",[])]
    result = {"schema":SCHEMA,"semantics_schema":SEMANTICS_SCHEMA,
              "field_semantics_schema":FIELD_SEMANTICS_SCHEMA,"revision":0,
              "source_policy":source_policy,"source_identity":source_identity,"sample_rate":sample_rate,
              "global_key_definition":key_definition(data.get("key"),confirmed_key),"canonical_events":events,
              "legacy_consistency":legacy_consistency(data) if source_policy == "legacy_studio_bars_spans" else None,
              "quantization_artifacts":artifacts,"bar_grid":skeleton,"history":[]}
    return derive_views(result)


def validate_canonical(record):
    if record.get("schema") != SCHEMA:
        raise ValueError("Wrong canonical schema")
    previous = -1
    seen = set()
    for e in record["canonical_events"]:
        if e["event_id"] in seen or e["start_sample"] < previous or e["end_sample"] <= e["start_sample"]:
            raise ValueError("Duplicate events, overlapping or invalid canonical intervals")
        seen.add(e["event_id"]);previous=e["end_sample"]


def derive_views(record):
    validate_canonical(record)
    result = deepcopy(record)
    rev = result["revision"]
    result["events"] = [dict(e,derived_revision=rev) for e in result["canonical_events"]]
    bars = []
    for grid in result["bar_grid"]:
        members = [deepcopy(e) for e in result["canonical_events"] if e.get("bar") == grid["bar"]]
        beats = []
        for beat in grid["beat_grid"]:
            left, right = round(beat["start"]*record["sample_rate"]), round(beat["end"]*record["sample_rate"])
            segments = []
            for e in members:
                a,b = max(left,e["start_sample"]), min(right,e["end_sample"])
                if b > a:
                    segments.append({"event_id":e["event_id"],"start_sample":a,"end_sample":b,"chord":e["chord"]})
            unique = list(dict.fromkeys(e["chord"] for e in segments))
            beats.append({**beat,"segments":segments,"chord":unique[0] if len(unique)==1 else None,
                          "summary_kind":"single_label" if len(unique)==1 else "mixed" if unique else "missing", "derived_revision":rev})
        bars.append({**grid,"spans":[dict(e,derived_revision=rev) for e in members],"beats":beats,"derived_revision":rev})
    result["bars"] = bars
    result["derived_views_revision"] = rev
    return result


def edit_transaction(record, changes, *, expected_revision, reviewer):
    """Confirmations accumulate only while their semantic values stay unchanged.

    Explicit unreviewed/unresolved/pending status withdraws all confirmations.
    A changed label or key invalidates the changed fields even when the caller
    continues listened_confirmed; those values must be explicitly reconfirmed.
    """
    require_current_field_semantics(record)
    if expected_revision != record["revision"]:
        raise ValueError("Revision conflict")
    if not reviewer:
        raise ValueError("Reviewer identity required")
    # Reject stale derived copies rather than choosing whichever field is first.
    if record != derive_views(record):
        raise ValueError("Stale derived view; rebuild explicitly from canonical_events")
    result = deepcopy(record)
    by_id = {e["event_id"]:e for e in result["canonical_events"]}
    if len({c["event_id"] for c in changes}) != len(changes):
        raise ValueError("Repeated event edit in one transaction")
    for change in changes:
        if set(change)-{"event_id","label","review_status","confirmed_fields","confirmed_key"}:
            raise ValueError("Unsupported edit fields")
        e = by_id[change["event_id"]]
        label = change.get("label",e["raw_label"])
        sem = parse_chord(label)
        if sem.parse_error:
            raise ValueError(sem.parse_error)
        status = change.get("review_status","unreviewed")
        fields = change.get("confirmed_fields",[])
        if status not in REVIEW_STATUSES or not isinstance(fields,list) or set(fields)-set(FIELDS):
            raise ValueError("Invalid review status or fields")
        if fields and status != "listened_confirmed":
            raise ValueError("Only explicit listened confirmation can enable training fields")
        if "confirmed_key" in change:
            e["key_definition"] = key_definition(e["key_definition"]["display_key"], change["confirmed_key"])
        known = structural_mask(sem,e["key_definition"])
        remaining_conflicts = [f for f in e["review"].get("conflict_fields",[]) if not (status=="listened_confirmed" and f in fields)]
        for field in remaining_conflicts:known[field]=False
        if any(not known[f] for f in fields):
            raise ValueError("Cannot confirm an absent/unknown field; add explicit annotation first")
        values = field_values(sem,e["key_definition"],known)
        prior = {f for f in e["review"].get("confirmed_fields", [])
                 if e["training_mask"].get(f) and known[f] and e["field_values"][f] == values[f]}
        confirmed = [f for f in FIELDS if f in prior or f in fields] if status=="listened_confirmed" else []
        e.update(raw_label=label,chord=sem.harte,semantics=asdict(sem),known_mask=known,field_values=values)
        e["training_mask"] = {f:f in confirmed for f in FIELDS}
        e["review"] = {"status":status,"confirmed_fields":confirmed,"reviewer":reviewer,"conflict_fields":remaining_conflicts}
        origin = "explicit_listened_review" if status=="listened_confirmed" else "explicit_unresolved_or_unreviewed_edit"
        e["provenance"]["origins"] = list(dict.fromkeys(e["provenance"]["origins"] + [origin]))
    result["revision"] += 1
    result["history"].append({"revision":result["revision"],"reviewer":reviewer,"changes":deepcopy(changes)})
    return derive_views(result)


def save_reference_atomic(path, record, *, expected_disk_revision):
    """One lock, one revision, one atomic file containing all derived views."""
    require_current_field_semantics(record)
    path = Path(path)
    if not path.name.endswith(".reference_events_v1.json"):
        raise ValueError("Versioned filename required; cannot overwrite legacy exports")
    if record != derive_views(record):
        raise ValueError("Refusing stale views")
    path.parent.mkdir(parents=True,exist_ok=True)
    lock = path.with_suffix(path.suffix+".lock")
    try:
        fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise ValueError("Concurrent save in progress") from None
    pending = path.with_suffix(path.suffix+"."+uuid.uuid4().hex+".pending")
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        actual = current["revision"] if current is not None else None
        if actual != expected_disk_revision:
            raise ValueError("Disk revision conflict")
        if current is not None and reference_binding(current) != reference_binding(record):
            raise ValueError("Reference identity binding mismatch; use a new reference filename")
        if record["revision"] != (actual+1 if actual is not None else 0):
            raise ValueError("Save must advance exactly one revision")
        with pending.open("x",encoding="utf-8") as f:
            json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(pending,path)
    finally:
        os.close(fd)
        if pending.exists():pending.unlink()
        lock.unlink()


def coverage(record):
    known, trusted = dict.fromkeys(FIELDS,0.0), dict.fromkeys(FIELDS,0.0)
    origin_seconds = {}
    for e in record["canonical_events"]:
        seconds=(e["end_sample"]-e["start_sample"])/record["sample_rate"]
        for f in FIELDS:
            known[f] += seconds * e["known_mask"][f]
            trusted[f] += seconds * e["training_mask"][f]
        for origin in set(e["provenance"]["origins"]):origin_seconds[origin]=origin_seconds.get(origin,0)+seconds
    return {"labelled_seconds":sum((e["end_sample"]-e["start_sample"])/record["sample_rate"] for e in record["canonical_events"]),
            "structurally_known_seconds":known,"listened_confirmed_training_seconds":trusted,
            "origin_seconds_overlapping_categories_do_not_add":origin_seconds,
            "denominator":"one revision; exact declared analysis sample support; origin categories can overlap"}
