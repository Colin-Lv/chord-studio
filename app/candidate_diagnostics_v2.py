"""Offline diagnostics only. Reference labels never enter inference features."""
from __future__ import annotations

import numpy as np

from analyze_multisource_candidate_potential import candidate_pool as legacy_pool
from candidate_targets_v2 import HEADS, ReferenceCoveragePolicy, candidate_targets
from chord_semantics_v2 import parse_chord
from evaluate_chords import TimedChord, evaluate
from multisource_candidates_v2 import SOURCE_NAMES, candidate_pool, interval_values


def checked_rows(rows):
    ordered=sorted(rows,key=lambda row:interval_values(row)[:2])
    previous=-float("inf")
    for row in ordered:
        left,right,_=interval_values(row)
        if left<0 or left<previous:
            raise ValueError("V2 diagnostics require nonnegative, disjoint intervals; clean rounding explicitly before calling")
        previous=right
    return ordered


def aligned_v2(reference,prediction):
    """Exact partitions that preserve reference metadata and reject overlap."""
    reference,prediction=checked_rows(reference),checked_rows(prediction)
    if not reference:
        return
    start,end=interval_values(reference[0])[0],interval_values(reference[-1])[1]
    boundaries=sorted({start,end,*[v for row in reference+prediction
                                   for v in interval_values(row)[:2] if start<=v<=end]})
    ri=pi=0
    for left,right in zip(boundaries,boundaries[1:]):
        middle=(left+right)/2
        while ri<len(reference) and interval_values(reference[ri])[1]<=middle:
            ri+=1
        while pi<len(prediction) and interval_values(prediction[pi])[1]<=middle:
            pi+=1
        ref=reference[ri] if ri<len(reference) and interval_values(reference[ri])[0]<=middle else None
        pred=prediction[pi] if pi<len(prediction) and interval_values(prediction[pi])[0]<=middle else None
        yield left,right,ref,pred


def field_metrics(reference, prediction, policy=None):
    policy = policy or ReferenceCoveragePolicy()
    correct, valid = np.zeros(len(HEADS)), np.zeros(len(HEADS))
    span = 0.
    for left, right, ref, pred in aligned_v2(reference, prediction):
        span += right-left
        target = candidate_targets(TimedChord(left, right, None), [interval_values(pred)[2] if pred is not None else None],
                                   [ref] if ref else [], policy)
        valid += target.valid_seconds[0]
        correct += np.nan_to_num(target.values[0]) * target.valid_seconds[0]
    return {head: dict(correct_seconds=float(correct[i]), valid_seconds=float(valid[i]),
                       coverage=valid[i]/span if span else 0.,
                       accuracy=correct[i]/valid[i] if valid[i] else None)
            for i, head in enumerate(HEADS)}


def change_diagnostics(reference, raw, predicted, policy=None):
    """Duration of corrections/harms, independently for each annotated field."""
    policy = policy or ReferenceCoveragePolicy()
    checked_rows(raw); checked_rows(predicted); checked_rows(reference)
    if [interval_values(r)[:2] for r in raw] != [interval_values(r)[:2] for r in predicted]:
        raise ValueError("Change diagnosis requires unchanged raw intervals")
    hurt, helped = np.zeros(len(HEADS)), np.zeros(len(HEADS))
    changed = 0.
    for baseline, output in zip(raw, predicted):
        bstart,bend,blabel=interval_values(baseline)
        olabel=interval_values(output)[2]
        if parse_chord(blabel).identity != parse_chord(olabel).identity:
            changed += bend-bstart
        # Aggregate fractional utility differences within an interval. Report
        # gross errors below at exact reference intersections, not just net gain.
        for ref in reference:
            rstart,rend,_=interval_values(ref)
            left,right=max(bstart,rstart),min(bend,rend)
            if right <= left:
                continue
            part=candidate_targets(TimedChord(left,right,None),[blabel,olabel],[ref],policy)
            delta=np.nan_to_num(part.values[1]-part.values[0])*part.valid_seconds[0]
            hurt+=np.maximum(-delta,0)
            helped+=np.maximum(delta,0)
    return dict(changed_seconds=changed, fields={head:dict(
        harmful_seconds=float(hurt[i]),helpful_seconds=float(helped[i])) for i,head in enumerate(HEADS)})


def candidate_diagnostics(raw, sources, reference, groups=None, policy=None):
    checked_rows(raw); checked_rows(reference)
    policy=policy or ReferenceCoveragePolicy()
    count={"v1":0,"v2":0}
    retained={"v1":0,"v2":0}
    eligible=0
    oracle={key:np.zeros(len(HEADS)) for key in count}
    validity=np.zeros(len(HEADS))
    total=0.
    for index,event in enumerate(raw):
        pool=(groups[index]["pool"] if groups is not None else candidate_pool(event,sources))
        old=legacy_pool(event,[sources.get("chordino",[]),sources.get("largevocab",[])])
        labels={"v1":[c["label"] for c in old],"v2":[c.label for c in pool.candidates]}
        extension_identities={c.semantics.identity for c in pool.candidates if c.semantics.extension_degrees}
        eligible+=len(extension_identities)
        for variant,values in labels.items():
            count[variant]+=len(values)
            identities={parse_chord(label).identity for label in values}
            retained[variant]+=len(extension_identities&identities)
            targets=candidate_targets(event,values,reference,policy)
            if len(values):
                best=np.max(np.nan_to_num(targets.values,nan=0.),axis=0)
                oracle[variant]+=best*targets.valid_seconds[0]
            if variant=="v2":
                validity+=targets.valid_seconds[0]
        total+=event.end-event.start
    full_coverage=field_metrics(reference,[],policy)
    return dict(intervals=len(raw),candidates=count,
        mean_candidates={name:count[name]/len(raw) if raw else 0 for name in count},
        extended_source_candidate_identities=eligible,
        extensions_retained={name:dict(count=retained[name],fraction=retained[name]/eligible if eligible else None)
                             for name in count},
        oracle_warning="Unrealizable independent field maxima; reference-only offline analysis, never inference or joint achievable performance",
        oracle={variant:{head:dict(correct_seconds=float(oracle[variant][i]),
            valid_reference_seconds=full_coverage[head]["valid_seconds"],
            candidate_support_valid_seconds=float(validity[i]),
            accuracy=float(oracle[variant][i]/full_coverage[head]["valid_seconds"])
                if full_coverage[head]["valid_seconds"] else None) for i,head in enumerate(HEADS)} for variant in count},
        original_boundary_support_seconds=total)


def summarize_song_metrics(songs):
    """Both song macro and exact-duration micro; no reference filtering by method."""
    if not songs:
        return {}
    methods=songs[0]["metrics"]
    result={}
    for method in methods:
        rows=[song["metrics"][method] for song in songs]
        metrics=[key for key in rows[0] if key.endswith(("_accuracy","_wcsr")) or key=="prediction_coverage"]
        macro={key:float(np.mean([row[key] for row in rows])) for key in metrics}
        micro={}
        for key in metrics:
            denominator=("no_chord_seconds" if key=="no_chord_accuracy" else
                         "non_no_chord_seconds" if key.endswith("_accuracy") else "evaluated_seconds")
            seconds=sum(row[denominator] for row in rows)
            micro[key]=sum(row[key]*row[denominator] for row in rows)/seconds if seconds else None
        result[method]=dict(song_macro=macro,duration_micro=micro)
    return result
