"""Experimental auditable key consensus. Scores are evidence, never probabilities.

Policy fixed before regression runs; no learned weights or title features.
Harmonic global/window evidence shares raw labels and is NOT two independent votes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math

from chord_studio import normalize_external_chord_label, normalize_key_label, parse_chord_name
import detect_key_timeline as harmonic

SCHEMA = "fused_harmonic_key_v1"


@dataclass(frozen=True)
class ConsensusPolicy:
    # Conservative engineering defaults, not calibrated/deployment-approved values.
    window_bars: int = 8
    step_bars: int = 2
    min_segment_bars: int = 8
    local_margin: float = .12  # Conservative engineering threshold, uncalibrated.
    min_harmonic_relative_margin: float = .05
    min_bar_share: float = 2 / 3
    min_pitched_coverage: float = .5
    min_events: int = 4
    min_roots: int = 3
    cnn_candidate_rank_limit: int = 2


POLICY = ConsensusPolicy()


def key_identity(value):
    normalized = normalize_key_label(value)
    if not normalized:
        raise ValueError(f"Invalid key: {value!r}")
    root, suffix = parse_chord_name(normalized)
    if root is None or suffix not in {"", "m"}:
        raise ValueError(f"Expected major/minor key: {value!r}")
    return harmonic.NOTE_NAMES[root] + suffix


def normalize_key_scores(scores):
    """Retain raw scores; max over metrical variants, tie-aware ranks, no imputation."""
    items = scores.items() if isinstance(scores, dict) else [(r["key"], r["score"]) if isinstance(r,dict) else r for r in scores]
    unique = {}
    for label, score in items:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("Non-finite key score")
        key = key_identity(label)
        unique[key] = max(value, unique.get(key, -math.inf))
    ordered = sorted(unique.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return dict(available=False, candidates=[], key=None, margin=None, relative_margin=None, complete_24=False)
    spread = ordered[0][1] - ordered[-1][1]
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else None
    candidates = [dict(key=key, score=score, rank=1+sum(other > score for other in unique.values()),
                       relative_score=(score-ordered[-1][1])/spread if spread else 0.) for key, score in ordered]
    return dict(available=True, key=ordered[0][0], candidates=candidates, margin=margin,
                relative_margin=margin/spread if margin is not None and spread else 0.,
                tied=margin == 0, complete_24=len(unique)==24,
                normalization="within-source range and ranks; missing candidates remain missing",
                score_kind="uncalibrated_evidence")


def timed_rows(events, duration):
    """Use genuine support lengths; discard N/X without inventing tonic evidence."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be finite and positive")
    result, previous_end = [], 0.
    for event in events:
        start, end = float(event["start"]), float(event["end"])
        if not math.isfinite(start+end) or start < previous_end or start < 0 or end <= start:
            raise ValueError("Invalid/overlapping raw chord intervals")
        previous_end = end
        label = normalize_external_chord_label(event.get("label", event.get("chord", "X")), "sharps")
        start, end = max(0., start), min(duration, end)
        if label in {"N", "X"} or end <= start:
            continue
        label = label.split("/", 1)[0]
        if result and result[-1].chord == label and abs(result[-1].end-start) < 1e-8:
            old = result.pop()
            result.append(harmonic.BarChord(0, old.start, end, label, end-old.start))
        else:
            result.append(harmonic.BarChord(0, start, end, label, end-start))
    return result


def score_timed_chords_by_key(events, duration):
    rows = timed_rows(events, duration)
    coverage = sum(row.weight for row in rows)/duration
    if not rows:
        return {**normalize_key_scores([]), "pitched_coverage":0., "event_count":0, "root_count":0, "breakdown":{}}
    result = normalize_key_scores(harmonic.all_key_scores(rows))
    details = {}
    for key in harmonic.key_candidates():
        weights, seq = Counter(), []
        for row in rows:
            degree = harmonic.functional_label(row.chord, key)
            weights[degree] += row.weight
            if not seq or seq[-1] != degree:
                seq.append(degree)
        total = sum(weights.values())
        details[key] = dict(duration_weighted_fit=sum(harmonic.chord_fit(r.chord,key)*r.weight for r in rows)/total,
            function_seconds=dict(weights), strongest_function=max(weights,key=weights.get),
            out_of_scale_fraction=sum(r.weight for r in rows if harmonic.chord_fit(r.chord,key)<0)/total,
            common_pair_hits=sum(pair in harmonic.COMMON_MAJOR_PAIRS for pair in zip(seq,seq[1:])), pair_total=max(0,len(seq)-1),
            cadence_vi_count=sum(pair==("V","I") for pair in zip(seq,seq[1:])),
            common_loop_hits=sum(group in harmonic.COMMON_MAJOR_LOOPS for group in zip(seq,seq[1:],seq[2:],seq[3:])),
            ending_functions=[harmonic.functional_label(r.chord,key) for r in rows[-4:]],
            first_function=seq[0], entry_policy="existing soft score; never a veto")
    return {**result,"pitched_coverage":coverage,"event_count":len(rows),
            "root_count":len({parse_chord_name(row.chord)[0] for row in rows}),"breakdown":details,
            "method":"detect_key_timeline.score_key with exact duration weights",
            "known_limitations":["Existing rule engine has major-mode-biased pair/loop/entry priors; minor ambiguity must remain visible"]}


def summarize_key_windows(events, bars, duration, policy=POLICY):
    empty = dict(available=False, windows=[], segments=[], bar_shares={}, modulation_detected=False, eligible_bars=0, measured_bars=0, excluded_bars=0,
                 **stability_metrics([]))
    if not bars:
        return {**empty,"reason":"No measured bar grid"}
    rows, projected = timed_rows(events,duration), []
    checked, previous_end, seen = [], 0., set()
    for bar in bars:
        start,end,number = float(bar["start"]),float(bar["end"]),int(bar["bar"])
        if not math.isfinite(start+end) or start < previous_end-1e-7 or start<0 or end<=start or number in seen:
            raise ValueError("Invalid/overlapping neutral bars")
        previous_end=end;seen.add(number)
        if start>=duration: continue
        checked.append(dict(bar=number,start=start,end=min(duration,end)))
        for row in rows:
            a,b=max(start,row.start),min(end,row.end)
            if b>a: projected.append(harmonic.BarChord(number,a,b,row.chord,b-a))
    if not projected: return {**empty,"reason":"No pitched chords in bar grid", "measured_bars":len(checked), "excluded_bars":len(checked)}
    windows=harmonic.viterbi_window_track(projected,policy.window_bars,policy.step_bars,None)
    # The legacy window.margin belongs to the local winner, not necessarily
    # the Viterbi-selected key. Only the latter's own gap may support a segment.
    corrected_windows=[]
    for window in windows:
        scores=dict(harmonic.all_key_scores(harmonic.window_rows(projected,window.start_bar,window.end_bar)))
        margin=scores[window.key]-max(value for key,value in scores.items() if key!=window.key)
        corrected_windows.append(replace(window,margin=max(0.,margin)))
    windows=corrected_windows
    # Count each bar exactly once using its nearest window center. Overlapping
    # 8-bar windows must not inflate the denominator or masquerade as coverage.
    assignments=[]
    for bar in checked:
        support=sum(r.weight for r in projected if r.bar==bar["bar"])
        if support/(bar["end"]-bar["start"]) < policy.min_pitched_coverage: continue
        matching=[w for w in windows if w.start_bar<=bar["bar"]<=w.end_bar]
        if not matching:continue
        w=min(matching,key=lambda w:(abs((w.start_bar+w.end_bar)/2-bar["bar"]),w.start_bar))
        selected_top=w.margin>=policy.local_margin
        assignments.append(dict(**bar,key=w.key,stable=selected_top))
    counts=Counter(row["key"] for row in assignments if row["stable"])
    shares={key:count/len(assignments) for key,count in counts.items()} if assignments else {}
    segments=harmonic.stable_segments(windows,policy.window_bars,policy.min_segment_bars,policy.local_margin,None)
    max_bar=max(row["bar"] for row in checked)
    timeline=[]
    for segment in segments:
        a,b=int(segment["start_bar"]),int(segment["end_bar"] or max_bar)
        covered=[bar for bar in checked if a<=bar["bar"]<=b]
        if covered:
            supported=sum(row["stable"] and row["key"]==segment["key"] for row in assignments if a<=row["bar"]<=b)
            timeline.append({**segment,"end_bar":b,"start":covered[0]["start"],"end":covered[-1]["end"],
                             "stable_bars":supported,"status":"supported" if supported>=policy.min_segment_bars else "tentative"})
    # Avoid treating one brief/noisy excursion as an established modulation.
    stable_keys={key for key,count in counts.items() if count>=policy.min_segment_bars}
    return dict(available=len(assignments)>=policy.window_bars,windows=[asdict(w) for w in windows],segments=timeline,
                bar_shares=shares,eligible_bars=len(assignments),bar_assignments=assignments,**stability_metrics(assignments),
                measured_bars=len(checked),excluded_bars=len(checked)-len(assignments),
                modulation_detected=len({s["key"] for s in timeline if s["status"]=="supported"})>1,
                local_conflict_detected=len(stable_keys)>1,source="raw labels projected on neutral measured bars",
                dependence="Same raw evidence as global harmonic score; not an independent vote",
                stability_method="one nearest window per pitched bar; require local winner margin")


def stability_metrics(assignments):
    """Separate abstention from counter-evidence; retain the old denominator.

    valid = acoustically supported bars assigned to exactly one window.
    Undefined proportions are None, never invented zero certainty.
    candidate_vs_competitor_gap uses the decisive denominator; also export
    the count gap and valid-denominator gap to make its meaning explicit.
    """
    numbers = [row["bar"] for row in assignments]
    if len(set(numbers)) != len(numbers):
        raise ValueError("Duplicate bar assignments")
    valid = len(assignments)
    counts = Counter(key_identity(row["key"]) for row in assignments if row["stable"])
    decisive = sum(counts.values())
    candidates = {}
    for key in harmonic.key_candidates():
        others = sorted(((count,k) for k,count in counts.items() if k!=key), key=lambda x:(-x[0],x[1]))
        competitor, other_key = others[0] if others else (0,None)
        own = counts[key]
        candidates[key] = dict(candidate_stable_bars=own, stable_coverage=own/valid if valid else None,
            candidate_decisive_share=own/decisive if decisive else None,
            top_competitor_key=other_key, top_competitor_stable_bars=competitor,
            top_competitor_share_valid=competitor/valid if valid else None,
            top_competitor_share_decisive=competitor/decisive if decisive else None,
            candidate_vs_competitor_gap=(own-competitor)/decisive if decisive else None,
            candidate_vs_competitor_gap_valid=(own-competitor)/valid if valid else None,
            candidate_vs_competitor_gap_bars=own-competitor)
    return dict(valid_bars=valid,decisive_bars=decisive,unresolved_bars=valid-decisive,
                decisive_coverage=decisive/valid if valid else None,
                unresolved_coverage=(valid-decisive)/valid if valid else None,
                stable_coverage={key:count/valid for key,count in counts.items()} if valid else {},
                candidate_metrics=candidates,stability_schema="decisive_bar_support_v2")


def functional_joint_evidence(neutral):
    if neutral.get("key_source") != "functional_joint":
        return {**normalize_key_scores([]),"reason":"Neutral analysis is missing or key-conditioned"}
    result=normalize_key_scores(neutral.get("key_candidates",[]))
    if len(result["candidates"])<2:
        return {**normalize_key_scores([]),"reason":"Insufficient independent joint key candidates"}
    return {**result,"selected_key":neutral.get("key"),"key_conditioned":False,
            "variants":neutral.get("key_candidates",[])}


def fuse_key_evidence(cnn, joint, harmonic_evidence, windows, *, requested_key=None, policy=POLICY):
    evidence=dict(madmom_cnn={**normalize_key_scores((cnn or {}).get("candidates",[])),"original":cnn},
                  functional_joint=joint,raw_chord_functional=harmonic_evidence,window_timeline=windows)
    if requested_key:
        return dict(key=normalize_key_label(requested_key),source="provided",status="resolved",needs_review=False,
                    evidence=evidence,candidates=[],conflict=dict(detected=False),reason=["Human key has priority"])
    a,b,c=evidence["madmom_cnn"],joint,harmonic_evidence
    tops={name:value.get("key") for name,value in [("madmom_cnn",a),("functional_joint",b),("raw_chord_functional",c)]}
    conflict=len({value for value in tops.values() if value})>1
    selected=a.get("key") or b.get("key") or c.get("key")
    checks=dict(three_sources=all(v.get("available") for v in (a,b,c)),
                complete_candidates=all(v.get("complete_24") for v in (a,b,c)),
                joint_unique=b.get("margin") is not None and b["margin"]>0,
                harmonic_sufficient=c.get("event_count",0)>=policy.min_events and c.get("root_count",0)>=policy.min_roots and c.get("pitched_coverage",0)>=policy.min_pitched_coverage,
                harmonic_unique=(c.get("relative_margin") or 0)>=policy.min_harmonic_relative_margin,
                joint_harmonic_agree=b.get("key") is not None and b.get("key")==c.get("key"),
                stable_windows=windows.get("available",False) and windows.get("bar_shares",{}).get(c.get("key"),0)>=policy.min_bar_share)
    candidate=next((r for r in a.get("candidates",[]) if r["key"]==c.get("key")),None)
    tied_count=sum(r["score"]==candidate["score"] for r in a.get("candidates",[])) if candidate else 0
    checks["cnn_support"]=bool(candidate and candidate["rank"]+tied_count-1<=policy.cnn_candidate_rank_limit and candidate["score"]>0)
    accepted=all(checks.values())
    if accepted:selected=c["key"]
    reason=[]
    if accepted:
        reason.append("Independent joint and raw harmony agree; unique harmonic winner and stable multi-bar support")
        reason.append("CNN top overridden within its top-two candidates" if selected!=a.get("key") else "CNN and harmonic evidence agree")
    else:
        reason.append("Insufficient consensus; provisional display key only")
        reason.extend("gate_failed:"+name for name,value in checks.items() if not value)
    if windows.get("modulation_detected"):reason.append("Local key changes detected; inspect local timeline")
    candidates=[dict(key=key,source_ranks={name:next((r["rank"] for r in value.get("candidates",[]) if r["key"]==key),None) for name,value in [("madmom_cnn",a),("functional_joint",b),("raw_chord_functional",c)]}) for key in harmonic.key_candidates()]
    candidates.sort(key=lambda r:(r["key"]!=selected,r["source_ranks"]["raw_chord_functional"] or 99,r["key"]))
    return dict(key=selected,source=SCHEMA,status="resolved" if accepted else "ambiguous",
                needs_review=not accepted or bool(windows.get("modulation_detected")),evidence=evidence,candidates=candidates,
                conflict=dict(detected=conflict,source_tops=tops,cnn_overridden=accepted and selected!=a.get("key")),
                gate_checks=checks,reason=reason,policy=asdict(policy),policy_status="untrained engineering gate; experimental, not confidence calibration")


def analyze_key_evidence(cnn, neutral, raw_events, *, requested_key=None):
    duration=float(neutral["duration"])
    return fuse_key_evidence(cnn,functional_joint_evidence(neutral),
                            score_timed_chords_by_key(raw_events,duration),
                            summarize_key_windows(raw_events,neutral.get("bars",[]),duration),requested_key=requested_key)


def read_raw_lab(path):
    from pathlib import Path
    events = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            start, end, label = line.split(maxsplit=2)
            events.append(dict(start=float(start), end=float(end), label=label))
    return events


def attach_key_metadata(payload, fusion, cnn):
    """Add evidence without losing legacy flat key_estimation fields."""
    payload["key"] = fusion["key"]
    payload["key_source"] = fusion["source"]
    payload["key_fusion"] = fusion
    payload["key_estimation"] = {**(cnn or {}), "madmom_cnn": cnn}
    return payload


def run_neutral_and_fuse(studio_command, base_analysis, raw_lab, cnn, run, *, use_resolver=False, resolver_model=None):
    """Never key-condition the evidence pass; at most one final rerun.

    Caller supplies an ordinary studio command without --key. Independent
    evidence is saved before the final pass can overwrite the base analysis.
    """
    import json
    from pathlib import Path
    if "--key" in studio_command:
        raise ValueError("Neutral command must not contain --key")
    run([*studio_command, "--all-key-candidates"])
    base_analysis = Path(base_analysis)
    neutral = json.loads(base_analysis.read_text(encoding="utf-8"))
    neutral_path = base_analysis.with_name(base_analysis.stem + "_neutral.json")
    neutral_path.write_text(json.dumps(neutral, ensure_ascii=False, indent=2), encoding="utf-8")
    if use_resolver:
        from key_disagreement import analyze, load_model
        fusion = analyze(cnn, neutral, read_raw_lab(raw_lab), model=load_model(resolver_model))
    else:
        fusion = analyze_key_evidence(cnn, neutral, read_raw_lab(raw_lab))
    fusion["neutral_artifact"] = str(neutral_path)
    fusion["studio_passes"] = 1
    if fusion["key"] and key_identity(fusion["key"]) != key_identity(neutral["key"]):
        run([*studio_command, "--key", fusion["key"]])
        fusion["studio_passes"] = 2
    payload = json.loads(base_analysis.read_text(encoding="utf-8"))
    attach_key_metadata(payload, fusion, cnn)
    base_analysis.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fusion
