"""A only: versioned symbolic/reference repairs and local listening queue.

Never replaces frozen models, legacy references, predictions or the default UI.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from chord_semantics_v2 import parse_chord as old_parse
from chord_semantics_v3 import parse_chord, SCHEMA
from reference_events_v1 import normalize_reference, legacy_spans, coverage, save_reference_atomic

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)


def protect_check(out):
    before=load(out/"protected_before.json")
    mismatches=[p for p,s in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=s]
    if mismatches:raise AssertionError({"protected_changed":mismatches})
    return {"protected_file_count":len(before),"mismatches":mismatches}


def run(out, make_audio, reuse_dir=None):
    start=time.perf_counter()
    if (out/"delivery.json").exists():raise FileExistsError("Completed delivery exists; use a fresh output")
    if not (out/"protected_before.json").exists():raise ValueError("Baseline protected hashes required before implementation")
    inputs=[]
    for name in ["chord_audit_review.html","chord_audit_evidence.md","CODEX_CHORD_ERROR_REPAIR_20261006.md","chord_audit_report.md","chord_audit_all_differences.csv"]:
        source=Path("C:/Users/33573/Downloads")/name
        target=out/"review_inputs"/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        inputs.append({"path":str(source),"sha256":sha(source),"bytes":source.stat().st_size})
    with (out/"review_inputs/chord_audit_all_differences.csv").open(encoding="utf-8-sig") as f:
        difference_rows=list(csv.DictReader(f))
    with (ROOT/"exports/manual_model_comparison_20261006/cases.csv").open(encoding="utf-8-sig") as f:
        cases=list(csv.DictReader(f))
    normalized, semantic_changes, consistency, cov, fingerprints = {}, [], [], [], []
    for case in cases:
        p=ROOT/case["manual_source"];data=load(p)
        rows=legacy_spans(data,"legacy_studio_bars_spans");rows.sort(key=lambda s:(s["start"],s["end"]))
        conflicts={i:["seventh"] for i,s in enumerate(rows) if s.get("chord")=="D#m7b5" and any("D# F# A C" in n for n in data.get("correction_notes",[]))}
        identity={"path":case["manual_source"],"sha256":sha(p),"audio_path":data.get("source"),"audio_offset":data.get("audio_offset",0),
                  "analysis_sample_rate_basis":"legacy studio SAMPLE_RATE=22050; not source-container sample rate"}
        record=normalize_reference(data,source_policy="legacy_studio_bars_spans",sample_rate=22050,source_identity=identity,
            origin_policy="user_rule_assisted" if case["song_id"] in {"doudou_ma","pingfan"} else "legacy_export",conflicts=conflicts)
        target=out/"references"/(case["case_id"]+".reference_events_v1.json")
        if target.exists():
            raise FileExistsError("Reference snapshot exists; use a fresh run directory")
        save_reference_atomic(target,record,expected_disk_revision=None)
        normalized[case["case_id"]]=(record,data,case)
        fingerprints.append({"case_id":case["case_id"],**identity,"canonical_path":target.relative_to(out).as_posix(),"canonical_sha256":sha(target)})
        cov.append({"case_id":case["case_id"],"song_group":case["song_id"],**coverage(record)})
        consistency.append({"case_id":case["case_id"],**record["legacy_consistency"],"event_source":"bars.spans", "new_revision":record["revision"]})
        for e in record["canonical_events"]:
            old,new=old_parse(e["raw_label"]),parse_chord(e["raw_label"])
            semantic_changes.append({"case_id":case["case_id"],"event_id":e["event_id"],"bar":e["bar"],"start":e["raw_start"],"end":e["raw_end"],
                "label":e["raw_label"],"old_state":old.state,"new_state":new.state,"old_error":old.parse_error,"new_error":new.parse_error,
                "identity_changed":old.identity!=new.identity,"new_harte":new.harte,"alias_applied":new.alias_applied})
    queue=[]
    def add(case_id,a,b,topic,question,candidates=None):
        record,data,case=normalized[case_id]
        audio=Path(data["source"])
        queue.append({"review_id":f"R{len(queue)+1:03d}","case_id":case_id,"song_group":case["song_id"],"topic":topic,
            "target_start":a,"target_end":b,"phrase_start":max(0,a-2),"phrase_end":min(data["duration"],b+2),
            "audio_path":str(audio),"audio_sha256":sha(audio),"audio_offset":data.get("audio_offset",0),"review_status":"unresolved",
            "review_question":question,"hypotheses_not_truth":candidates or [],"field_decisions_required":["distinguishing_pitch","bass","boundary","actual_tonic_mode_vs_display"],
            "no_listening_performed":True})
    record,data,case=normalized["09_gurenge"]
    ambiguous=[e for e in record["canonical_events"] if e["review"]["conflict_fields"]]
    for e in ambiguous:
        add("09_gurenge",e["raw_start"],e["raw_end"],"pitch_set_label_conflict","确认第四音 C / C# / B 或省略；D#低音是否和弦根音？",["D#dim","D#dim7","D#m7b5","B7/D#"])
    if len(ambiguous)!=7:raise AssertionError("Expected seven recorded pitch-set conflicts")
    for a,b in [(15.8,19.66),(31.17,35.0),(163.64,165.6)]:add("03_qimeidi",a,b,"C7_distinguishing_tones","核查 E/Eb、Bb 与低音；C7→Fm/Db 为待听辨假设。",["C7","Cm7","C","Fm","Db"])
    for a,b in [(34.435,36.084),(79.319,81.061),(289.414,292.873)]:add("07_laonanhai",a,b,"borrowed_and_extended_chords","核查 F/Fm 的 A/Ab、A7区别音、挂留第四音与九音；保留未决。",["F","Fm","A7","G9sus4"])
    bar61=next(b for b in normalized["07_laonanhai"][1]["bars"] if b["bar"]==61)
    add("07_laonanhai",bar61["start"],bar61["end"],"copied_phrase_independent_review","61–78为自动复制区；独立听该句与来源句是否真正相同。")
    ydata=normalized["01_yorushika"][1]
    for bars in [(97,98),(99,100)]:
        bs=[b for b in ydata["bars"] if bars[0]<=b["bar"]<=bars[1]]
        add("01_yorushika",bs[0]["start"],bs[-1]["end"],"fast_descent_and_modulation","D#dim 与 B/D# 的 A/B区别音；短和弦边界及 E→F 是否实际成立？",["D#dim","B/D#"])
    for a,b in [(5,12),(100,108),(260,268)]:add("11_pingfan",a,b,"independent_time_anchor","从录音记录实际起音/换和弦锚点；先不按参考最佳偏移移动预测。")
    add("10_doudou_ma",175.98349,177.794632,"three_four_transition","确认3/4的F#7→Bm；C/D是书写母调还是听感主音/调式？",["F#7","Bm"])
    for a,b in [(41.8,50),(140,148)]:add("05_starrail",a,b,"absolute_chord_vs_local_key","分别确认绝对和弦、实际局部主音/调式、级数显示基准；不根据级数差异判和弦错误。")
    for a,b in [(52.03,53.82),(94.69,96.48)]:add("09_gurenge",a,b,"G_Em_shared_tones","确认低音E、和弦体与落点；G母调是否只是显示坐标？",["G","Em"])
    nspans=[e for e in record["canonical_events"] if e["raw_label"]=="N"]
    for e in nspans[:2]:add("09_gurenge",e["raw_start"],e["raw_end"],"short_N_activity_vs_harmony","区分真正无稳定和声、断奏无新起音、残响与未知；不要统一删除N。",["N","sustained_harmonic_background","unknown"])
    audio_commands=[]
    cache_hits=0
    prior = {r['review_id']:r for r in load(reuse_dir/'listening_queue.json')} if reuse_dir else {}
    if make_audio:
        ffmpeg=shutil.which("ffmpeg")
        if not ffmpeg:raise FileNotFoundError("ffmpeg required for optional listening excerpts")
        for row in queue:
            dest=out/"listening_audio"/(row["review_id"]+".wav");dest.parent.mkdir(parents=True,exist_ok=True)
            old=prior.get(row['review_id'])
            cached=reuse_dir/old['local_excerpt'] if old and old.get('local_excerpt') else None
            if cached and cached.is_file() and all(old.get(k)==row[k] for k in ['audio_sha256','audio_offset','phrase_start','phrase_end']) and sha(cached)==old.get('excerpt_sha256'):
                dest.write_bytes(cached.read_bytes());cache_hits+=1
                row['excerpt_cache_source']=str(cached)
            else:
                argv=[ffmpeg,"-v","error","-nostdin","-n","-i",row["audio_path"],"-ss",str(row["phrase_start"]),"-t",str(row["phrase_end"]-row["phrase_start"]),"-ar","22050","-ac","1","-c:a","pcm_s16le",str(dest)]
                t=time.perf_counter();p=subprocess.run(argv,capture_output=True,text=True)
                audio_commands.append({"argv":argv,"returncode":p.returncode,"elapsed_seconds":time.perf_counter()-t,"stderr":p.stderr})
                if p.returncode:raise RuntimeError(p.stderr)
            row.update(local_excerpt=dest.relative_to(out).as_posix(),excerpt_sha256=sha(dest),excerpt_encoding="mono PCM16 22050 Hz; listening convenience, not model input")
    table(out/"listening_queue.csv",queue);write(out/"listening_queue.json",queue)
    table(out/"gurenge_conflicts.csv",queue[:7])
    table(out/"semantic_changes.csv",semantic_changes);write(out/"consistency_audit.json",consistency)
    write(out/"supervision_coverage.json",cov);write(out/"reference_identities.json",fingerprints)
    table(out/"supervision_coverage.csv",[{"case_id":r["case_id"],"song_group":r["song_group"],"labelled_seconds":r["labelled_seconds"],
        **{f"known_{f}_seconds":v for f,v in r["structurally_known_seconds"].items()},
        **{f"confirmed_{f}_seconds":v for f,v in r["listened_confirmed_training_seconds"].items()}} for r in cov])
    # Each row remains pending. The page deliberately hides model hypotheses until requested.
    cards=[]
    for r in queue:
        player='<audio controls preload="none" src="'+r.get("local_excerpt","")+'"></audio>' if r.get("local_excerpt") else ""
        cards.append('<article><h3>'+html.escape(r["review_id"]+' · '+r["song_group"]+' · '+r["topic"])+'</h3><p>'+str(round(r["target_start"],3))+'–'+str(round(r["target_end"],3))+'秒，状态：待复核</p>'+player+'<details><summary>听后查看区别音问题与候选</summary><p>'+html.escape(r["review_question"])+'</p><p>'+html.escape(str(r["hypotheses_not_truth"]))+'</p></details></article>')
    (out/"listening_queue.html").write_text('<!doctype html><meta charset="utf-8"><title>原声复核队列</title><style>body{font:16px/1.6 system-ui;max-width:950px;margin:25px auto;padding:15px;background:#f5f7fb}article{background:white;padding:15px;margin:15px 0;border-radius:10px}audio{width:100%}</style><h1>原声复核队列</h1><p>片段来自已存原音频，前后各保留2秒上下文。先听再展开候选；所有项目尚未裁决。用带字段的显式 review 事务另存确认，播放本页不会自动修改参考。</p>'+''.join(cards),encoding="utf-8")
    selected={"01_yorushika","03_qimeidi","05_starrail","07_laonanhai","09_gurenge","10_doudou_ma","11_pingfan"}
    summary={"new_semantics_schema":SCHEMA,"reference_versions":len(cases),"selected_work_groups":7,"all_versions_are_known_diagnostic":True,
             "changed_event_count":sum(r["identity_changed"] for r in semantic_changes),"unchanged_event_count":sum(not r["identity_changed"] for r in semantic_changes),
             "conflict_count":7,"conflict_seconds":sum(e["raw_end"]-e["raw_start"] for e in ambiguous),
             "listening_queue_phrases":len(queue),"listened_adjudications":0,"new_acoustic_inference":False,"new_training":False,
             "new_audio_decodes_for_listening_only":len(audio_commands),"listening_audio_cache_hits":cache_hits,
             "selected_group_coverage":{cid:coverage(normalized[cid][0]) for cid in sorted(selected)},
             "missing_input":"review evidence JSON not supplied; facts reconstructed from original local source files",
             "B_C_ablation":"not started: task explicitly restricts first round to A; reference conflicts remain pending",
             "model_net_correction_or_new_scores":"not evaluated; no model changed", "input_review_files":inputs,
             "commands":audio_commands,"actual_argv":sys.argv,"elapsed_seconds":time.perf_counter()-start,**protect_check(out)}
    write(out/"delivery.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"commands","input_review_files","selected_group_coverage"}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--listening-audio",action="store_true")
    p.add_argument('--reuse-listening-audio',type=Path)
    args=p.parse_args();run(args.output.resolve(),args.listening_audio,args.reuse_listening_audio.resolve() if args.reuse_listening_audio else None)
