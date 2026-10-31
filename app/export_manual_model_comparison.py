"""Read-only export of persisted user corrections and historical model evidence.

No recognition, feature extraction, training, threshold selection or file repair.
The archive is a forensic comparison, not a new accuracy benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
USER = ROOT / "benchmarks/predictions/user_tests"
REPORTS = ROOT / "benchmarks/reports"
sys.path.insert(0, str(ROOT / "app"))
from chord_semantics_v2 import parse_chord


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def spans(data):
    # bars.spans is the display/edit source of truth. Legacy events can be stale.
    result = []
    for bar in data.get("bars", []):
        for item in bar.get("spans", []):
            result.append(dict(item, bar=bar.get("bar")))
    if not result:
        result = [dict(x) for x in data.get("events", [])]
    return sorted(result, key=lambda x: (float(x["start"]), float(x["end"])))


def lab_spans(path):
    result = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        a, b, label = line.split(maxsplit=2)
        result.append({"start": float(a), "end": float(b), "start_text": a, "end_text": b, "chord": label})
    return result


def chord(item):
    return item.get("chord_display") or item.get("chord") or item.get("label") if item else None


def roman(item):
    return item.get("roman_display") or item.get("roman") if item else None


def semantic(label):
    s = parse_chord(label)
    # Invalid historical spellings must not all collapse into the identity X.
    if s.parse_error:
        return ("unparsed_literal", label), "unparsed_literal", s.parse_error
    return s.identity, s.state, None


def key_at(item, data):
    if item and item.get("local_key"):
        return item["local_key"]
    bar = item.get("bar") if item else None
    if bar is not None:
        for section in data.get("modulations", []):
            if section.get("start_bar", 0) <= bar and (section.get("end_bar") is None or bar <= section["end_bar"]):
                return section.get("key")
    value = data.get("key")
    # A whole-song transition descriptor is not the key of every interval.
    return value if isinstance(value, str) and "->" not in value else None


def key_identity(value):
    if value is None:
        return None
    match = re.fullmatch(r"([A-G])([#b]*)(?:\s*(major|minor|maj|min|m))?", value.strip())
    if not match:
        return None
    note, accidental, mode = match.groups()
    pc = ({"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[note] + accidental.count("#") - accidental.count("b")) % 12
    return pc, "minor" if mode in {"minor", "min", "m"} else "major"


def cover(items, t):
    return [i for i, x in enumerate(items) if float(x["start"]) <= t < float(x["end"])]


def compare(manual, model, case_id, stage_id):
    left, right = spans(manual), spans(model)
    points = sorted({float(s[k]) for s in left + right for k in ("start", "end")})
    rows = []
    for a, b in zip(points, points[1:]):
        if b <= a:
            continue
        li, ri = cover(left, (a + b) / 2), cover(right, (a + b) / 2)
        if not li and not ri:
            continue
        l = left[li[0]] if len(li) == 1 else None
        r = right[ri[0]] if len(ri) == 1 else None
        cl, cr = chord(l), chord(r)
        sl, ls, le = semantic(cl)
        sr, rs, re = semantic(cr)
        valid = len(li) == 1 and len(ri) == 1
        copy_range = (manual.get("auto_completion") or {}).get("target_bars")
        in_copy = bool(l and copy_range and copy_range[0] <= int(l.get("bar", 0)) <= copy_range[1])
        edited = bool(l and l.get("edited"))
        kind = "overlap_ambiguous" if len(li) > 1 or len(ri) > 1 else "manual_missing" if not li else "model_missing" if not ri else "same" if sl == sr else "different"
        rows.append({
            "case_id": case_id, "stage_id": stage_id, "start_seconds": a, "end_seconds": b,
            "duration_seconds": b - a, "manual_bar": l.get("bar") if l else None,
            "manual_beat_start": l.get("beat_start") if l else None,
            "manual_beat_end": l.get("beat_end") if l else None,
            "model_bar": r.get("bar") if r else None,
            "manual_span_index": li[0] if len(li) == 1 else None,
            "model_span_index": ri[0] if len(ri) == 1 else None,
            "manual_chord": cl, "model_chord": cr,
            "manual_roman": roman(l), "model_roman": roman(r),
            "manual_key": key_at(l, manual), "model_key": key_at(r, model),
            "manual_bass_root": l.get("bass_root") if l else None,
            "model_bass_root": r.get("bass_root") if r else None,
            "manual_edited_flag": edited, "in_auto_copy_target_range": in_copy,
            "manual_provenance": "auto_copy_region_may_have_later_human_edits" if in_copy else "edited_flag" if edited else "not_explicitly_edited",
            "comparison": kind, "literal_label_changed": cl != cr if valid else None,
            "full_semantic_label_changed": sl != sr if valid else None,
            "roman_changed": roman(l) != roman(r) if valid else None,
            "key_changed": key_identity(key_at(l, manual)) != key_identity(key_at(r, model))
                if valid and key_identity(key_at(l, manual)) is not None and key_identity(key_at(r, model)) is not None else None,
            "key_literal_changed": key_at(l, manual) != key_at(r, model) if valid else None,
            "manual_parse_status": ls, "model_parse_status": rs,
            "manual_parse_error": le, "model_parse_error": re,
            "model_confidence_stored": r.get("confidence") if r else None,
            "model_bass_confidence_stored": r.get("bass_confidence") if r else None,
            "manual_cover_count": len(li), "model_cover_count": len(ri),
            "model_original_start": r.get("start") if r else None,
            "model_original_end": r.get("end") if r else None,
            "manual_original_start": l.get("start") if l else None,
            "manual_original_end": l.get("end") if l else None,
        })
    comparable = sum(x["duration_seconds"] for x in rows if x["comparison"] in {"same", "different"})
    same = sum(x["duration_seconds"] for x in rows if x["comparison"] == "same")
    return rows, {
        "aligned_rows": len(rows), "comparable_seconds": comparable,
        "same_label_seconds": same, "different_label_seconds": comparable - same,
        "label_agreement_fraction_not_accuracy": same / comparable if comparable else None,
        "edited_flag_different_seconds": sum(x["duration_seconds"] for x in rows if x["manual_edited_flag"] and x["comparison"] == "different"),
        "roman_changed_seconds": sum(x["duration_seconds"] for x in rows if x["roman_changed"]),
        "key_changed_seconds": sum(x["duration_seconds"] for x in rows if x["key_changed"]),
        "model_missing_seconds": sum(x["duration_seconds"] for x in rows if x["comparison"] == "model_missing"),
        "manual_missing_seconds": sum(x["duration_seconds"] for x in rows if x["comparison"] == "manual_missing"),
        "ambiguous_overlap_seconds": sum(x["duration_seconds"] for x in rows if x["comparison"] == "overlap_ambiguous"),
    }


def alignment_self_checks():
    m = {"key":"C", "events":[{"start":0,"end":1,"chord":"C/E"},{"start":1,"end":3,"chord":"C9"}]}
    b = {"key":"G", "events":[{"start":0,"end":2,"chord":"C/G"},{"start":2,"end":3,"chord":"C9"}]}
    rr, ss = compare(m, b, "probe", "stage")
    assert len(rr) == 3 and ss["different_label_seconds"] == 2 and ss["same_label_seconds"] == 1
    assert ss["key_changed_seconds"] == 3
    assert semantic("H?")[0] != semantic("Z?")[0]
    assert semantic("C:maj/3")[0] == semantic("C/E")[0] != semantic("C/G")[0]
    assert semantic("C9")[0] != semantic("C7")[0]
    _, ss = compare({"events":[{"start":0,"end":2,"chord":"C"}]}, {"events":[{"start":1,"end":2,"chord":"N"}]}, "probe", "stage")
    assert ss["model_missing_seconds"] == 1 and ss["different_label_seconds"] == 1 and ss["key_changed_seconds"] == 0
    _, ss = compare({"events":[{"start":0,"end":2,"chord":"C"},{"start":1,"end":2,"chord":"G"}]}, {"events":[{"start":0,"end":2,"chord":"C"}]}, "probe", "stage")
    assert ss["ambiguous_overlap_seconds"] == 1 and ss["comparable_seconds"] == 1
    assert key_identity("Ab") == key_identity("G# major") and key_identity("A") != key_identity("Am")
    assert key_at({"bar":3}, {"key":"B major -> Ab major", "modulations":[{"start_bar":1,"end_bar":4,"key":"B"}]}) == "B"
    assert key_at({}, {"key":"B major -> Ab major"}) is None
    return ["boundary_union_and_duration", "literal_unknowns", "slash_and_harte", "extensions", "missing_mask", "overlap_exclusion", "key_equivalence_and_local_key_mask"]


class Export:
    def __init__(self, output):
        self.out = output
        self.records = {}
        self.stage_rows = []
        self.omitted = []

    def copy(self, path, role, song=None):
        path = Path(path).resolve()
        if not path.is_file():
            self.omitted.append({"path": str(path), "reason": "missing_input", "role": role, "song_id": song})
            return None
        if not path.is_relative_to(ROOT):
            self.omitted.append({"path": str(path), "reason": "external_dependency_not_packaged", "role": role, "song_id": song})
            return None
        name = rel(path)
        if name in self.records:
            rec = self.records[name]
            if role not in rec["roles"]:
                rec["roles"].append(role)
            if song and song not in rec["song_ids"]:
                rec["song_ids"].append(song)
            return rec["package_path"]
        data = path.read_bytes()
        target = self.out / "originals" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        package = target.relative_to(self.out).as_posix()
        self.records[name] = {"source_path": name, "package_path": package, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest(), "roles": [role], "song_ids": [song] if song else []}
        return package

    def stage(self, path, data, song, role):
        package = self.copy(path, role, song)
        stage_id = hashlib.sha256(rel(path).encode()).hexdigest()[:12]
        target = self.out / "songs" / song / "stages" / stage_id
        events = spans(data)
        metadata = {k: v for k, v in data.items() if k not in {"bars", "events", "waveform"}}
        dump(target / "decision_metadata.json", {"source": package, "role": role, "data": metadata})
        csv_write(target / "interval_decisions.csv", [{"span_index": i, **s} for i, s in enumerate(events)])
        self.stage_rows.append({"song_id": song, "stage_id": stage_id, "role": role, "source_path": rel(path),
                                "package_path": package, "decisions": target.relative_to(self.out).as_posix(),
                                "key": data.get("key"), "meter": data.get("meter"), "grid": data.get("beat_grid"),
                                "analysis_model": data.get("analysis_model"), "span_count": len(events),
                                "stored_key_candidates": len(data.get("key_candidates", [])),
                                "model_candidates_stored_in_intervals": any(s.get("candidates") for s in events),
                                "confidence_semantics": "stored value, not a calibrated correctness probability"})
        return stage_id


def resolve_audio(data):
    source = data.get("source")
    if isinstance(source, str) and Path(source).is_file():
        return Path(source).resolve()
    return None


def same_audio(a, b):
    pa, pb = resolve_audio(a), resolve_audio(b)
    # Same immutable current file; no claim that an un-hashed legacy run used these bytes.
    return bool(pa and pb and pa == pb and a.get("audio_offset", 0) == b.get("audio_offset", 0))


def classify(path, data):
    name = path.name.lower()
    if data.get("corrected"):
        return "derived_or_imported_correction_not_new_annotation"
    if data.get("progression_override") or data.get("bar_overrides") or "user_sections" in name or "bar97_chain" in name:
        return "assisted_output_contains_user_constraints"
    if "candidate" in name or "selector" in name or "specialist" in name or "probe" in name or "gate" in name or "pipeline" in name:
        return "historical_experiment_not_default"
    return "historical_model_output_configuration_not_fully_verified"


def inventory_models():
    paths = set(USER.rglob("*analysis.json"))
    paths.update((ROOT / ".tmp/user_regression").glob("*analysis.json"))
    folder = REPORTS / "session_20260926"
    for p in folder.rglob("*.json"):
        if "analysis" in p.name or "acoustic" in p.name or "functional" in p.name:
            paths.add(p)
    result = []
    for p in sorted(paths):
        try:
            d = load(p)
        except (ValueError, OSError):
            continue
        if isinstance(d, dict) and d.get("bars") and isinstance(d.get("source"), str):
            result.append((p, d))
    return result


def references_from_json(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in {"path", "lab", "prediction", "analysis", "neutral", "cnn"} and isinstance(v, str):
                p = Path(v)
                if p.is_absolute():
                    yield p
                elif (ROOT / p).is_file():
                    yield ROOT / p
            if isinstance(v, (dict, list)):
                yield from references_from_json(v)
    elif isinstance(data, list):
        for v in data:
            yield from references_from_json(v)


def package_linked(exp, path, song, depth=0, seen=None):
    seen = seen if seen is not None else set()
    path = Path(path).resolve()
    if path in seen or depth > 4:
        return
    seen.add(path)
    if path.suffix.lower() not in {".json", ".lab", ".csv", ".beats", ".txt"}:
        return
    exp.copy(path, "linked_provenance_or_contract", song)
    if path.suffix == ".json" and path.is_file():
        # Keep contract/batch provenance, without pulling unrelated training or
        # corpus-wide songs into a user-correction archive through its inputs.
        if "contract" in path.name or (depth > 0 and path.name == "report.json"):
            return
        try:
            d = load(path)
        except ValueError:
            return
        for p in references_from_json(d):
            package_linked(exp, p, song, depth + 1, seen)


def make_html(out, cases, samples):
    payload = json.dumps({"cases": cases, "rows": samples}, ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>人工修订与模型判断对照</title>
<style>body{font:15px/1.6 system-ui;margin:28px;color:#203047;background:#f6f8fc}h1{font-size:25px}select,input{padding:8px;margin:5px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}th{position:sticky;top:0;background:#e9eff7}.diff{background:#fff2dc}.card{background:white;padding:18px;border-radius:10px;margin:16px 0}a{color:#155caa}small{color:#52647b}</style>
<h1>人工修订与模型判断对照</h1><p>逐秒时间对齐；展示保存的实际结果。差异比例不是独立金标准准确率。原始 JSON、所有历史阶段与完整 CSV 均保留。</p><p><a href="songs/xuni/key_decision_comparison.csv">《虚拟》各分支主调与前三候选对照</a> · <a href="candidate_table_index.json">候选判断数据索引</a> · <a href="README.md">完整说明</a></p>
<label>歌曲/修订版本 <select id="cases"></select></label><label><input id="changed" type="checkbox" checked>只看差异</label><div id="meta" class="card"></div><table><thead><tr><th>时间 / 小节</th><th>原模型和弦 / 级数</th><th>人工结果 / 级数</th><th>主调 原 → 修订</th><th>修订来源</th><th>原模型分数</th></tr></thead><tbody id="rows"></tbody></table>
<script>const DATA=__PAYLOAD__;const esc=x=>String(x??'缺失').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const sel=document.getElementById('cases');DATA.cases.forEach(c=>{const o=document.createElement('option');o.value=c.case_id;o.textContent=c.title+' · '+c.case_id;sel.append(o)});function draw(){const c=DATA.cases.find(c=>c.case_id===sel.value);document.getElementById('meta').innerHTML='<b>'+esc(c.title)+'</b><br>基线：'+esc(c.baseline_source)+'<br>'+esc(c.notes)+'<br><a href="'+esc(c.directory)+'/interval_comparison.csv">完整逐区间 CSV</a> · <a href="'+esc(c.directory)+'/summary.json">统计与来源 JSON</a> · <a href="stages.csv">全部模型阶段索引</a><br><small>置信数值仅为原文件保存值，人工修改后的 1 不表示模型概率。自动复制区域不等于独立人工听辨。</small>';document.getElementById('rows').innerHTML=DATA.rows.filter(r=>r.case_id===c.case_id&&(!document.getElementById('changed').checked||r.comparison!=='same'||r.roman_changed||r.key_changed)).map(r=>'<tr class="'+(r.comparison==='different'?'diff':'')+'"><td>'+r.start_seconds.toFixed(3)+'–'+r.end_seconds.toFixed(3)+'<br>第 '+esc(r.manual_bar)+' 小节</td><td>'+esc(r.model_chord)+' / '+esc(r.model_roman)+'</td><td>'+esc(r.manual_chord)+' / '+esc(r.manual_roman)+'</td><td>'+esc(r.model_key)+' → '+esc(r.manual_key)+'</td><td>'+esc(r.manual_provenance)+'<br>'+esc(r.comparison)+'</td><td>'+esc(r.model_confidence_stored)+'</td></tr>').join('')}sel.onchange=draw;document.getElementById('changed').onchange=draw;draw();</script></html>'''
    (out / "index.html").write_text(page.replace("__PAYLOAD__", payload), encoding="utf-8")


def build(output):
    start = time.perf_counter()
    probes = alignment_self_checks()
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Refusing to replace an existing delivery")
    output.mkdir(parents=True)
    exp = Export(output)
    exp.copy(Path(__file__), "export_tool")
    exp.copy(ROOT / "app/chord_semantics_v2.py", "comparison_semantics")
    for p in [ROOT / "benchmarks/user_regression_cases.json", ROOT / "benchmarks/manifest.user_regression.csv",
              ROOT / "benchmarks/manifest.user4_corrected.csv", ROOT / "PROJECT_REPORT.md"]:
        exp.copy(p, "historical_context_not_new_evaluation")
    originals = sorted(list(ROOT.glob("*corrected_analysis*.json")) + list((ROOT / "archive/root_exports_20260811").glob("*corrected_analysis*.json")))
    prefixes = {"BV1UTKBz9Eog": ("gurenge", "紅蓮華", "gurenge_madmom_current_analysis.json"),
                "BV15x4y1i7m9": ("starrail", "使一颗心免于哀伤", "starrail_heart_sorrow_current_analysis.json"),
                "BV1Vb57ztE37": ("laonanhai", "老男孩", "laonanhai_current_analysis.json"),
                "BV1Qb411g7bY": ("yorushika", "所以我放弃了音乐", "yorushika_dakara_boku_key_timeline_analysis.json"),
                "BV1Sx4y157RQ": ("qimeidi", "凄美地", "qimeidi_current_flats_analysis.json")}
    targets = []
    for p in originals:
        song, title, base = next(v for k, v in prefixes.items() if k in p.name)
        targets.append({"path": p, "song": song, "title": title, "base": USER / base, "kind": "original_manual_export"})
    targets.extend([
        {"path": USER / "gurenge_corrected_analysis.json", "song": "gurenge", "title": "紅蓮華（后续半减七澄清）", "base": USER / "gurenge_madmom_current_analysis.json", "kind": "imported_revision_after_user_pitch_set_clarification"},
        {"path": ROOT / ".tmp/user_regression/doudou_ma_post2_analysis.json", "song": "doudou_ma", "title": "马（人工规则辅助）", "base": ROOT / ".tmp/user_regression/doudou_ma_base_analysis.json", "kind": "scripted_user_verified_constraints_not_free_annotation"},
        {"path": USER / "pingfan_regularized0_6415_with_I49_64_analysis.json", "song": "pingfan", "title": "平凡之路（人工规则辅助）", "base": REPORTS / "session_20260926/madmom_user4_final/pingfan_analysis.json", "kind": "scripted_user_verified_constraints_not_free_annotation"},
    ])
    cases, main_rows, comparisons, revision_rows, audio_identities = [], [], [], [], []
    models = inventory_models()
    latest = {}
    song_models = {}
    for song in sorted({x["song"] for x in targets}):
        choices = [t for t in targets if t["song"] == song]
        choices.sort(key=lambda t: (load(t["path"]).get("corrected_at", ""), t["kind"] != "original_manual_export"))
        latest[song] = choices[-1]
        matching = [(p, d) for p, d in models if same_audio(load(choices[-1]["path"]), d)]
        song_models[song] = matching
        for p, d in matching:
            exp.stage(p, d, song, classify(p, d))
        for p in REPORTS.glob(song + "*"):
            if p.is_file() and p.suffix in {".json", ".csv"}:
                exp.copy(p, "historical_diagnostic_may_contain_reference_targets_not_inference_only", song)
        if song == "doudou_ma":
            for p in (ROOT / ".tmp/user_regression").glob("doudou_ma*"):
                if p.suffix in {".json", ".csv", ".beats", ".txt"}:
                    exp.copy(p, "user_constrained_pipeline_evidence", song)
        data = load(choices[-1]["path"])
        audio = resolve_audio(data)
        audio_identities.append({"song_id": song, "source_path": rel(audio) if audio else data.get("source"),
                                 "current_file_sha256": digest(audio) if audio else None,
                                 "current_file_bytes": audio.stat().st_size if audio else None,
                                 "packaged": False, "reason": "audio excluded from results archive",
                                 "legacy_generation_sha_verified": False, "audio_offset": data.get("audio_offset", 0),
                                 "decoded_duration_stored": data.get("duration")})
    # All exports are distinct revisions even when they have the same title.
    for i, t in enumerate(targets, 1):
        p, bp = t["path"], t["base"]
        manual, base = load(p), load(bp)
        if not same_audio(manual, base):
            raise ValueError(f"Audio/time-offset pairing mismatch: {p} / {bp}")
        case_id = f"{i:02d}_{t['song']}"
        target = output / "cases" / case_id
        baseline_stage = hashlib.sha256(rel(bp).encode()).hexdigest()[:12]
        rows, summary = compare(manual, base, case_id, baseline_stage)
        source = exp.copy(p, t["kind"], t["song"])
        baseline = exp.copy(bp, "primary_comparison_baseline", t["song"])
        ev, top_ev = spans(manual), manual.get("events", [])
        matching_grid = [(b["start"], b["end"]) for b in manual["bars"]] == [(b["start"], b["end"]) for b in base["bars"]]
        notes = "同源音频与偏移匹配；旧导出没有原运行输出 SHA，不能证明为同次运行。" + ("小节网格相同。" if matching_grid else "网格不同，采用实际时间并集对齐。")
        if t["song"] == "starrail":
            notes += "额外保留 user_sections：在人工指定转调之后、逐和弦编辑之前的辅助结果。"
        if t["song"] == "yorushika":
            notes += "基线已包含历史局部调性处理，并非纯 CNN；两个用户导出不是经哈希证明的连续修订链。"
        if manual.get("auto_completion"):
            notes += "自动复制区段与后续手改混合，无法逐项追溯复制之后的操作。"
        if t["song"] == "pingfan":
            notes += "未找到同次修订前的自动 JSON；这里是同一剪辑的后期冻结工作台对照，不能把它冒充原页面版本。"
        case = {"case_id": case_id, "song_id": t["song"], "title": t["title"], "kind": t["kind"],
                "manual_source": rel(p), "manual_package_path": source, "baseline_source": rel(bp),
                "baseline_package_path": baseline, "directory": f"cases/{case_id}", "notes": notes,
                "same_current_audio_path": True, "same_bar_grid": matching_grid, "same_original_run_proven": False,
                "manual_key": manual.get("key"), "model_key": base.get("key"), "corrected_at": manual.get("corrected_at"),
                "export_corrected_event_count": manual.get("corrected_event_count"),
                "bars_span_edited_flag_count": sum(bool(x.get("edited")) for x in ev),
                "top_level_events_edited_flag_count": sum(bool(x.get("edited")) for x in top_ev),
                "bars_span_count": len(ev), "top_level_events_count": len(top_ev),
                "top_level_events_equal_bars_spans_ignoring_bar": [{k:v for k,v in e.items() if k != "bar"} for e in top_ev] == [{k:v for k,v in e.items() if k != "bar"} for e in ev],
                "auto_completion": manual.get("auto_completion"), "progression_override": manual.get("progression_override"),
                "bar_overrides": manual.get("bar_overrides"), **summary}
        dump(target / "summary.json", case)
        csv_write(target / "interval_comparison.csv", rows)
        csv_write(target / "changes_only.csv", [r for r in rows if r["comparison"] != "same" or r["roman_changed"] or r["key_changed"]])
        csv_write(target / "manual_spans.csv", ev)
        # Kept separately: an export flag is not an independently heard whole-song label.
        cases.append(case)
        main_rows.extend(rows)
        parent = manual.get("continued_from_export")
        if parent and Path(parent).is_file():
            previous = load(parent)
            if same_audio(manual, previous):
                rr, ss = compare(manual, previous, case_id, "previous_user_export")
                csv_write(target / "previous_revision_comparison.csv", rr)
                revision_rows.append({"case_id": case_id, "previous_revision": rel(parent), **ss})
    # Compare latest retained correction to every same-input historical model stage.
    for song, t in latest.items():
        manual = load(t["path"])
        final_case = next(c for c in reversed(cases) if c["manual_source"] == rel(t["path"]))
        all_rows = []
        for p, data in song_models[song]:
            if data.get("corrected") or p == t["path"]:
                continue
            sid = hashlib.sha256(rel(p).encode()).hexdigest()[:12]
            rr, ss = compare(manual, data, final_case["case_id"], sid)
            all_rows.extend(rr)
            comparisons.append({"song_id": song, "manual_case_id": final_case["case_id"], "model_stage_id": sid,
                                "model_source": rel(p), "role": classify(p, data), **ss})
        csv_write(output / "songs" / song / "all_model_interval_comparisons.csv", all_rows)
    # Current frozen chord model: original independent sources, selected head, guard.
    guarded = REPORTS / "session_20260926_accuracy2/candidate_frozen_user4_root_guarded_eval"
    for p in guarded.glob("*.json"):
        package_linked(exp, p, None)
    for p in guarded.glob("predictions_root_guarded/*.lab.provenance.json"):
        sid = p.name.split(".")[0]
        package_linked(exp, p, sid)
        prov = load(p)
        for name in ["lab", "full_head_prediction"]:
            if prov.get(name, {}).get("path"):
                path = Path(prov[name]["path"])
                if path.is_file():
                    exp.copy(path, f"frozen_v1_{name}", sid)
        # Same-input source claims are checked against the actual current audio SHA.
        target = latest.get(sid)
        if not target:
            continue
        manual = load(target["path"])
        actual_audio = resolve_audio(manual)
        identity_ok = bool(actual_audio and digest(actual_audio) == prov.get("audio", {}).get("sha256"))
        branches = [(name, prov.get(name, {})) for name in ["lab", "full_head_prediction"]]
        branches.extend((f"independent_source_{i}", s.get("lab", {})) for i, s in enumerate(prov.get("sources", [])))
        for name, info in branches:
            path = Path(info.get("path", "missing"))
            if identity_ok and path.is_file():
                exp.copy(path, f"frozen_v1_{name}", sid)
                decisions = {"events": lab_spans(path), "generation_provenance": rel(p), "margin": prov.get("margin"), "preserve_root": prov.get("preserve_root")}
                exp.stage(path, decisions, sid, f"frozen_v1_{name}")
                rr, ss = compare(manual, decisions, f"latest_{sid}", f"frozen_v1_{name}")
                csv_write(output / "songs" / sid / f"frozen_v1_{name}_comparison.csv", rr)
                comparisons.append({"song_id": sid, "model_source": rel(path), "role": f"frozen_v1_{name}", "audio_sha_verified": True,
                                    "source_generation_identity": prov.get("sources", [])[int(name.rsplit('_',1)[1])].get("status") if name.startswith("independent_source_") else "frozen_model_provenance", **ss})
        raw_path = Path(prov.get("sources", [{}])[0].get("lab", {}).get("path", "missing"))
        head_path = Path(prov.get("full_head_prediction", {}).get("path", "missing"))
        final_path = Path(prov.get("lab", {}).get("path", "missing"))
        if identity_ok and all(q.is_file() for q in [raw_path, head_path, final_path]):
            raw, head, final = [lab_spans(q) for q in [raw_path, head_path, final_path]]
            guard_rows = []
            for f in final:
                t = (f["start"] + f["end"]) / 2
                ri, hi = cover(raw, t), cover(head, t)
                r, h = raw[ri[0]] if len(ri) == 1 else {}, head[hi[0]] if len(hi) == 1 else {}
                guard_rows.append({"start":f["start"], "end":f["end"], "raw_label":chord(r), "selected_head_label":chord(h), "guarded_label":chord(f),
                    "stored_guard_action": "kept_selected_head" if chord(f) == chord(h) else "returned_raw" if chord(f) == chord(r) else "unexplained_difference",
                    "margin_configuration":prov.get("margin"), "head_candidate_scores":None, "raw_candidate_score":None,
                    "score_missing_reason":"not persisted in this historical top1 prediction/provenance", "provenance":rel(p)})
            csv_write(output / "songs" / sid / "frozen_v1_guard_decisions.csv", guard_rows)
    # Virtual song: user-confirmed global C; no fictitious persisted UI edit.
    xuni = REPORTS / "harmonic_key_v2_20261006/regressions/xuni/report.json"
    xr = load(xuni)["rows"][0]
    package_linked(exp, xuni, "xuni")
    for p in (REPORTS / "harmonic_key_v2_20261006/regressions/xuni").rglob("*.json"):
        exp.copy(p, "known_regression_key_evidence", "xuni")
    run = ROOT / "projects/web/087d8b0c86414fe6b675cc622f505128/runs/971c479a56844032ae1e231d08f90073"
    for p in run.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".json", ".lab", ".log", ".beats"} and "nnls_cache" not in p.parts:
            exp.copy(p, "actual_web_run_evidence", "xuni")
    exp.copy(run.parents[1] / "project.json", "actual_web_project", "xuni")
    for folder in [REPORTS / "key_fusion_20261006/xuni", REPORTS / "key_resolver_20261006/xuni_evidence"]:
        for p in folder.glob("*.json"):
            exp.copy(p, "known_regression_key_branch_evidence", "xuni")
    ranking_dir = REPORTS / "key_top3_rerank_20261006_v1"
    for name in ["xuni_regression.json", "xuni_M1_cli.json", "xuni_M2_cli.json", "protocol.json", "models/M1_full.json", "models/M2_full.json"]:
        exp.copy(ranking_dir / name, "offline_top3_ranking_not_default", "xuni")
    xrank = load(ranking_dir / "xuni_regression.json")
    key_rows = [{"song_id": "xuni", "stage": k, "predicted_key": xr[k], "user_confirmed_key": "C",
                 "reference_origin": "prior user confirmation recorded in project report; not a persisted UI manual edit"}
                for k in ["cnn", "harmonic_v1", "chroma", "harmonic_v2"]]
    for k in ["M1", "M2"]:
        for row in xrank["models"][k]["ranking"]:
            key_rows.append({"song_id": "xuni", "stage": k, "predicted_key": row["key"], "rank": row["rank"],
                             "cnn_rank": row["cnn_rank"], "score_uncalibrated": row["score"], "user_confirmed_key": "C"})
    csv_write(output / "songs/xuni/key_decision_comparison.csv", key_rows)
    dump(output / "songs/xuni/human_confirmation.json", {"key": "C", "origin": "prior user confirmation documented in PROJECT_REPORT.md",
         "not_persisted_ui_correction": True, "no_chord_reference_provided": True, "automatic_key": "G", "offline_rerank_key": {k:xrank["models"][k]["key"] for k in ["M1", "M2"]}})
    audio_identities.append({"song_id": "xuni", **xr["inputs"]["audio"], "packaged": False, "reason": "results-only archive"})
    # Explicit audit of web saves and reviews, including no-op/validation records.
    web_audit = []
    for p in (ROOT / "projects/web").rglob("corrections.json"):
        d = load(p)
        exp.copy(p, "web_save_audit_not_assumed_user_annotation")
        for sibling in [p.with_name("corrected.lab"), p.parents[2] / "analysis.json"]:
            if sibling.is_file():
                exp.copy(sibling, "web_save_audit_original")
        web_audit.append({"path": rel(p), "edits": len(d.get("edits", [])), "revision": d.get("revision"),
                          "key": d.get("key"), "key_origin": d.get("key_origin"),
                          "included_as_human_correction": False,
                          "reason": "saved edits empty; existing upload_excerpt functional validation record"})
    for p in (ROOT / "projects/web").rglob("key_review.json"):
        d = load(p)
        exp.copy(p, "key_review_audit")
        web_audit.append({"path": rel(p), "manual": d.get("manual"), "actions": [e.get("action") for e in d.get("history", [])],
                          "included_as_human_correction": False, "reason": "only enable actions, no adoption/custom key stored"})
    dump(output / "web_save_audit.json", web_audit)
    dump(output / "missing_and_excluded.json", {"referenced_files_missing_or_external": exp.omitted,
        "excluded_as_user_annotations": [".tmp/key_review_integration: automated browser fixtures", "derived acoustic/functional analyses carrying corrected=true", "web save with no edits and only enable review actions"],
        "not_available": ["complete historical frame-level posterior for every source", "all old HGB per-candidate utility scores", "verified original run/output SHA in old browser exports", "complete user edit/undo history and browser localStorage drafts", "same-run pre-correction JSON for pingfan"],
        "not_run": ["acoustic inference", "training", "feature extraction", "historical full test suite", "10-song byte replay"]})
    csv_write(output / "cases.csv", cases)
    csv_write(output / "all_primary_interval_comparisons.csv", main_rows)
    csv_write(output / "stages.csv", exp.stage_rows)
    csv_write(output / "model_comparison_summary.csv", comparisons)
    csv_write(output / "revision_comparison_summary.csv", revision_rows)
    dump(output / "audio_identities.json", audio_identities)
    candidate_tables = []
    for rec in list(exp.records.values()):
        p = output / rec["package_path"]
        if p.suffix == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                if any("candidate" in x or "score" in x for x in columns):
                    candidate_tables.append({"path": rec["package_path"], "columns": columns,
                                             "row_count": sum(1 for _ in reader),
                                             "reference_columns_present": [x for x in columns if "reference" in x or x.startswith("target_")],
                                             "role": "historical diagnostic table; reference/target columns are offline supervision, never prediction evidence"})
    dump(output / "candidate_table_index.json", candidate_tables)
    make_html(output, cases, main_rows)
    text = f'''# 人工修订与模型原始输出：详细对照包

整理日期：2026-10-06。先打开 `index.html`（解压后直接双击）；Excel 可直接打开带 UTF-8 BOM 的 CSV。

包含 **8 首作品**：老男孩、紅蓮華、使一颗心免于哀伤、凄美地、所以我放弃了音乐、马、平凡之路、虚拟。
保留 **{len(originals)} 份原始人工导出**及其各版修订；另列紅蓮華后续半减七澄清、马/平凡之路的人工规则辅助结果、虚拟的既有主调确认。
`cases.csv` 共 {len(cases)} 个逐和弦对照版本；虚拟只有主调确认，没有伪造逐和弦参考。

## 如何查看

- `index.html`：选择歌曲和修订版本，查看时间、小节、原和弦/级数、人工和弦/级数、主调变化和原保存分数。
- `cases/*/interval_comparison.csv`：完整逐区间对照；`changes_only.csv` 只列差异；`summary.json` 包含主调、小节网格、来源、修订数及缺失时长。
- `cases/*/previous_revision_comparison.csv`：有明确 continued_from_export 的连续版本增量。不把ヨルシカ的两个导出强行串成已证明的操作历史。
- `songs/*/all_model_interval_comparisons.csv`：最新保存修订与同音频其他历史模型结果的逐区间对照。`model_comparison_summary.csv` 保留全部对照，不只选最佳结果。
- `stages.csv` → `songs/*/stages/*/decision_metadata.json`：每个保存阶段的参数、主调候选及分数、小节/速度/转调/循环规则；`interval_decisions.csv` 保留逐段判断、低音证据、原置信值和原附加字段。
- `candidate_table_index.json`：历史候选表的文件、字段与行数；原表保留候选排名、特征、分数、门控/选择结果（以实际保存字段为准）。含 reference/target 的列属于离线诊断/监督，不能当成推理特征。
- `originals/`：逐字节保留源 JSON/LAB/CSV/报告、三来源识别、冻结候选头/根音保护输出、契约及 provenance。各实验为独立对照，并非当前默认串行执行的所有步骤。
- `songs/xuni/`：CNN/v1/chroma/v2 与离线 M1/M2 判断、前三候选和线性分项贡献；原分歧裁决 gate/margin/原因在对应原始证据 JSON。默认 G、v2 建议 C、离线 M1/M2 仍 G；未启用自动覆盖。
- `audio_identities.json`：音源位置与 SHA/时长。包内不复制音频、庞大声学权重、可重算 NPZ；已有权重身份/契约在原 provenance 中保留。
- `source_manifest.json` 与 `package_manifest.json`：源路径、ZIP 内路径、文件字节数与 SHA；`verification.json`：完整性验证。

## 对照与人工来源的边界

旧导出没有原运行输出 SHA。基线按音频路径、偏移与网格核对，属于可追溯的历史对应输出，但不能宣称已证明同次运行。
平凡之路没有找到同次修订前的原 JSON，使用同一剪辑的后期冻结工作台作额外对照，已在该条明确标出。
星穹铁道还保留人工指定转调后的 user_sections，区别于自动 G# 基线。马的 base 也使用既有用户候选调/速度等约束，不是完全无人工输入的识别。
老男孩后两版含 61–78 小节自动复制建议，之后可能再手改；无法凭现存文件还原逐操作来源。不能把整曲、所有 edited=true 或导出 count 当作独立人工金标准。
紅蓮華原导出与后续 D#7→D#m7b5 澄清分别保留；导入副本与继承 corrected 标志的模型实验不重复计为新增人工注释。
虚拟 C 的来源是此前用户确认及项目记录；网页 key_review 只有 enable，没有保存 adopt/custom 改调。

以 bars.spans 为展示与修订时间源；顶层 events 可能陈旧，因此同时报告两种数量与差异，保留两份原始字段。缺少 span.local_key 时可按已保存 modulations 的小节范围取调；整曲多调描述不冒充每段局部调。C/C major 与 Ab/G# 按相同调性比较；未知局部调不计为调性变化。
使用两路边界并集，逐区间对齐而非按行号/小节硬配；缺失和重叠明确标出，不用参考标签补模型空白。起止范围不一致的部分不计入一致率分母。
统计的“相同/不同秒数”是结果一致性，不是新的独立准确率。完整语义使用保留扩展音/显式低音的解析；旧不支持写法按原字符串比较并标记解析错误，不能把它们都合并成 X。
参考中的无 slash 不能证明根音低音；bass_root 是原提取字段，不一定为明确标签低音。置信值不统一为正确概率；人工编辑写入的 1 不是声学置信度。

## 保存不完整的步骤

每首都附真实存在的中间结果。某些旧流程只保存 top1 LAB/最终 span，未保存整帧概率或每候选 HGB 分数：这些不能从已选标签逆向恢复。具体缺项见 `missing_and_excluded.json`；没有重新跑声学网络来冒充旧决策。
工作台空 edits 保存和仅 enable 审阅记录、自动浏览器验证夹具不计人工修订；审计在 `web_save_audit.json`。
本轮只整理、对齐、导出与验证，未训练、推理或重跑历史全套测试。

## 实际命令

```powershell
C:/Users/33573/miniconda3/python.exe app/export_manual_model_comparison.py --output {rel(output)}
```
导出工具源文件在包内。为防覆盖已交付数据，同一路径重复执行会拒绝；重新整理请换输出名。
'''
    (output / "README.md").write_text(text, encoding="utf-8")
    dump(output / "source_manifest.json", list(exp.records.values()))
    checks = {"alignment_probes_passed": probes, "same_input_primary_pairings": len(cases), "original_export_count": len(originals),
              "song_count": len(latest) + 1, "case_count": len(cases), "primary_interval_row_count": len(main_rows),
              "stage_count": len(exp.stage_rows), "source_file_count": len(exp.records),
              "candidate_table_count": len(candidate_tables), "failures": []}
    for rec in exp.records.values():
        if digest(ROOT / rec["source_path"]) != rec["sha256"]:
            checks["failures"].append({"source_changed": rec["source_path"]})
        if digest(output / rec["package_path"]) != rec["sha256"]:
            checks["failures"].append({"copy_hash_mismatch": rec["package_path"]})
    for c in cases:
        rr = [r for r in main_rows if r["case_id"] == c["case_id"]]
        if abs(sum(r["duration_seconds"] for r in rr if r["comparison"] in {"same", "different"}) - c["comparable_seconds"]) > 1e-7:
            checks["failures"].append({"duration_accounting": c["case_id"]})
    if checks["failures"]:
        raise AssertionError(checks)
    dump(output / "verification.json", checks)
    dump(output / "delivery.json", {**checks, "new_inference": False, "new_training": False,
          "created_at_utc": datetime.now(timezone.utc).isoformat(), "export_elapsed_seconds": time.perf_counter() - start,
          "actual_argv": sys.argv, "input_manifest": "source_manifest.json", "limitations": "README.md and missing_and_excluded.json"})
    members = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(output.rglob("*")) if p.is_file()]
    dump(output / "package_manifest.json", members)
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            raise AssertionError(f"ZIP CRC mismatch: {bad}")
        for m in members:
            if hashlib.sha256(z.read(m["path"])).hexdigest() != m["sha256"]:
                raise AssertionError(f"ZIP SHA mismatch: {m['path']}")
    zip_sha = digest(zip_path)
    zip_path.with_suffix(".zip.sha256").write_text(f"{zip_sha}  {zip_path.name}\n", encoding="ascii")
    receipt = {**checks, "zip": rel(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": zip_sha,
               "zip_members": len(members) + 1, "zip_crc_verified": True, "zip_content_sha_verified": True,
               "source_files_unchanged": True, "total_elapsed_seconds": time.perf_counter() - start}
    dump(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve())
