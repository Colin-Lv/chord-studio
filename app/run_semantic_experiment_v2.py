"""E1 only: new complete semantics, old utility/HGB/margin/post-selection guard.

Reuses B52/dev14 and existing NNLS evidence as a KNOWN diagnostic. No new blind
test claim, no E2 target tuning, no new acoustic inference. Baselines from the
paired policy diagnostic are re-used only after input/model hash verification.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from candidate_data_v2 import load_inputs, standard_reference
from candidate_diagnostics_v2 import candidate_diagnostics, field_metrics, change_diagnostics, summarize_song_metrics
from candidate_targets_v2 import annotation_coverage
from evaluate_chords import evaluate, read_reference, validate_timeline
from experiment_protocol_v2 import signature, write_new_json
from multisource_candidates_v2 import SOURCE_NAMES
from multisource_features_v2 import SCHEMA as FEATURE_SCHEMA, DIMENSION, automatic_candidate_features
from multisource_model_v2 import HGB_PARAMETERS, fit_hgb, save_artifact, load_artifact, score_candidates
from multisource_root_guard import preserve_raw_root
from multisource_root_guard_v2 import frozen_v1_full_head_from_scores
from train_multisource_candidate_reliability import reference_utilities

ROOT=Path(__file__).resolve().parents[1]
SESSION=ROOT/"benchmarks/reports/session_20260926_accuracy2"


def prepare_rows(training_report):
    path=Path(training_report)
    report=json.loads(path.read_text(encoding="utf-8"))
    for item in report["protocol"]["implementation"]:
        if signature(item["path"])!=item:
            raise ValueError("Historical feature/utility implementation changed: "+item["path"])
    manifest=Path(report["protocol"]["manifest"]["path"])
    if signature(manifest)!=report["protocol"]["manifest"]:
        raise ValueError("Original training manifest changed")
    with manifest.open(encoding="utf-8-sig",newline="") as stream:
        table={r["id"]:r for r in csv.DictReader(stream)}
    rows=[]
    for source in report["inputs"]:
        sid=source["id"]
        row=dict(id=sid,audio=source["audio"]["path"],notes=source["notes"]["path"],
            reference=str((manifest.parent/table[sid]["chord_reference"]).resolve()),
            input_identities={"audio":source["audio"],"notes":source["notes"]})
        for name,item in zip(SOURCE_NAMES,source["labs"],strict=True):
            row[name]=item["path"]; row["input_identities"][name]=item
        row["split"]="development" if sid in report["protocol"]["development_ids"] else "fit"
        rows.append(row)
    if {r["id"] for r in rows if r["split"]=="fit"}!=set(report["protocol"]["training_ids"]):
        raise ValueError("B52 fit IDs changed")
    return rows,report


def checked_baseline(song, old_row):
    original=old_row["inputs"]
    for name in ("audio","notes","reference"):
        if song["input_identities"][name]!=original[name]:
            raise ValueError("Baseline used different "+name)
    for name,source in zip(SOURCE_NAMES,original["sources"],strict=True):
        if song["input_identities"][name]!=source["lab"]:
            raise ValueError("Baseline used a different source: "+name)
    output=old_row["predictions"]["frozen_v1"]
    if signature(output["path"])!=output:
        raise ValueError("Frozen v1 comparison prediction changed")
    return validate_timeline(read_reference(Path(output["path"])),"paired frozen v1")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report",type=Path,default=SESSION/"multisource_candidate_reliability/report.json")
    parser.add_argument("--baseline-report",type=Path,required=True,help="Completed paired dev14 v1-policy diagnostic")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite an E1 experiment")
    rows,old_report=prepare_rows(args.training_report)
    baseline=json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if baseline["model"]["model"]!=old_report["development_model"] or baseline["failure_count"]:
        raise ValueError("Need a completed paired B52 model diagnostic on dev14")
    baseline_rows={row["id"]:row for row in baseline["songs"]}
    margin=float(old_report["selected"]["margin"])
    args.output.mkdir(parents=True)
    protocol=dict(schema="E1_complete_semantics_legacy_objective_v2",status="fixed_before_new_fit_or_scores",
        feature_schema=FEATURE_SCHEMA,feature_dimension=DIMENSION,HGB=HGB_PARAMETERS,margin=margin,
        objective=dict(root=.25,triad=.50,tetrad=.25),guard="unchanged v1 post-selection root guard",
        changed_factor="Complete semantic candidate representation: full extensions, named union supports and unannotated bass/state representation",
        unchanged=["B52 fit/dev14 diagnostic split","old utility weights","HGB parameters","margin","post-selection guard","standard evaluator"],
        training_report=signature(args.training_report),baseline_report=signature(args.baseline_report),
        implementation=signature(__file__),fit_ids=[r["id"] for r in rows if r["split"]=="fit"],
        development_ids=[r["id"] for r in rows if r["split"]=="development"],
        sources=list(SOURCE_NAMES),known_regression_only=True,new_blind_test=False,
        source_identity_limitation="Legacy three-source training LAB generation identities remain incomplete; file bytes and current pairing are checked",
        E2_trained=False,E2_reason="Requires reliable masks and separately frozen grouped development/calibration inputs; E1 does not choose new weights",
        acoustic_inference_calls=0,training_calls=1,model_promoted=False)
    write_new_json(args.output/"protocol.json",protocol)
    wall,cpu=time.perf_counter(),time.process_time()
    songs=[]; nnls=None
    for index,row in enumerate(rows):
        song=load_inputs(row,references=True,expected_nnls=nnls)
        nnls=song["nnls_contract"]
        song["split"]=row["split"]
        if song["split"]=="development":
            song["frozen_v1"]=checked_baseline(song,baseline_rows[song["id"]])
        song["features"],song["groups"]=automatic_candidate_features(song["raw"],song["sources"],song["evidence"])
        song["candidate_diagnostics"]=candidate_diagnostics(song["raw"],song["sources"],song["reference"],song["groups"])
        song["annotation_coverage"]=annotation_coverage(song["reference"])
        songs.append(song)
        if index%8==0 or index+1==len(rows):
            print(f"E1 prepared {index+1}/{len(rows)}",flush=True)
    x,y,w=[],[],[]
    fit=[song for song in songs if song["split"]=="fit"]
    for song in fit:
        target,weight=reference_utilities(song["raw"],song["groups"],standard_reference(song["reference"]))
        keep=np.isfinite(target)&(weight>0)
        x.append(song["features"][keep]); y.append(target[keep]); w.append(weight[keep])
    x,y,w=np.concatenate(x),np.concatenate(y),np.concatenate(w)
    model=fit_hgb(x,y,w)
    model_path=args.output/"semantic_e1_B52_v2.joblib"
    model_identity,contract_identity=save_artifact(model_path,model,
        objective=dict(schema="legacy_root_triad_tetrad_for_E1_only",weights=protocol["objective"]),
        margin=margin,nnls_contract=nnls,provenance=dict(protocol=signature(args.output/"protocol.json"),
            implementation=signature(__file__),utility_implementation=signature(ROOT/"app/train_multisource_candidate_reliability.py"),
            training_inputs=[dict(id=s["id"],**s["input_identities"]) for s in fit]),diagnostic_only=True)
    artifact,_=load_artifact(model_path)
    scored=[]; predictions=args.output/"predictions"; predictions.mkdir()
    for song in songs:
        if song["split"]!="development":
            continue
        scores=score_candidates(artifact,song["features"])
        full=frozen_v1_full_head_from_scores(song["raw"],song["groups"],scores,margin)
        prediction,_=preserve_raw_root(song["raw"],full)
        if [(r.start,r.end) for r in prediction]!=[(r.start,r.end) for r in song["raw"]]:
            raise AssertionError("E1 changed the raw time axis")
        pred_path=predictions/(song["id"]+".lab")
        pred_path.write_text("".join(f"{r.start!r}\t{r.end!r}\t{r.chord}\n" for r in prediction),encoding="utf-8")
        old=song["frozen_v1"]
        systems=dict(raw=song["raw"],chordino=song["sources"]["chordino"],largevocab=song["sources"]["largevocab"],frozen_v1=old,semantic_v2=prediction)
        scored.append(dict(id=song["id"],input_identities=song["input_identities"],
            metrics={name:evaluate(standard_reference(song["reference"]),values) for name,values in systems.items()},
            field_metrics={name:field_metrics(song["reference"],values) for name,values in systems.items()},
            changes_vs_raw=change_diagnostics(song["reference"],song["raw"],prediction),
            changes_vs_v1=change_diagnostics(song["reference"],old,prediction),
            candidates=song["candidate_diagnostics"],annotation_coverage=song["annotation_coverage"],prediction=signature(pred_path)))
    diagnostic_rows=[dict(id=s["id"],split=s["split"],inputs=s["input_identities"],
        timeline_repairs=s["timeline_repairs"],candidates=s["candidate_diagnostics"],annotation_coverage=s["annotation_coverage"]) for s in songs]
    peak=None
    try:
        import psutil
        mem=psutil.Process().memory_info()
        peak=getattr(mem,"peak_wset",None)
    except ImportError:
        pass
    report=dict(schema="semantic_E1_diagnostic_v2",status="complete_diagnostic_not_promoted",model=model_identity,
        contract=contract_identity,protocol=signature(args.output/"protocol.json"),fit_shape=list(x.shape),
        summary=summarize_song_metrics(scored),songs=scored,training_and_candidate_diagnostics=diagnostic_rows,
        regressions=[dict(id=s["id"],metric=m,delta=s["metrics"]["semantic_v2"][m]-s["metrics"]["frozen_v1"][m])
                     for s in scored for m in ("triad_accuracy","tetrad_accuracy","tetrad_inv_accuracy")
                     if s["metrics"]["semantic_v2"][m]<s["metrics"]["frozen_v1"][m]-1e-12],
        resources=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,peak_working_set_bytes=peak,
                       numeric_threads=2,fit_calls=1,acoustic_inference_calls=0),
        unverified=["E2 masked-target real training","cross-style grouped development","calibration","new sealed test","full default workbench scores"],
        known_regression_only=True)
    write_new_json(args.output/"report.json",report)
    print(json.dumps(dict(report=str(args.output/"report.json"),summary=report["summary"],resources=report["resources"]),ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
