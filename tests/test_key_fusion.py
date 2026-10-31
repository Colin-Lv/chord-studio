"""Engineering/regression checks; synthetic scores do not measure acoustics."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import key_fusion as k
from evaluate_global_key_fusion import confusion, evaluate


def scores(first, second=None):
    values = {key:.001 for key in k.harmonic.key_candidates()}
    values[first] = .65
    if second: values[second] = .3
    return dict(key=first, candidates=[dict(key=key,score=value) for key,value in values.items()])


def example(labels, cnn="C", joint="C", alternative=None):
    if isinstance(labels,str): labels=labels.split()*8
    events=[dict(start=i*2,end=(i+1)*2,label=label) for i,label in enumerate(labels)]
    bars=[dict(bar=i+1,start=i*2,end=(i+1)*2) for i in range(len(labels))]
    neutral=dict(key=joint,key_source="functional_joint",key_candidates=scores(joint)["candidates"],
                 duration=len(labels)*2,bars=bars)
    return scores(cnn,alternative),neutral,events


@pytest.mark.parametrize("loop,key",[("C G Am F","C"),("G D Em C","G")])
def test_standard_loops(loop,key):
    result=k.analyze_key_evidence(*example(loop,key,key))
    assert result["key"]==key and result["status"]=="resolved"
    assert result["evidence"]["window_timeline"]["bar_shares"][key]==1


def test_consensus_can_override_cnn_without_averaging_scores():
    result=k.analyze_key_evidence(*example("C G Am F","G","C","C"))
    assert result["key"]=="C" and result["conflict"]["cnn_overridden"]
    assert result["evidence"]["madmom_cnn"]["original"]["key"]=="G"


def test_v_entry_is_not_veto_and_conflicting_rule_is_visible():
    result=k.analyze_key_evidence(*example("G Am F C","C","C","G"))
    assert result["key"]=="C"  # provisional CNN retained, no false rejection
    assert result["status"]=="ambiguous" and result["needs_review"]
    rows=result["evidence"]["raw_chord_functional"]["candidates"]
    assert any(r["key"]=="C" and r["score"]>0 for r in rows)
    # The legacy rule still prefers G for this rotated loop: expose the limitation.
    assert result["evidence"]["raw_chord_functional"]["key"]=="G"


def test_borrowed_chords_do_not_force_modulation():
    labels="C G Am F".split()*16
    labels[21]="Bb";labels[43]="Fm"
    result=k.analyze_key_evidence(*example(labels))
    assert result["key"]=="C" and result["status"]=="resolved"
    assert not result["evidence"]["window_timeline"]["modulation_detected"]


@pytest.mark.parametrize("major,minor,loop",[("C","Am","Am Dm E Am"),("G","Em","Em Am B Em")])
def test_minor_bias_is_not_silently_trusted(major,minor,loop):
    result=k.analyze_key_evidence(*example(loop,minor,minor,major))
    assert result["key"]==minor and result["status"]=="ambiguous"
    assert result["needs_review"]
    assert confusion(major,minor)==confusion(minor,major)=="relative_major_minor"


@pytest.mark.parametrize("labels",[["N"]*16,["X"]*16,["C"]*16,["C","G"]])
def test_sparse_or_unpitched_abstains(labels):
    result=k.analyze_key_evidence(*example(labels))
    assert result["status"]=="ambiguous" and result["needs_review"]


def test_local_modulation_keeps_home_key_and_timeline():
    labels="C G Am F".split()*12+"G D Em C".split()*4+"C G Am F".split()*8
    result=k.analyze_key_evidence(*example(labels))
    timeline=result["evidence"]["window_timeline"]
    assert result["key"]=="C" and timeline["modulation_detected"] and result["needs_review"]
    assert [s["key"] for s in timeline["segments"]]==["C","G","C"]
    assert sum(timeline["bar_shares"].values())<=1.0000001
    assert len({b["bar"] for b in timeline["bar_assignments"]})==timeline["eligible_bars"]


def test_normalization_ties_enharmonic_keys_and_scaling():
    norm=k.normalize_key_scores([("Db",2),("C#",2),("Am",1),("D",2)])
    assert len(norm["candidates"])==3 and norm["margin"]==0
    assert [r["rank"] for r in norm["candidates"]]==[1,1,3]
    a=k.normalize_key_scores({"C":2,"G":1,"Am":0})
    b=k.normalize_key_scores({"C":21,"G":11,"Am":1})
    assert a["relative_margin"]==b["relative_margin"]
    for value in [float("nan"),float("inf")]:
        with pytest.raises(ValueError):k.normalize_key_scores({"C":value})


def test_duration_weights_exact_clipping_and_no_n_evidence():
    rows=k.timed_rows([dict(start=0,end=1,label="N"),dict(start=1,end=9,label="C:maj"),dict(start=9,end=12,label="G:maj")],10)
    assert [(r.chord,r.weight) for r in rows]==[("C",8),("G",1)]
    with pytest.raises(ValueError):k.timed_rows([dict(start=0,end=3,label="C"),dict(start=2,end=4,label="G")],4)


def test_key_conditioned_evidence_rejected_and_human_priority():
    cnn,neutral,events=example("C G Am F")
    neutral["key_source"]="provided"
    result=k.analyze_key_evidence(cnn,neutral,events)
    assert result["status"]=="ambiguous" and not result["evidence"]["functional_joint"]["available"]
    human=k.analyze_key_evidence(cnn,neutral,events,requested_key="F#m")
    assert human["key"]=="F#m" and human["source"]=="provided"


def test_missing_cnn_and_equal_joint_do_not_resolve():
    cnn,neutral,events=example("C G Am F")
    assert k.analyze_key_evidence(None,neutral,events)["status"]=="ambiguous"
    neutral["key_candidates"].append(dict(key="G",score=.65))
    assert k.analyze_key_evidence(cnn,neutral,events)["status"]=="ambiguous"


def test_flat_or_incomplete_cnn_is_not_top_two_support():
    cnn,neutral,events=example("C G Am F")
    flat=copy.deepcopy(cnn)
    for row in flat["candidates"]:row["score"]=1/24
    assert not k.analyze_key_evidence(flat,neutral,events)["gate_checks"]["cnn_support"]
    cnn["candidates"]=cnn["candidates"][:2]
    result=k.analyze_key_evidence(cnn,neutral,events)
    assert result["status"]=="ambiguous" and not result["gate_checks"]["complete_candidates"]


def test_viterbi_margin_must_belong_to_selected_key(monkeypatch):
    cnn,neutral,events=example("C G Am F")
    monkeypatch.setattr(k.harmonic,"viterbi_window_track",lambda *a:[
        k.harmonic.KeyWindow(start,start+7,"G",1.,999.) for start in range(1,26,2)])
    windows=k.summarize_key_windows(events,neutral["bars"],neutral["duration"])
    assert windows["bar_shares"]=={}
    assert all(w["margin"]==0 for w in windows["windows"])
    assert windows["segments"][0]["status"]=="tentative"
    assert not windows["modulation_detected"]


def test_title_reference_and_final_chords_are_not_features():
    cnn,neutral,events=example("C G Am F")
    before=k.analyze_key_evidence(cnn,neutral,events)
    neutral.update(title="different song",reference="G",events=[dict(chord="F#")])
    assert before==k.analyze_key_evidence(cnn,neutral,events)


@pytest.mark.parametrize("cnn_key,joint,expected_passes",[("C","C",1),("G","C",2)])
def test_neutral_then_at_most_one_final_pass(tmp_path,cnn_key,joint,expected_passes):
    cnn,neutral,events=example(["C"]*16,cnn_key,joint)
    base=tmp_path/"base.json";lab=tmp_path/"raw.lab"
    lab.write_text("0 32 C:maj\n")
    calls=[]
    def run(command):
        calls.append(command)
        payload=copy.deepcopy(neutral)
        if "--key" in command:
            payload.update(key=command[command.index("--key")+1],key_source="provided")
        base.write_text(json.dumps(payload))
    result=k.run_neutral_and_fuse(["python","studio.py"],base,lab,cnn,run)
    assert len(calls)==expected_passes
    assert "--key" not in calls[0] and "--all-key-candidates" in calls[0]
    payload=json.loads(base.read_text())
    assert payload["key_source"]==k.SCHEMA
    assert payload["key_estimation"]["madmom_cnn"]==cnn
    assert json.loads(Path(result["neutral_artifact"]).read_text())["key_source"]=="functional_joint"
    with pytest.raises(ValueError):k.run_neutral_and_fuse(["--key","C"],base,lab,cnn,run)


FIXTURE=Path(__file__).parent/"fixtures/key_fusion_xuni"


def xuni():
    return k.analyze_key_evidence(json.loads((FIXTURE/"cnn.json").read_text()),
        json.loads((FIXTURE/"neutral.json").read_text()),k.read_raw_lab(FIXTURE/"raw_madmom.lab"))


def test_xuni_conflict_visible_without_threshold_tuning():
    result=xuni()
    assert result["conflict"]["source_tops"]==dict(madmom_cnn="G",functional_joint="C",raw_chord_functional="C")
    assert result["needs_review"] and result["status"]=="ambiguous"
    assert not result["gate_checks"]["stable_windows"]
    assert len(result["evidence"]["madmom_cnn"]["candidates"])==24


@pytest.mark.xfail(strict=True,reason="Deployment gate NOT met: frozen stability rule rejects Xuni C override; no per-song tuning")
def test_xuni_requested_deployment_gate():
    result=xuni()
    assert result["key"]=="C" and result["conflict"]["cnn_overridden"]


def test_evaluation_reference_changes_only_scoring(tmp_path):
    import csv
    manifest=tmp_path/"manifest.csv"
    def run(reference,output):
        with manifest.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["id","cnn","neutral","raw","reference"])
            writer.writeheader();writer.writerow(dict(id="test",cnn=str((FIXTURE/"cnn.json").resolve()),
                neutral=str((FIXTURE/"neutral.json").resolve()),raw=str((FIXTURE/"raw_madmom.lab").resolve()),reference=reference))
        return evaluate(manifest,tmp_path/output)
    a,b=run("C","a"),run("G","b")
    assert a["songs"][0]["predictions"]==b["songs"][0]["predictions"]
    assert a["songs"][0]["confusions"]!=b["songs"][0]["confusions"]
    assert a["summary"]["fused"]["resolved_wrong"]==0
    assert a["unresolved_conflicts"]==1 and not a["independent_validation_completed"]


@pytest.mark.parametrize("human",[None,"D"])
def test_cli_downstream_key_and_manual_priority(tmp_path,monkeypatch,human):
    import analyze_with_madmom as cli
    import key_bridge
    import chord_studio
    audio=tmp_path/"audio.wav";audio.write_bytes(b"audio")
    lab=tmp_path/"raw.lab";lab.write_text("0 64 C:maj\n")
    out=tmp_path/"result"
    argv=["cli",str(audio),"--madmom-lab",str(lab),"-o",str(out),"--key-source","fused","--structure","none"]
    if human:argv += ["--key",human]
    monkeypatch.setattr(sys,"argv",argv)
    cnn,neutral,_=example("C G Am F")
    cnn_calls=[];commands=[]
    def estimate(*args):cnn_calls.append(args);return cnn
    monkeypatch.setattr(key_bridge,"run_madmom_key",estimate)
    monkeypatch.setattr(cli,"cache_matches_audio",lambda *a:(True,"test"))
    monkeypatch.setattr(chord_studio,"write_html",lambda *a:None)
    def run(command):
        commands.append(command)
        output=Path(command[command.index("-o")+1])
        payload=copy.deepcopy(neutral)
        if "--key" in command:payload.update(key=command[command.index("--key")+1],key_source="provided")
        output.with_name(output.name+"_analysis.json").write_text(json.dumps(payload))
    monkeypatch.setattr(cli,"run",run)
    assert cli.main()==0
    downstream=next(c for c in commands if Path(c[1]).name=="downbeat_external_fusion.py")
    assert downstream[downstream.index("--key")+1]==(human or "C")
    saved=json.loads((tmp_path/"result_analysis.json").read_text())
    from key_disagreement import SCHEMA as resolver_schema
    assert saved["key_source"]==("provided" if human else resolver_schema)
    assert len(cnn_calls)==(0 if human else 1)
    if human:assert "key_fusion" not in saved


def test_full_candidate_export_preserves_default_selection(monkeypatch):
    import chord_studio as studio
    # Complete export is opt-in; legacy top-eight output remains the default.
    from inspect import signature
    assert signature(studio.select_meter_and_key).parameters["retain_all_key_candidates"].default is False
    assert not studio.parse_args(["a.wav"]).all_key_candidates
