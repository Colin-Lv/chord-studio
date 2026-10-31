import hashlib
import json
from pathlib import Path
import sys
import subprocess
import wave

import joblib
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"app"))
import apply_chord_extensions as apply
from chord_extension_model import FEATURE_SCHEMA, FEATURE_SCHEMA_V2


class Evidence:
    classes_ = np.array([0,1,2])
    def __init__(self, dimensions):
        self.n_features_in_ = dimensions
    def predict_proba(self, features):
        return np.tile([.01,.98,.01],(len(features),1))


def files(tmp_path, *, threshold=1, schema=FEATURE_SCHEMA):
    audio = tmp_path/"song.wav"
    with wave.open(str(audio),"wb") as handle:
        handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(22050)
        handle.writeframes(np.zeros(22050,dtype=np.int16).tobytes())
    lab = tmp_path/"upstream.lab"
    lab.write_bytes(b"# upstream\r\n0.000000 0.2 C:maj\r\n0.2 0.400000 D:min/E\r\n0.4 0.6 N\r\n0.6 0.8 Gsus4\r\n0.8 1 Bbmaj7")
    model = tmp_path/"model.joblib"
    joblib.dump(dict(feature_schema=schema,threshold=threshold,
                     model=Evidence(148 if schema==FEATURE_SCHEMA_V2 else 87)),model)
    return audio,lab,model,tmp_path/"output.lab"


def test_identity_skips_decode_and_preserves_exact_original_bytes(tmp_path,monkeypatch):
    audio,lab,model,output = files(tmp_path)
    monkeypatch.setattr(apply,"decode_audio",lambda *a,**k:pytest.fail("identity must not decode"))
    result = apply.apply_model(audio,lab,model,output)
    assert output.read_bytes()==lab.read_bytes()
    assert result["change_count"]==0 and result["mode"]=="identity" and not result["audio_decoded"]
    assert result["audio_offset"]==0 and result["decoded_duration"] is None
    for key,path in [("audio",audio),("upstream_lab",lab),("model",model)]:
        assert result[key]["sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(result["code"])>=6
    evidence=json.loads(output.with_name(output.name+".evidence.json").read_text(encoding="utf-8"))
    assert all(row["score_kind"]=="disabled_identity" for row in evidence["intervals"])


@pytest.mark.parametrize("schema",[FEATURE_SCHEMA,FEATURE_SCHEMA_V2])
def test_active_head_runs_real_audio_features_and_preserves_times_and_bass(tmp_path,schema):
    audio,lab,model,output = files(tmp_path,threshold=.7,schema=schema)
    before=lab.read_bytes()
    result=apply.apply_model(audio,lab,model,output)
    assert lab.read_bytes()==before
    assert result["change_count"]==2 and result["audio_decoded"]
    _,_,rows,_=apply.read_upstream(output)
    assert [row.chord for row in rows]==["C7","Dm7/E","N","Gsus4","Bbmaj7"]
    assert [(row.start,row.end) for row in rows]==[(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1)]
    assert output.read_bytes().startswith(b"# upstream\r\n0.000000 0.2 C7\r\n0.2 0.400000 Dm7/E")
    assert result["output_lab"]["sha256"]==hashlib.sha256(output.read_bytes()).hexdigest()


def test_inputs_and_existing_outputs_survive_rejected_run(tmp_path):
    audio,lab,model,output=files(tmp_path)
    original=lab.read_bytes()
    with pytest.raises(ValueError,match="must not overwrite"):
        apply.apply_model(audio,lab,model,lab)
    assert lab.read_bytes()==original
    output.write_bytes(b"old completed output")
    lab.write_text("0 0.7 C\n0.5 1 G\n",encoding="utf-8")
    with pytest.raises(ValueError,match="overlapping"):
        apply.apply_model(audio,lab,model,output)
    assert output.read_bytes()==b"old completed output"
    assert not output.with_name(output.name+".provenance.json").exists()


def test_cli_has_no_threshold_tuning_option(tmp_path):
    audio,lab,model,output=files(tmp_path)
    with pytest.raises(SystemExit) as error:
        apply.main([str(audio),str(lab),"--model",str(model),"-o",str(output),"--threshold",".2"])
    assert error.value.code==2


def test_atomic_replace_failure_does_not_truncate_prior_output(tmp_path,monkeypatch):
    target=tmp_path/"existing.lab"
    target.write_bytes(b"previous complete LAB")
    def fail(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(apply.os,"replace",fail)
    with pytest.raises(OSError):
        apply.atomic_write(target,b"new LAB")
    assert target.read_bytes()==b"previous complete LAB"
    assert sorted(path.name for path in tmp_path.iterdir())==["existing.lab"]


def test_actual_cli_loads_standard_artifact_and_writes_identity_sidecars(tmp_path):
    from sklearn.dummy import DummyClassifier
    audio,lab,model,output=files(tmp_path)
    classifier=DummyClassifier(strategy="constant",constant=0).fit(np.zeros((2,87)),[0,1])
    joblib.dump(dict(feature_schema=FEATURE_SCHEMA,threshold=1.,model=classifier),model)
    process=subprocess.run([sys.executable,str(Path(apply.__file__)),str(audio),str(lab),
                            "--model",str(model),"-o",str(output)],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    assert "0 changes | identity" in process.stdout
    assert output.read_bytes()==lab.read_bytes()
    record=json.loads(output.with_name(output.name+".provenance.json").read_text(encoding="utf-8"))
    evidence=output.with_name(output.name+".evidence.json").read_bytes()
    assert record["evidence_sha256"]==hashlib.sha256(evidence).hexdigest()


@pytest.mark.parametrize("extension,accepted",[(.08,True),(.11,False)])
def test_source_frame_rounding_preserves_legal_tail_and_rejects_longer_overrun(tmp_path,extension,accepted):
    audio,lab,model,output=files(tmp_path,threshold=.7)
    lab.write_bytes(lab.read_bytes().replace(b"0.8 1 Bbmaj7",f"0.8 {1+extension} Bbmaj7".encode()))
    if accepted:
        report=apply.apply_model(audio,lab,model,output)
        assert report["source_end_beyond_audio_seconds"]==pytest.approx(extension)
        assert apply.read_upstream(output)[2][-1].end==1+extension
    else:
        with pytest.raises(ValueError,match="beyond the supplied audio"):
            apply.apply_model(audio,lab,model,output)
        assert not output.exists()
