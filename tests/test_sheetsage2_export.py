"""Model export preserves the canonical clip clock and raw chord boundaries."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from sheetsage2_export import export_studio, model_beat_regions


def make_run(path):
    run = dict(status="ok", clip_duration_seconds=2., source_audio="original.mp3", original_offset_seconds=30.)
    (path / "run.json").write_text(json.dumps(run))
    (path / "chord.lab").write_text("0 0.125 C:maj7/3\n0.125 2 A:min\n")
    (path / "key.lab").write_text("0 1 C:major\n1 2 A:minor\n")
    beats = [dict(time=time, beat=beat, meter_numerator=4, meter_denominator=4)
             for time, beat in [(0.1, 1), (0.6, 2), (1.1, 3), (1.6, 4)]]
    (path / "beats.json").write_text(json.dumps({"beats": beats}))
    return run, beats


def test_export_keeps_audio_clock_local_and_chords_exact(tmp_path):
    make_run(tmp_path)
    output = json.loads(export_studio(tmp_path).read_text(encoding="utf-8"))
    assert output["audio_offset"] == 0
    assert output["original_offset_seconds"] == 30
    assert [(event["start"], event["end"], event["chord"]) for event in output["events"]] == [(0, .125, "Cmaj7/E"), (.125, 2, "Am")]
    assert output["key_timeline"]["segments"][1]["key"] == "Am"
    assert output["bars"][0]["grid_status"] == "uncovered"
    assert output["events"][0]["confidence_status"] == "not_exported_by_model"
    assert (tmp_path / "sheetsage2_studio.html").is_file()


def test_invalid_local_beat_never_fabricates_a_constant_grid(tmp_path):
    _, beats = make_run(tmp_path)
    beats[-1]["meter_numerator"] = 3
    (tmp_path / "beats.json").write_text(json.dumps({"beats": beats}))
    output = json.loads(export_studio(tmp_path).read_text(encoding="utf-8"))
    assert output["bars"][0]["beats"] == []
    assert output["bars"][0]["grid_status"] == "uncovered"
    assert output["export_warnings"]


def test_variable_meter_keeps_local_bars_and_exact_events(tmp_path):
    make_run(tmp_path)
    rows = [(0., 1, 2, 4), (.3, 2, 2, 4), (.6, 1, 3, 8), (.8, 2, 3, 8),
            (1., 2, 3, 8), (1.2, 3, 3, 8), (1.4, 1, 1, 8), (1.6, 1, 1, 8), (1.8, 1, 1, 8)]
    beats = [dict(time=t, beat=b, meter_numerator=n, meter_denominator=d) for t,b,n,d in rows]
    (tmp_path / "beats.json").write_text(json.dumps({"beats": beats}))
    output = json.loads(export_studio(tmp_path).read_text(encoding="utf-8"))
    placed = [bar for bar in output["bars"] if bar["beats"]]
    assert [bar["meter"] for bar in placed] == ["2/4", "3/8", "1/8", "1/8"]
    assert [bar["beats_per_bar"] for bar in placed] == [2, 3, 1, 1]
    assert placed[1]["grid_status"] == "incomplete"
    assert [beat["beat"] for beat in placed[1]["beats"]] == [1, 2, 2, 3]
    assert output["meter_mode"] == "variable"
    assert output["bars"][-1]["grid_status"] == "uncovered"
    assert [(event["start"], event["end"]) for event in output["events"]] == [(0., .125), (.125, 2.)]
    assert len({bar["bar"] for bar in output["bars"]}) == len(output["bars"])


def test_variable_meter_rejects_duplicate_marker_time():
    rows = [dict(time=0., beat=b, meter_numerator=4, meter_denominator=4) for b in [1, 2]]
    with pytest.raises(ValueError, match="strictly"):
        model_beat_regions(rows, 2.)


def test_incomplete_generation_cannot_be_exported_as_finished(tmp_path):
    run, _ = make_run(tmp_path)
    run["status"] = "incomplete"
    (tmp_path / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="completed"):
        export_studio(tmp_path)
