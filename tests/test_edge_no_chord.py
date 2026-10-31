"""Explicit edge N is evidence; an absent source interval remains a gap."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import downbeat_external_fusion as fusion
import analyze_with_madmom as wrapper
import generate_external_fusion_from_manifest as manifest_tool
from chord_studio import BeatInterval, ExternalChordHint
from evaluate_chords import TimedChord, evaluate, read_prediction
from validate_analysis import audit_analysis


def test_segment_edges_preserve_explicit_n_but_never_cover_source_gaps():
    beats = [BeatInterval(1, n + 1, 1. + n, 2. + n) for n in range(4)]
    hints = [ExternalChordHint(0., .4, "N"), ExternalChordHint(.7, 1., "G"),
             ExternalChordHint(1., 5., "C"), ExternalChordHint(5., 6., "N")]
    arguments = (beats, 6., hints, [], 0., "off", .001, 0.)
    legacy = fusion.edge_fill_candidates(*arguments, mode="segments")
    preserved = fusion.edge_fill_candidates(*arguments, mode="segments", include_no_chord=True)
    assert [(row.start, row.end, row.chord) for row in preserved] == [(0., .4, "N"), (.7, 1., "G"), (5., 6., "N")]
    assert [row for row in preserved if row.chord != "N"] == legacy
    assert fusion.edge_fill_candidates(*arguments, mode="segments", include_no_chord=False) == legacy
    assert not fusion.edge_fill_candidates(beats, 6., [], [], 0., "off", .001, 0.,
                                           mode="segments", include_no_chord=True)


@pytest.mark.parametrize("include_n", [True, False])
def test_complete_fusion_edge_n_matches_bar_views_and_records_policy(tmp_path, monkeypatch, include_n):
    beats = [dict(beat=n + 1, start=1. + n, end=2. + n, chord="C", confidence=1.) for n in range(4)]
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(duration=6., key="C", bars=[dict(bar=1, start=1., end=5., beats=beats, spans=[])])))
    lab = tmp_path / "source.lab"
    lab.write_text("0 1 N\n1 5 C:maj\n5 6 N\n")
    command = ["fusion", str(base), str(base), str(lab), "-o", str(tmp_path / "result"),
               "--beat-grid-from-base", "--meter", "4/4", "--no-html"]
    if not include_n:
        command.append("--no-edge-fill-include-no-chord")
    monkeypatch.setattr(sys, "argv", command)
    assert fusion.main() == 0
    path = tmp_path / "result_analysis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fusion_provenance"]["edge_fill_include_no_chord"] is include_n
    assert audit_analysis(payload)["valid"]
    assert [(row["start"], row["end"], row["chord"]) for row in payload["events"]] == (
        [(0., 1., "N"), (1., 5., "C"), (5., 6., "N")] if include_n else [(1., 5., "C")])
    if include_n:
        assert [bar["bar"] for bar in payload["bars"]] == [0, 1, 2]
        assert payload["bars"][0]["grid_status"] == payload["bars"][-1]["grid_status"] == "uncovered"
        assert payload["bars"][0]["beats"] == payload["bars"][-1]["beats"] == []
    score = evaluate([TimedChord(0., 1., "N"), TimedChord(1., 5., "C:maj"), TimedChord(5., 6., "N")], read_prediction(path))
    assert score["prediction_coverage"] == pytest.approx(1. if include_n else 4. / 6.)
    # Under strict-missing scoring, omitted edge predictions are errors even
    # when the reference there is N; silence is not an implicit gap filler.
    expected_wcsr = 1. if include_n else 4. / 6.
    assert score["triad_wcsr"] == pytest.approx(expected_wcsr)
    assert score["tetrad_wcsr"] == pytest.approx(expected_wcsr)


@pytest.mark.parametrize("module,args", [(wrapper, ["song.wav"]), (manifest_tool, ["manifest.csv", "labs"])])
def test_entrypoints_default_to_preserving_n_with_explicit_legacy_opt_out(monkeypatch, module, args):
    monkeypatch.setattr(sys, "argv", ["entry", *args])
    assert module.parse_args().edge_fill_include_no_chord is True
    monkeypatch.setattr(sys, "argv", ["entry", *args, "--no-edge-fill-include-no-chord"])
    assert module.parse_args().edge_fill_include_no_chord is False
