"""Conservative inversion display preserves source labels and acoustic evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import downbeat_external_fusion as fusion
from evaluate_chords import TimedChord, evaluate


def test_suppressed_inferred_bass_remains_a_review_candidate_in_all_views():
    inferred = dict(start=0., end=.25, chord="C", chord_display="C/E", roman_display="IV/VI",
                    local_key="G", bass_root="E", bass_confidence=.95, confidence=.9, manual_note="keep")
    explicit = dict(start=.25, end=1., chord="C/E", chord_display="C/E", roman_display="I/III",
                    bass_root="G", bass_confidence=.99, confidence=.9)
    original = dict(key="C", events=[inferred, explicit],
                    bars=[dict(spans=[deepcopy(inferred), deepcopy(explicit)],
                               beats=[dict(deepcopy(inferred), spans=[deepcopy(inferred)])])])
    payload = fusion.apply_bass_display_policy(deepcopy(original), False)
    assert payload["bass_display_policy"] == "explicit_source_only"
    first, second = payload["events"]
    assert first["chord_display"] == "C" and first["roman_display"] == "IV"
    assert first["inferred_bass_candidate"]["chord_display"] == "C/E"
    assert second["chord_display"] == "C/E" and second["roman_display"] == "I/III"
    assert "inferred_bass_candidate" not in second
    for before, after in zip(original["events"], payload["events"]):
        for field in ("start", "end", "chord", "bass_root", "bass_confidence", "confidence"):
            assert after[field] == before[field]
    assert first["manual_note"] == "keep"
    assert payload["bars"][0]["spans"][0]["chord_display"] == "C"
    assert payload["bars"][0]["beats"][0]["chord_display"] == "C"
    assert payload["bars"][0]["beats"][0]["spans"][0]["chord_display"] == "C"
    legacy = fusion.apply_bass_display_policy(deepcopy(original), True)
    assert legacy["events"] == original["events"]
    reference = [TimedChord(0, .25, "C:maj"), TimedChord(.25, 1., "C:maj/3")]
    def score(data):
        return evaluate(reference, [TimedChord(row["start"], row["end"], row["chord_display"]) for row in data["events"]])
    a, b = score(original), score(payload)
    for metric in ("root_accuracy", "triad_accuracy", "tetrad_accuracy", "root_wcsr", "triad_wcsr", "tetrad_wcsr", "prediction_coverage"):
        assert a[metric] == b[metric]
    assert a["tetrad_inv_accuracy"] < b["tetrad_inv_accuracy"] == 1.


@pytest.mark.parametrize("infer,expected", [(False, "C"), (True, "C/E")])
def test_full_fusion_defaults_to_source_inversions_only(tmp_path, monkeypatch, infer, expected):
    beats = [dict(start=i * .5, end=(i + 1) * .5, beat=i + 1, chord="C", confidence=1.,
                  bass_root="E", bass_confidence=.99) for i in range(4)]
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(duration=2., key="C", bars=[dict(bar=1, start=0., end=2., beats=beats,
                    spans=[dict(start=0., end=2., chord="C", confidence=1., bass_root="E", bass_confidence=.99)])])))
    lab = tmp_path / "chords.lab"
    lab.write_text("0 2 C:maj\n")
    argv = ["fusion", str(base), str(base), str(lab), "-o", str(tmp_path / "result"),
            "--beat-grid-from-base", "--meter", "4/4", "--no-html"]
    if infer:
        argv.append("--infer-slash-bass")
    monkeypatch.setattr(sys, "argv", argv)
    assert fusion.main() == 0
    payload = json.loads((tmp_path / "result_analysis.json").read_text(encoding="utf-8"))
    assert payload["fusion_provenance"]["infer_slash_bass"] is infer
    assert {row["chord"] for row in payload["events"]} == {"C"}
    assert {row["chord_display"] for row in payload["events"]} == {expected}
