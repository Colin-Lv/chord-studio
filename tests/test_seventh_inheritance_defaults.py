"""Full fusion must not fabricate template sevenths under its default policy."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import downbeat_external_fusion as fusion


@pytest.mark.parametrize("source,vote,expected", [
    ("C", "C7/E", "C7"),
    ("C/E", "C7/G", "C7/E"),
    ("C:maj/3", "C7/G", "C7/E"),
])
def test_quality_inheritance_does_not_promote_template_bass_to_source(source, vote, expected):
    assert fusion.inherit_seventh_colors([source], [(vote, .9)], .45) == [expected]


@pytest.mark.parametrize("threshold, expected", [(None, "C"), (0.45, "C7")])
def test_complete_fusion_keeps_triads_unless_legacy_color_is_requested(tmp_path, monkeypatch, threshold, expected):
    bars = []
    for bar in range(2):
        beats = [{"start": bar * 2 + beat * .5, "end": bar * 2 + (beat + 1) * .5,
                  "beat": beat + 1, "chord": "C7", "confidence": .9,
                  "bass_root": "C", "bass_confidence": 0} for beat in range(4)]
        bars.append({"bar": bar + 1, "start": bar * 2, "end": (bar + 1) * 2,
                     "beats": beats, "spans": [{"start": bar * 2, "end": (bar + 1) * 2,
                                                "chord": "C7", "confidence": .9}]})
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"source": str(tmp_path / "audio.wav"), "duration": 4,
                                "key": "C", "bars": bars, "analysis_model": "template"}), encoding="utf-8")
    lab = tmp_path / "chords.lab"
    lab.write_text("0 4 C:maj\n")
    output = tmp_path / "fusion"
    command = ["fusion", str(base), str(base), str(lab), "-o", str(output),
               "--beat-grid-from-base", "--meter", "4/4"]
    if threshold is not None:
        command += ["--dominant-seventh-threshold", str(threshold)]
    monkeypatch.setattr(sys, "argv", command)
    assert fusion.main() == 0
    result = json.loads((tmp_path / "fusion_analysis.json").read_text(encoding="utf-8"))
    assert {event["chord"] for event in result["events"]} == {expected}
    assert result["fusion_provenance"]["template_seventh_inheritance_enabled"] == (threshold is not None)
    assert (tmp_path / "fusion_studio.html").is_file()
