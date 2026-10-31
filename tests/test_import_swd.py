from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from import_swd import convert_reference, safe_member


def test_swd_normalizes_absolute_bass_without_losing_extension_or_timing(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text('start;end;shorthand\n0.125;1.750;"C:maj7/G"\n1.750;2.750;X\n', encoding="utf-8")
    text, report = convert_reference(path)
    assert text == "0.125 1.750 C:maj7/5\n1.750 2.750 X\n"
    assert report["explicit_unknown_intervals"] == 1
    assert not report["timing_changed"]


@pytest.mark.parametrize("value", ["", "not_a_chord"])
def test_swd_never_converts_unsupported_or_blank_labels_to_silence(tmp_path, value):
    path = tmp_path / "reference.csv"
    path.write_text(f"start;end;shorthand\n0;1;{value}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        convert_reference(path)


def test_swd_rejects_overlaps_and_path_escape(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("start;end;shorthand\n0;2;C:maj\n1;3;G:7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Overlapping"):
        convert_reference(path)
    with pytest.raises(ValueError, match="escapes"):
        safe_member(tmp_path, "../outside.csv")
