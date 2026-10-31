import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from validate_analysis import audit_analysis


def test_conflicting_legacy_events_are_reported_without_mutating_labels():
    payload = {"duration": 4., "events": [{"start": 0., "end": 2., "chord": "A7"},
                                         {"start": 0., "end": 2., "chord": "F"}],
               "bars": [{"bar": 1, "start": 0., "end": 4., "spans": [{"start": 0., "end": 2., "chord": "F"}]}]}
    report = audit_analysis(payload)
    assert not report["valid"]
    assert {issue["kind"] for issue in report["issues"]} == {"overlap", "events_bars_disagree"}
    assert payload["events"][0]["chord"] == "A7"


def test_adjacent_non_overlapping_segments_are_valid():
    payload = {"duration": 4., "events": [{"start": 0., "end": 2., "chord": "C"},
                                         {"start": 2., "end": 4., "chord": "G"}]}
    assert audit_analysis(payload)["valid"]
