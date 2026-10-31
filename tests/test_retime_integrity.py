"""Retiming must preserve annotations, gaps and unsupported grid regions."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from retime_analysis_to_downbeats import active_event_at, retime_payload
from validate_analysis import audit_analysis


class RetimeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [(1., 1), (2., 2), (3., 3), (4., 4), (5., 1), (6., 2), (7., 3), (8., 4)]
        self.payload = {"duration": 10., "beat_source": "old_source", "events": [
            {"start": 0., "end": 1.125, "chord": "C", "bar": 7, "beat_start": 2,
             "unit": "manual", "manual_corrected": True, "annotation": {"note": "keep me"}},
            {"start": 1.125, "end": 1.25, "chord": "Cmaj7", "confidence": .6},
            {"start": 1.25, "end": 4.25, "chord": "F", "bass_root": "A"},
            {"start": 4.25, "end": 9.75, "chord": "G7", "reviewed": True},
        ], "structure": {"sections": [{"start": .5, "end": 7., "start_bar": 4, "end_bar": 8,
                                        "role": "chorus", "manual": True}]}}

    def test_short_offbeat_events_and_metadata_survive(self):
        before = deepcopy(self.payload)
        output = retime_payload(self.payload, self.rows, "4/4")
        self.assertEqual(self.payload, before)
        self.assertEqual([(e["start"], e["end"], e["chord"]) for e in output["events"]],
                         [(e["start"], e["end"], e["chord"]) for e in before["events"]])
        self.assertEqual(output["events"][0]["annotation"], {"note": "keep me"})
        self.assertTrue(output["events"][0]["manual_corrected"])
        self.assertEqual(output["events"][0]["retime_original_grid"]["bar"], 7)
        tiny = [span for bar in output["bars"] for span in bar["spans"] if span["chord"] == "Cmaj7"]
        self.assertEqual([(span["start"], span["end"]) for span in tiny], [(1.125, 1.25)])
        self.assertEqual(tiny[0]["beat_start"], 1.125)
        self.assertTrue(audit_analysis(output)["valid"])

    def test_uncovered_head_and_tail_have_no_fabricated_beats(self):
        output = retime_payload(self.payload, self.rows, "4/4")
        uncovered = [bar for bar in output["bars"] if bar["grid_status"] == "uncovered"]
        self.assertEqual([(bar["start"], bar["end"]) for bar in uncovered], [(0., 1.), (8., 10.)])
        self.assertTrue(all(bar["beats"] == [] for bar in uncovered))
        self.assertEqual(uncovered[0]["spans"][0]["chord"], "C")
        self.assertEqual(uncovered[-1]["spans"][0]["chord"], "G7")
        self.assertEqual(output["retime_provenance"]["observed_markers"][-1], {"time": 8., "beat": 4})
        self.assertEqual(len({bar["bar"] for bar in output["bars"]}), len(output["bars"]))

    def test_missing_annotation_stays_missing(self):
        payload = {"duration": 5., "events": [{"start": 1., "end": 2., "chord": "C"},
                                                {"start": 4., "end": 5., "chord": "G"}]}
        self.assertIsNone(active_event_at(payload["events"], 0))
        self.assertIsNone(active_event_at(payload["events"], 3))
        self.assertIsNone(active_event_at(payload["events"], 5))
        output = retime_payload(payload, [(0., 1), (1., 2), (2., 3), (3., 4), (4., 1), (5., 2)], "4/4")
        beats = [beat for bar in output["bars"] for beat in bar["beats"]]
        self.assertEqual(beats[2]["chord"], "X")
        self.assertEqual(beats[2]["spans"], [])
        self.assertEqual(len(output["events"]), 2)

    def test_bar_crossing_is_a_view_not_a_new_event(self):
        output = retime_payload(self.payload, self.rows, "4/4")
        cross = output["events"][-1]
        self.assertEqual((cross["start"], cross["end"]), (4.25, 9.75))
        self.assertEqual([(part["start"], part["end"]) for part in cross["grid_placement"]],
                         [(4.25, 5.), (5., 8.), (8., 9.75)])
        section = output["structure"]["sections"][0]
        self.assertEqual((section["start_bar"], section["end_bar"]), (0, 2))
        self.assertTrue(section["manual"])

    def test_provenance_and_confidence_not_invented(self):
        output = retime_payload(self.payload, self.rows, "4/4")
        self.assertEqual(output["beat_source"], "external")
        self.assertEqual(output["meter_confidence"], 0.)
        self.assertEqual(output["meter_confidence_status"], "not_supplied")
        known = retime_payload(self.payload, self.rows, "4/4", beat_source="manual", meter_confidence=.7)
        self.assertEqual(known["beat_source"], "manual")
        self.assertEqual(known["meter_confidence"], .7)

    def test_invalid_grid_rejected_and_missing_markers_reported(self):
        for rows in ([(1., 1), (1., 2)], [(-1., 1), (2., 2)], [(1., 1), (12., 2)], [(1., 1), (2., 5)],
                     [(1., 1), (2., 1.5)]):
            with self.assertRaises(ValueError):
                retime_payload(self.payload, rows, "4/4")
        output = retime_payload(self.payload, [(1., 1), (3., 3), (4., 4)], "4/4")
        self.assertEqual(len(output["retime_provenance"]["beat_sequence_gaps"]), 1)
        self.assertEqual(output["bars"][1]["grid_status"], "incomplete")
        self.assertEqual([beat["beat"] for beat in output["bars"][1]["beats"]], [1, 3])

    def test_conflicting_source_events_are_preserved_and_reported(self):
        payload = deepcopy(self.payload)
        payload["events"][1]["start"] = 1.0
        output = retime_payload(payload, self.rows, "4/4")
        self.assertEqual(output["events"][1]["start"], 1.0)
        self.assertAlmostEqual(output["retime_provenance"]["source_event_overlap_seconds"], .125)
        self.assertFalse(audit_analysis(output)["valid"])

    def test_uncovered_is_a_warning_and_missing_span_is_still_an_error(self):
        output = retime_payload(self.payload, self.rows, "4/4")
        report = audit_analysis(output)
        self.assertTrue(report["valid"])
        self.assertEqual(sum(issue["kind"] == "uncovered_grid_region" for issue in report["issues"]), 2)
        for bar in output["bars"]:
            bar["spans"] = []
        report = audit_analysis(output)
        self.assertFalse(report["valid"])
        self.assertIn("events_bars_disagree", [issue["kind"] for issue in report["issues"]])


if __name__ == "__main__":
    unittest.main()
