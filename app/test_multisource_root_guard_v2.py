import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from evaluate_chords import TimedChord
from multisource_root_guard import preserve_raw_root
from multisource_root_guard_v2 import preserve_root_allowed_first, frozen_v1_full_head_from_scores


def case(labels, utilities, raw="C"):
    events = [TimedChord(.125, 2.375, raw)]
    groups = [dict(start=0, end=len(labels), candidates=[dict(label=label) for label in labels])]
    return events, groups, utilities


class AllowedFirstPolicyTests(unittest.TestCase):
    def test_disallowed_winner_does_not_hide_allowed_extension(self):
        raw, groups, values = case(["C", "G", "C7"], [.60, .95, .90])
        full = frozen_v1_full_head_from_scores(raw, groups, values)
        old, _ = preserve_raw_root(raw, full)
        new, details = preserve_root_allowed_first(raw, groups, values)
        self.assertEqual(old[0].chord, "C")
        self.assertEqual(new[0].chord, "C7")
        self.assertEqual(details["decisions"][0]["allowed_indices"], [0, 2])

    def test_n_x_and_invalid_raw_are_unchanged(self):
        for label in ("N", "X", "unrecognized"):
            raw, groups, values = case([label, "C", "N", "X"], [.1, .99, .98, .97], raw=label)
            predicted, _ = preserve_root_allowed_first(raw, groups, values)
            self.assertEqual(predicted, raw)

    def test_raw_only_and_empty_candidates(self):
        for labels, values in ((["C"], [.6]), ([], [])):
            raw, groups, values = case(labels, values)
            self.assertEqual(preserve_root_allowed_first(raw, groups, values)[0], raw)
        self.assertEqual(preserve_root_allowed_first([], [], [])[0], [])

    def test_raw_ties_exact_margin_and_candidate_order_ties(self):
        for alternative in (.6, .62, .6200000000005):
            raw, groups, values = case(["C", "C7"], [.6, alternative])
            self.assertEqual(preserve_root_allowed_first(raw, groups, values)[0], raw)
        raw, groups, values = case(["C", "Cmaj7", "C7"], [.6, .9, .9])
        self.assertEqual(preserve_root_allowed_first(raw, groups, values)[0][0].chord, "Cmaj7")

    def test_same_root_quality_and_bass_changes_are_permitted(self):
        raw, groups, values = case(["C", "N", "X", "Cm/Eb"], [.1, .99, .98, .9])
        self.assertEqual(preserve_root_allowed_first(raw, groups, values)[0][0].chord, "Cm/Eb")

    def test_gaps_coverage_and_boundaries_do_not_change(self):
        raw = [TimedChord(.123456789123, 1.3, "C"), TimedChord(2.75, 4.1, "D")]
        groups = [dict(start=0, end=2, candidates=[dict(label="C"), dict(label="C7")]),
                  dict(start=2, end=4, candidates=[dict(label="D"), dict(label="Dmaj7")])]
        predicted, _ = preserve_root_allowed_first(raw, groups, [.5, .9, .5, .9])
        self.assertEqual([(r.start, r.end) for r in predicted], [(r.start, r.end) for r in raw])
        self.assertEqual(sum(r.end-r.start for r in predicted), sum(r.end-r.start for r in raw))

    def test_invalid_scores_groups_and_timeline_fail(self):
        raw, groups, _ = case(["C", "C7"], [0, 0])
        for values in ([.5, np.nan], [[.5, .9]], [.5], [.5, np.inf]):
            with self.assertRaises(ValueError):
                preserve_root_allowed_first(raw, groups, values)
        with self.assertRaises(ValueError):
            preserve_root_allowed_first(*case(["D", "C7"], [.5, .9]))
        with self.assertRaises(ValueError):
            preserve_root_allowed_first(raw, groups, [.5, .9], margin=float("nan"))

    def test_source_identity_paths_cannot_shadow_or_hide_conflicting_hashes(self):
        from evaluate_root_guard_v2 import source_identity
        from train_multisource_seventh_gate import signature
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"audio identity only; no decoding")
            lab = Path(directory) / "output.lab"
            lab.write_text("0 1 C\n", encoding="utf-8")
            sidecar = lab.with_suffix(".provenance.json")
            data = dict(audio=str(audio), source_audio=signature(audio),
                        output_lab=signature(lab), audio_offset=0)
            sidecar.write_text(json.dumps(data), encoding="utf-8")
            result = source_identity(lab, signature(audio))
            self.assertEqual(result["generation_identity"],
                             "audio_and_lab_bytes_verified_from_existing_sidecar")
            data["audio"] = dict(sha256="conflicting alias")
            sidecar.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audio SHA mismatch"):
                source_identity(lab, signature(audio))


if __name__ == "__main__":
    unittest.main()
