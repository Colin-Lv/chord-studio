"""Small mechanism checks; no recordings, training or model evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_chords import TimedChord
from apply_multisource_candidates import lab_provenance, reject_input_overwrite
from evaluate_multisource_candidate_reliability import (read_evaluation_reference,
    reject_training_overlap, same_interval_support, validate_resume_protocol, main)
from train_multisource_seventh_gate import signature


class DeploymentGuards(unittest.TestCase):
    def test_studio_reference_keeps_display_labels_and_absolute_times(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.json"
            path.write_text(json.dumps({"title": "Corrected example", "events": [
                {"start": 1.25, "end": 3.5, "chord": "C", "chord_display": "Cmaj7/E"},
                {"start": 3.5, "end": 4, "chord": "N"}]}), encoding="utf-8")
            rows = read_evaluation_reference(path)
            self.assertEqual(rows, [TimedChord(1.25, 3.5, "Cmaj7/E"), TimedChord(3.5, 4, "N")])
            path.write_text('{"events": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain timed chords"):
                read_evaluation_reference(path)

    def test_resume_rejects_changed_frozen_model(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            protocol = {"model": {"sha256": "model"}, "manifest": {"sha256": "manifest"},
                "model_contract": {"sha256": "contract"}, "protocol": "frozen",
                "frozen_method": "hgb", "frozen_margin": .02}
            (directory / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            (directory / "report.json").write_text("{}", encoding="utf-8")
            self.assertIn("protocol", validate_resume_protocol(directory, protocol))
            protocol["model"] = {"sha256": "different"}
            with self.assertRaisesRegex(ValueError, "frozen model"):
                validate_resume_protocol(directory, protocol)
        self.assertTrue(same_interval_support([TimedChord(.123456789, 1, "C")],
                                             [TimedChord(.1234567891, 1, "D")]))
        self.assertFalse(same_interval_support([TimedChord(0, 1, "C")], [TimedChord(.001, 1, "C")]))

    def test_renamed_training_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "new-name.wav"
            audio.write_bytes(b"same training recording bytes")
            artifact = {"training_song_ids": ["old-name"], "training_audio_paths": [str(Path(folder) / "old.wav")]}
            contract = {"training_audio": [{"sha256": signature(audio)["sha256"]}]}
            with self.assertRaisesRegex(ValueError, "overlaps fitted training"):
                reject_training_overlap([{"id": "new-name", "audio": audio}], artifact, contract)

    def test_alternate_sidecar_verifies_audio_output_and_full_window(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            audio, lab = directory / "source.wav", directory / "large.lab"
            audio.write_bytes(b"audio")
            lab.write_text("0 2 C\n", encoding="utf-8")
            sidecar = lab.with_suffix(".provenance.json")
            record = dict(source_audio=signature(audio), output=signature(lab),
                          requested_duration_seconds=None, audio_seconds=2)
            sidecar.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(lab_provenance(lab, signature(audio), "largevocab", 2)["status"], "audio_and_lab_sha_verified")
            with self.assertRaisesRegex(ValueError, "duration differs"):
                lab_provenance(lab, signature(audio), "largevocab", 3)
            record["requested_duration_seconds"] = 2
            sidecar.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "limited-duration"):
                lab_provenance(lab, signature(audio), "largevocab", 2)
            with self.assertRaisesRegex(ValueError, "overwrite an input"):
                reject_input_overwrite([lab], [lab])

    def test_failed_reference_returns_nonzero_and_retains_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            model, manifest = directory / "model.joblib", directory / "manifest.csv"
            model.write_bytes(b"fixture")
            model.with_suffix(".contract.json").write_text("{}", encoding="utf-8")
            manifest.write_text("fixture", encoding="utf-8")
            audio = directory / "source.wav"
            audio.write_bytes(b"audio")
            paths = []
            for name in ("raw", "chordino", "largevocab"):
                path = directory / (name + ".lab")
                path.write_text("0 2 C\n", encoding="utf-8")
                paths.append(path)
            reference = directory / "reference.json"
            reference.write_text('{"events": []}', encoding="utf-8")
            row = dict(id="fixture", audio=audio, upstream=paths[0], chordino=paths[1], largevocab=paths[2],
                       reference=reference, domain="synthetic", reference_status="mechanism_only", audio_identity=signature(audio))
            artifact = {"method": "hgb_regression", "margin": .02, "model": None}
            evidence = type("Evidence", (), {"metadata": {"duration_seconds": 2}})()
            module = "evaluate_multisource_candidate_reliability."
            with patch(module + "load_frozen_artifact", return_value=(artifact, {})), \
                 patch(module + "manifest_items", return_value=[row]), \
                 patch(module + "reject_training_overlap"), \
                 patch(module + "load_existing_evidence", return_value=(evidence, {})), \
                 patch(module + "validate_note_contract"), \
                 patch(module + "automatic_candidate_features", return_value=(None, None)), \
                 patch(module + "predict_song", return_value=[TimedChord(0, 2, "C")]):
                status = main(["--manifest", str(manifest), "--model", str(model), "--output", str(directory / "out")])
            self.assertEqual(status, 2)
            self.assertTrue((directory / "out/predictions/fixture.lab").exists())
            self.assertEqual(json.loads((directory / "out/report.json").read_text())["failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
