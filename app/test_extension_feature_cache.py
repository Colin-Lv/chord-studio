import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import train_chord_extensions as training


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root / "Song.wav"
        self.audio.write_bytes(b"original")
        (self.root / "song.lab").write_text("0 1 C:maj\n", encoding="utf-8")
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "audio", "chord_reference", "split"])
            writer.writeheader()
            writer.writerow(dict(id="song", audio="Song.wav", chord_reference="song.lab", split="train"))
        self.cache = self.root / "cache"

    def read(self):
        return training.read_songs(self.manifest, self.root, self.cache)

    def test_same_size_and_mtime_audio_replacement_invalidates_cache(self):
        with patch.object(training.studio, "decode_audio", return_value=np.zeros(1)), patch.object(
                training, "features_for_rows", side_effect=[(np.zeros((1, 87)), []), (np.ones((1, 87)), [])]) as extract:
            first = self.read()[0]
            self.read()
            stat = self.audio.stat()
            self.audio.write_bytes(b"modified")
            os.utime(self.audio, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(self.audio.stat().st_size, stat.st_size)
            self.assertEqual(self.audio.stat().st_mtime_ns, stat.st_mtime_ns)
            second = self.read()[0]
        self.assertEqual(extract.call_count, 2)
        np.testing.assert_array_equal(first["features"], np.zeros((1, 87)))
        np.testing.assert_array_equal(second["features"], np.ones((1, 87)))
        self.assertEqual(second["source_audio"], str(self.audio.resolve()))
        np.testing.assert_array_equal(first["targets"], second["targets"])

    def test_bad_cache_shape_or_nonfinite_is_rejected(self):
        with patch.object(training.studio, "decode_audio", return_value=np.zeros(1)), patch.object(
                training, "features_for_rows", return_value=(np.zeros((1, 87)), [])) as extract:
            self.read()
            cache = next(self.cache.glob("*.npz"))
            for bad in (np.zeros((2, 87)), np.zeros((1, 148)), np.full((1, 87), np.nan)):
                np.savez_compressed(cache, features=bad)
                with self.assertRaisesRegex(ValueError, "Invalid extension features"):
                    self.read()
        self.assertEqual(extract.call_count, 1)

    def test_implementation_change_invalidates_cache(self):
        with patch.object(training.studio, "decode_audio", return_value=np.zeros(1)), patch.object(
                training, "features_for_rows", return_value=(np.zeros((1, 87)), [])) as extract:
            with patch.object(training, "_feature_implementation_sha256", return_value="implementation-a"):
                self.read()
                self.read()
            with patch.object(training, "_feature_implementation_sha256", return_value="implementation-b"):
                self.read()
        self.assertEqual(extract.call_count, 2)


if __name__ == "__main__":
    unittest.main()
