import hashlib
import tempfile
import unittest
from pathlib import Path
import zipfile

from import_guitarset import extract_verified, select_chord_annotations, split_for, write_lab


class GuitarSetImportTests(unittest.TestCase):
    def test_performed_selected_by_provenance_not_namespace_order(self):
        instructed = {'namespace': 'chord', 'annotation_metadata': {}, 'data': []}
        performed = {'namespace': 'chord', 'annotation_metadata': {'annotation_rules': 'Chord sheet-informed symbolic chord transcription'}, 'data': []}
        score, played = select_chord_annotations({'annotations': [performed, instructed]})
        self.assertEqual(score['annotation_index'], 1)
        self.assertEqual(played['annotation_index'], 0)

    def test_ambiguous_derivative_annotation_is_rejected(self):
        annotation = {'namespace': 'chord', 'annotation_metadata': {'annotation_rules': 'sheet-informed'}}
        with self.assertRaises(ValueError):
            select_chord_annotations({'annotations': [annotation, annotation]})

    def test_holdout_axes_are_disjoint_without_implying_other_axis_is(self):
        for player in ['00', '01', '02', '03', '04', '05']:
            self.assertEqual(len({split_for(player, family, 'performer') for family in '123'}), 1)
        for family in '123':
            self.assertEqual(len({split_for(player, family, 'progression') for player in ['00', '04', '05']}), 1)
        self.assertEqual(split_for('04', '1', 'performer'), 'validation')
        self.assertEqual(split_for('04', '1', 'progression'), 'train')

    def test_extracted_same_size_tampering_is_repaired_from_verified_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive, destination = folder / 'annotations.zip', folder / 'extracted'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('../sample.jams', 'safe')
            expected = hashlib.md5(archive.read_bytes()).hexdigest()
            extract_verified(archive, destination, '.jams', expected)
            self.assertFalse((folder / 'sample.jams').exists())
            (destination / 'sample.jams').write_text('oops')
            extract_verified(archive, destination, '.jams', expected)
            self.assertEqual((destination / 'sample.jams').read_text(), 'safe')
            with self.assertRaises(ValueError):
                extract_verified(archive, destination, '.jams', '0' * 32)

    def test_chord_labels_and_original_times_are_not_quantized(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'performed.lab'
            annotation = {'data': [{'time': .012345678901, 'duration': .876543210987, 'value': 'C#:maj6(#9)/b3'}]}
            write_lab(destination, annotation, 1)
            start, end, label = destination.read_text().split()
            self.assertEqual(float(start), annotation['data'][0]['time'])
            self.assertEqual(label, 'C#:maj6(#9)/b3')
            self.assertAlmostEqual(float(end), .888888889888)

    def test_overlapping_chords_fail_instead_of_silent_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            annotation = {'data': [{'time': 0, 'duration': 1, 'value': 'C:maj'}, {'time': .5, 'duration': 1, 'value': 'D:min'}]}
            with self.assertRaises(ValueError):
                write_lab(Path(folder) / 'test.lab', annotation, 2)


if __name__ == '__main__':
    unittest.main()
