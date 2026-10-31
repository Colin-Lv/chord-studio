import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import build_aligned_manifest_from_version_audit as align


def test_alignment_search_cannot_improve_by_cropping_reference_duration():
    from scan_manifest_audio_versions import shifted_reference
    from evaluate_chords import TimedChord, evaluate
    reference = [TimedChord(0, 10, "C")]
    prediction = [TimedChord(0, 1, "C")]
    shifted = shifted_reference(reference, -9)
    score = evaluate(shifted, prediction)
    assert score["evaluated_seconds"] == 10
    assert score["triad_accuracy"] == pytest.approx(.1)


@pytest.mark.parametrize("shift", [2., -1.])
def test_generated_annotation_uses_audited_direction_and_rebases_paths(tmp_path, monkeypatch, shift):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "song.lab").write_text("3 5 C:maj\n")
    (raw / "song.beats").write_text("3 1\n4 2\n")
    manifest = raw / "manifest.csv"
    fields = ["id", "audio", "prediction", "chord_reference", "beat_reference"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerow(dict(id="song", audio="song.wav", prediction="old.json", chord_reference="song.lab", beat_reference="song.beats"))
    audit = tmp_path / "audit.csv"
    audit.write_text(f"id,best_triad,status,best_time_shift\nsong,0.8,strong_match,{shift}\n")
    accepted, rejected = tmp_path / "outputs/ok.csv", tmp_path / "rejected/no.csv"
    monkeypatch.setattr(sys, "argv", ["align", str(manifest), str(audit), "--output-accepted", str(accepted),
                                      "--output-rejected", str(rejected), "--aligned-dir", str(tmp_path / "aligned")])
    assert align.main() == 0
    row = next(csv.DictReader(accepted.open(encoding="utf-8-sig")))
    chord_path = (accepted.parent / row["chord_reference"]).resolve()
    beat_path = (accepted.parent / row["beat_reference"]).resolve()
    assert float(chord_path.read_text().split()[0]) == 3 + shift
    assert float(beat_path.read_text().split()[0]) == 3 + shift
    assert (accepted.parent / row["audio"]).resolve() == raw / "song.wav"
    assert (accepted.parent / row["original_chord_reference"]).resolve() == raw / "song.lab"
    assert row["alignment_method"] == "prediction_optimized_time_shift"
    assert (raw / "song.lab").read_text() == "3 5 C:maj\n"
