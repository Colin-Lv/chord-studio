"""External labels keep pitch content, explicit bass and unknown status end to end.

These fixtures are authored music examples, not model outputs. Expected pitch
sets and absolute bass notes are independent of the normalizer's spelling.
Only the new tests are needed: no acoustic model, waveform or browser is run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import mir_eval
import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import chord_studio as studio
import downbeat_external_fusion as fusion
from evaluate_chords import harte_chord, read_prediction, read_reference


# label, root pitch class, absolute chord-tone pitch classes, absolute bass.
CASES = [
    ("N", None, None, None),
    ("X", None, None, None),
    ("C:maj/E", 0, {0, 4, 7}, 4),
    ("G:maj/3", 7, {7, 11, 2}, 11),
    ("D:min7/b3", 2, {2, 5, 9, 0}, 5),
    ("Bb:7/3", 10, {10, 2, 5, 8}, 2),
    ("C:maj7/7", 0, {0, 4, 7, 11}, 11),
    ("C:9/b7", 0, {0, 2, 4, 7, 10}, 10),
    ("D:sus4/5", 2, {2, 7, 9}, 9),
    ("C:minmaj7/b3", 0, {0, 3, 7, 11}, 3),
    ("C:hdim7/b5", 0, {0, 3, 6, 10}, 6),
    ("C:maj(9)/3", 0, {0, 2, 4, 7}, 4),
    ("C:min(*5)/b3", 0, {0, 3}, 3),
    ("C:min/5", 0, {0, 3, 7}, 7),
    ("C:maj", 0, {0, 4, 7}, 0),
    ("C:7", 0, {0, 4, 7, 10}, 0),
]


def assert_pitch_semantics(label, case):
    original, root, members, bass = case
    if original in {"N", "X"}:
        assert label == original, "Unknown X must never become no-chord N"
        return
    normalized = harte_chord(label)
    assert normalized not in {"N", "X"}, f"Valid chord became {normalized}: {label}"
    actual_root, bitmap, relative_bass = mir_eval.chord.encode(normalized, reduce_extended_chords=True)
    assert actual_root == root
    assert {(root + interval) % 12 for interval, active in enumerate(bitmap) if active} == members
    assert (actual_root + relative_bass) % 12 == bass


def source_rows():
    return [dict(start=index / 4, end=(index + 1) / 4, chord=case[0],
                 confidence=.01 if case[0] == "X" else 1.)
            for index, case in enumerate(CASES)]


def write_source(path, rows):
    if path.suffix == ".json":
        path.write_text(json.dumps(dict(events=rows)), encoding="utf-8")
    elif path.suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["start", "end", "chord", "confidence"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text("".join(f"{r['start']} {r['end']} {r['chord']} {r['confidence']}\n" for r in rows), encoding="utf-8")


def frontend_exports(payload, edit_labels=None, key="C"):
    """Run the real parser/field setters/export functions in Node without DOM/audio.

    UI wiring is outside this regression. Function bodies and pitch tables come
    directly from the shipped asset; no parser implementation is copied here.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to exercise shipped frontend functions")
    source = (APP / "studio_assets/studio.js").read_text(encoding="utf-8")
    tables = source[source.index("    const ROOT_TO_PC"):source.index("    let events")]
    names = ["cloneData", "parseChord", "rootToRoman", "chordToRoman", "parseChordRoot", "noteName",
             "degreeTokenToChord", "degreeToChord", "parseEditableChord", "degreeTokenToBass", "degreeRootName",
             "applyChordFields", "exportEventsAsLab", "exportAnalysisJson", "baseExportName", "labelFor"]
    functions = []
    for name in names:
        match = re.search(rf"(?ms)^    function {name}\(.*?(?=^    function |\Z)", source)
        assert match, f"Frontend function moved or missing: {name}"
        functions.append(match.group())
    harness = """
const input = JSON.parse(require('fs').readFileSync(0,'utf8'));
const DATA = input.payload;
let events = DATA.events;
let displayMode = 'name';
function editingKey() { return input.key; }
function editedEventCount() { return events.filter(row=>row.edited).length; }
function autosaveDraft() {}
const downloads = [];
function downloadText(filename,text,mimeType) { downloads.push({filename,text,mimeType}); }
""" + tables + "\n".join(functions) + """
const parsed = input.edit_labels ? input.edit_labels.map(parseEditableChord) : [];
if (parsed.some(item=>item===null)) throw new Error('Valid external label rejected: '+JSON.stringify(parsed));
if (input.edit_labels) events.forEach((item,index)=>applyChordFields(item,parsed[index]));
exportEventsAsLab(); exportAnalysisJson();
console.log(JSON.stringify({parsed,downloads,display:events.map(labelFor),
  roman:events.map(item=>chordToRoman(item.chord)),
  invalidBass:['C/N','C/','C/E/G'].map(x=>parseEditableChord(x)?.chordDisplay||null),
  legacyDegrees:['II7','ii7','Imaj7','6m7','IV','G/3','C/E/G'].map(x=>parseEditableChord(x)?.chordDisplay||null)}));
"""
    result = subprocess.run([node, "-e", harness], input=json.dumps(dict(payload=payload, edit_labels=edit_labels, key=key)),
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("extension", [".lab", ".csv", ".json"])
def test_external_source_through_fusion_html_and_frontend_exports(tmp_path, monkeypatch, extension):
    rows = source_rows()
    source = tmp_path / ("external" + extension)
    write_source(source, rows)
    hints = studio.read_external_chord_hints(source, "sharps")
    assert [(hint.start, hint.end) for hint in hints] == [(row["start"], row["end"]) for row in rows]
    for hint, case in zip(hints, CASES):
        assert_pitch_semantics(hint.chord, case)
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(duration=4., key="C", source=str(tmp_path / "not_decoded.wav"),
        audio_file="not_decoded.wav", bars=[dict(bar=1, start=0., end=4., spans=[],
        beats=[dict(beat=i + 1, start=float(i), end=float(i+1), chord="N", confidence=0.) for i in range(4)])])), encoding="utf-8")
    target = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["fusion", str(base), str(base), str(source), "-o", str(target),
        "--beat-grid-from-base", "--meter", "4/4", "--harmonic-unit", "external", "--base-weight", "0", "--no-html"])
    assert fusion.main() == 0
    prediction = tmp_path / "result_analysis.json"
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    assert [(r.start, r.end) for r in read_prediction(prediction)] == [(row["start"], row["end"]) for row in rows]
    assert len(payload["events"]) == len(CASES)
    for item, case in zip(payload["events"], CASES):
        assert_pitch_semantics(item["chord"], case)
        assert_pitch_semantics(item["chord_display"], case)
    html = tmp_path / "result.html"
    studio.write_html(html, payload)
    content = html.read_text(encoding="utf-8")
    prefix = re.search(r"const SOURCE_DATA\s*=\s*", content)
    embedded, _ = json.JSONDecoder().raw_decode(content[prefix.end():])
    exported = frontend_exports(embedded)
    exported_lab = tmp_path / "browser.lab"
    exported_lab.write_text(exported["downloads"][0]["text"], encoding="utf-8")
    lab_rows = read_reference(exported_lab)
    exported_json = json.loads(exported["downloads"][1]["text"])
    assert len(lab_rows) == len(CASES)
    for lab_row, event, visible, case in zip(lab_rows, exported_json["events"], exported["display"], CASES):
        for label in (lab_row.chord, event["chord"], event["chord_display"], visible):
            assert_pitch_semantics(label, case)


def test_manual_frontend_roundtrip_keeps_x_and_explicit_canonical_bass():
    payload = dict(audio_file="fixture.wav", events=source_rows())
    exported = frontend_exports(payload, [case[0] for case in CASES])
    rows = json.loads(exported["downloads"][1]["text"])["events"]
    for row, case in zip(rows, CASES):
        assert_pitch_semantics(row["chord"], case)
        assert_pitch_semantics(row["chord_display"], case)
        if "/" in case[0]:
            assert "/" in row["chord"], "Manual bass must survive canonical JSON, not only its display field"
            assert row["bass_confidence"] == 1.
    assert exported["legacyDegrees"] == ["D7", "Dm7", "Cmaj7", "Am7", "F", "G/E", None]
    assert exported["roman"][3] == "V/VII"  # G:maj/3 is G/B, regardless of C global key.
    assert exported["roman"][1] == "X"
    assert exported["invalidBass"] == [None, None, None]


@pytest.mark.parametrize("label", ["CM7/E", "C:M7/E"])
def test_frontend_major_seventh_alias_does_not_become_minor(label):
    result = frontend_exports(dict(audio_file="alias.wav", events=[dict(start=0, end=1, chord=label)]), [label], key="G")
    row = json.loads(result["downloads"][1]["text"])["events"][0]
    assert_pitch_semantics(row["chord"], ("C:maj7/3", 0, {0, 4, 7, 11}, 4))
    assert row["roman_display"] == "IVmaj7/VI"


@pytest.mark.parametrize("label", ["CM", "C:M"])
def test_frontend_bare_major_alias_does_not_become_minor(label):
    result = frontend_exports(dict(audio_file="alias.wav", events=[dict(start=0, end=1, chord=label)]), [label], key="G")
    row = json.loads(result["downloads"][1]["text"])["events"][0]
    assert row["chord"] == row["chord_display"] == "C"
    assert_pitch_semantics(row["chord"], ("C:maj", 0, {0, 4, 7}, 0))
    assert row["roman_display"] == "IV"


def test_import_canonical_field_does_not_promote_old_inferred_display_bass(tmp_path):
    source = tmp_path / "old.json"
    source.write_text(json.dumps(dict(events=[dict(start=0, end=1, chord="C", chord_display="C/E",
        bass_root="E", bass_confidence=.99, confidence=1.)])), encoding="utf-8")
    assert studio.read_external_chord_hints(source, "sharps")[0].chord == "C"


@pytest.mark.parametrize("unknown", [{"X": .01}, {"X": 0., "C": 0.}])
def test_unknown_low_support_does_not_become_no_chord_or_neighboring_major(unknown):
    assert fusion.smooth_chords([{"C": 1.}, unknown, {"C": 1.}], "C") == ["C", "X", "C"]


@pytest.mark.parametrize("raw,expected", [("N", "N"), ("X", "X"), ("not_a_chord", "X"), ("", "X")])
def test_invalid_or_unknown_external_label_never_invents_silence(raw, expected):
    assert studio.normalize_external_chord_label(raw, "sharps") == expected


@pytest.mark.parametrize("spelling", ["sharps", "flats"])
def test_equivalent_absolute_and_relative_bass_with_non_c_root(spelling):
    labels = [studio.normalize_external_chord_label(label, spelling) for label in ("Eb:min/b3", "Ebm/Gb")]
    assert labels[0] == labels[1]
    for label in labels:
        assert_pitch_semantics(label, ("Eb:min/b3", 3, {3, 6, 10}, 6))
