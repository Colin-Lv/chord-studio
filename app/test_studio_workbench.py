"""Browser regression tests for standalone, offline Chord Studio exports.

Run: python -m unittest discover -s app -p test_studio_workbench.py -v
Browser tests use Playwright and a locally installed Edge/Chromium. They skip
when neither is available; no server, network, or downloaded songs are needed.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from chord_studio import write_html


def fixture_payload(audio: Path) -> dict:
    bars, events = [], []
    for index, chord in enumerate(["C", "Am", "F", "G", "C", "G"]):
        start = float(index * 2)
        beats = [dict(beat=beat + 1, start=start + beat * .5, end=start + (beat + 1) * .5,
                      chord=chord, chord_display=chord, roman="I", roman_display="I",
                      confidence=.3 if index == 2 else .8) for beat in range(4)]
        cuts = [start, start + .25, start + .75, start + 2] if index == 0 else [start, start + 2]
        spans = [dict(bar=index + 1, start=a, end=b, beat_start=(a-start)*2+1,
                      beat_end=(b-start)*2, chord=chord, chord_display=chord,
                      roman="I", roman_display="I", confidence=.3 if index == 2 else .8,
                      unit="subbeat") for a, b in zip(cuts[:-1], cuts[1:])]
        bars.append(dict(bar=index + 1, start=start, end=start + 2, beats=beats, spans=spans))
        events.extend(copy.deepcopy(spans))
    return dict(audio_file=audio.name, source=str(audio), audio_offset=1.5, duration=12,
                key="C", mode="major", meter="4/4", beats_per_bar=4, bpm=120,
                bars=bars, events=events, display="name", structure={"sections": [
                    dict(id="section-1", label="A", group="A", name="段落 A", role="unknown",
                         start=0, end=6, start_bar=1, end_bar=3, confidence=.8, source="automatic"),
                    dict(id="section-2", label="B", group="B", name="段落 B", role="unknown",
                         start=6, end=12, start_bar=4, end_bar=6, confidence=.7, source="automatic"),
                ], "loops": [], "manual_overrides": []})


class ExportSafetyTest(unittest.TestCase):
    def test_audio_filename_is_url_encoded(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "曲名 # 100%.wav"
            audio.write_bytes(b"")
            path = Path(directory) / "studio.html"
            write_html(path, fixture_payload(audio))
            text = path.read_text(encoding="utf-8")
            self.assertIn("%23%20100%25.wav", text)
            self.assertIn("%E6%9B%B2%E5%90%8D", text)

    def test_script_terminator_in_payload_is_escaped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "studio.html"
            payload = fixture_payload(Path(directory) / "sample.wav")
            payload["audio_file"] = '</script><script>window.injected=true</script>'
            write_html(path, payload)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("</script>"), 1)
            self.assertIn(r"\u003c/script\u003e", text)
            self.assertNotIn("__STUDIO_", text)


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Playwright is not installed")
class WorkbenchBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.playwright = sync_playwright().start()
        candidates = [os.environ.get("STUDIO_BROWSER_PATH", ""),
                      r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                      r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
        executable = next((path for path in candidates if path and Path(path).exists()), None)
        try:
            cls.browser = cls.playwright.chromium.launch(
                **({"executable_path": executable} if executable else {}),
                headless=True, args=["--allow-file-access-from-files", "--autoplay-policy=no-user-gesture-required"])
        except Exception as exc:
            cls.playwright.stop()
            raise unittest.SkipTest(f"No browser available: {exc}")
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        audio = cls.directory / "fixture.wav"
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"".join(struct.pack("<h", int(1000 * math.sin(index * 2 * math.pi * 220 / 8000))) for index in range(15 * 8000)))
        cls.payload = fixture_payload(audio)
        cls.html = cls.directory / "studio.html"
        write_html(cls.html, cls.payload)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.html.as_uri())
        self.page.wait_for_function("document.querySelectorAll('.bar').length === 6 && audio.readyState >= 1")

    def tearDown(self):
        self.assertEqual(self.errors, [], f"Browser errors: {self.errors}")
        self.context.close()

    def evaluate(self, script):
        return self.page.evaluate(script)

    def test_render_accessibility_and_mobile_layout(self):
        self.assertEqual(self.page.locator(".section-nav").count(), 2)
        self.assertEqual(self.page.locator(".bar").count(), 6)
        self.assertEqual(self.page.locator(".span-cell.selected").count(), 1)
        self.page.get_by_role("button", name="切换明暗主题").click()
        self.assertEqual(self.evaluate("document.documentElement.dataset.theme"), "dark")
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.assertLessEqual(self.evaluate("document.documentElement.scrollWidth"), 390)
        self.page.locator("#showInspector").click()
        self.page.locator("#tabStructure").click()
        self.assertTrue(self.page.locator("#structurePanel").is_visible())
        self.page.locator("#dismissInspector").click()
        self.assertFalse(self.page.locator("#structurePanel").is_visible())

    def test_select_seek_and_sync_calibration(self):
        self.page.locator('.bar[data-bar="2"] .span-cell').click()
        self.assertTrue(self.evaluate("audio.paused"))
        self.assertAlmostEqual(self.evaluate("audio.currentTime"), 3.5, places=2)
        self.evaluate("sync.value = '250'; sync.dispatchEvent(new Event('input'))")
        self.page.locator('.bar[data-bar="3"] .span-cell').click()
        self.assertAlmostEqual(self.evaluate("audio.currentTime"), 5.25, places=2)
        self.assertEqual(self.page.locator(".span-cell.selected").count(), 1)
        self.assertEqual(self.page.locator("#currentChord").inner_text(), "F")

    def test_chord_edit_undo_redo_and_invalid_input(self):
        self.page.locator('.bar[data-bar="2"] .span-cell').click()
        self.page.locator("#chordInput").fill("Dm7")
        self.page.locator("#saveChord").click()
        self.assertEqual(self.evaluate("events.find(e => e.bar === 2).chord"), "Dm7")
        self.page.locator("#undoEdit").click()
        self.assertEqual(self.evaluate("events.find(e => e.bar === 2).chord"), "Am")
        self.page.locator("#redoEdit").click()
        self.assertEqual(self.evaluate("events.find(e => e.bar === 2).chord"), "Dm7")
        self.page.locator("#chordInput").fill("Cgarbage")
        self.page.locator("#saveChord").click()
        self.assertEqual(self.page.locator("#chordInput").get_attribute("aria-invalid"), "true")
        self.assertEqual(self.evaluate("events.find(e => e.bar === 2).chord"), "Dm7")

    def test_beat_edit_keeps_unrelated_subbeat_boundaries(self):
        self.page.locator('.beat[data-bar="1"][data-beat="1"]').click()
        self.page.locator("#chordInput").fill("Dm")
        self.page.locator("#saveChord").click()
        spans = self.evaluate("DATA.bars[0].spans.map(s => [s.start,s.end,s.chord])")
        self.assertEqual(spans, [[0, .5, "Dm"], [.5, .75, "C"], [.75, 2, "C"]])

    def test_split_uses_selected_range_and_undo_restores(self):
        self.evaluate("selectEvent(1); splitSelectedEvent('half')")
        spans = self.evaluate("DATA.bars[0].spans.map(s => [s.start,s.end])")
        self.assertEqual(spans, [[0,.25],[.25,.5],[.5,.75],[.75,2]])
        self.evaluate("moveHistory(-1)")
        self.assertEqual(self.evaluate("DATA.bars[0].spans.length"), 3)

    def test_boundary_adjustment_keeps_neighbors_contiguous(self):
        self.evaluate("selectEvent(1)")
        self.page.locator("#eventStart").fill("0.3")
        self.page.locator("#eventEnd").fill("0.8")
        self.page.locator("#applyBoundary").click()
        self.assertEqual(self.evaluate("DATA.bars[0].spans.map(s => [s.start,s.end])"), [[0,.3],[.3,.8],[.8,2]])
        self.page.locator("#eventStart").fill("-1")
        self.page.locator("#applyBoundary").click()
        self.assertEqual(self.evaluate("DATA.bars[0].spans[1].start"), .3)

    def test_ab_loop_with_audio_offset_and_nonzero_sync(self):
        self.evaluate("syncMs = 200; setLoop(2, 2.25); seekToAnalysis(2, true)")
        self.page.wait_for_timeout(650)
        state = self.evaluate("({time:analysisTime(),paused:audio.paused,loop:loopRange.enabled})")
        self.assertFalse(state["paused"])
        self.assertTrue(state["loop"])
        self.assertGreaterEqual(state["time"], 1.99)
        self.assertLess(state["time"], 2.3)

    def test_keyboard_and_evidence_filter(self):
        self.page.locator("#reviewOnly").check()
        self.assertEqual(self.page.locator(".bar:visible").count(), 1)
        self.assertEqual(self.page.locator(".bar:visible").get_attribute("data-bar"), "3")
        self.page.locator("#reviewOnly").uncheck()
        self.page.locator("#bars").focus()
        self.page.keyboard.press("Space")
        self.page.wait_for_timeout(100)
        self.assertFalse(self.evaluate("audio.paused"))
        self.page.keyboard.press("Space")
        self.assertTrue(self.evaluate("audio.paused"))
        previous = self.evaluate("selectedEventIndex")
        self.page.keyboard.press("ArrowRight")
        self.assertEqual(self.evaluate("selectedEventIndex"), previous + 1)

    def test_section_edit_export_and_history(self):
        self.page.locator("#tabStructure").click()
        self.page.locator("#sectionName").fill("主歌 1")
        self.page.locator("#sectionRole").select_option("verse")
        self.page.locator("#saveSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections[0].name"), "主歌 1")
        self.assertEqual(self.evaluate("DATA.structure.manual_overrides.length"), 2)
        self.page.locator("#undoEdit").click()
        self.assertEqual(self.evaluate("DATA.structure.sections[0].name"), "段落 A")
        with self.page.expect_download() as download:
            self.page.locator("#exportJson").click()
        exported = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
        self.assertEqual(exported["audio_offset"], 1.5)
        self.assertEqual(len(exported["structure"]["sections"]), 2)

    def test_complete_import_preserves_timing_and_rejects_other_song(self):
        incoming = copy.deepcopy(self.payload)
        incoming["events"][0]["chord"] = "Em"
        incoming["events"][0]["chord_display"] = "Em"
        incoming["bars"][0]["spans"][0].update(chord="Em", chord_display="Em")
        self.page.locator("#tabTools").click()
        self.page.locator("#importAnalysis").set_input_files({"name":"corrected.json", "mimeType":"application/json", "buffer":json.dumps(incoming).encode()})
        self.page.wait_for_function("events[0].chord === 'Em'")
        self.assertEqual(self.evaluate("events[0].end"), .25)
        incoming["duration"] = 88
        self.page.locator("#importAnalysis").set_input_files({"name":"other.json", "mimeType":"application/json", "buffer":json.dumps(incoming).encode()})
        self.page.wait_for_function("document.getElementById('importStatus').textContent.includes('同一音频')")
        self.assertEqual(self.evaluate("DATA.duration"), 12)
        self.page.locator("#undoEdit").click()
        self.assertEqual(self.evaluate("events[0].chord"), "C")

    def test_overlapping_import_requires_explicit_reconciliation(self):
        incoming = copy.deepcopy(self.payload)
        duplicate = copy.deepcopy(incoming["events"][3])
        duplicate.pop("bar")
        duplicate.update(chord="A7", chord_display="A7")
        incoming["events"].append(duplicate)
        self.page.locator("#tabTools").click()
        self.page.locator("#importAnalysis").set_input_files({"name":"conflicted.json", "mimeType":"application/json", "buffer":json.dumps(incoming).encode()})
        self.page.wait_for_function("timelineAudit.conflicted")
        self.assertTrue(self.page.locator("#timelineAlert").is_visible())
        self.assertTrue(self.page.locator("#saveChord").is_disabled())
        self.assertTrue(self.evaluate("events.some(e => e.chord === 'A7')"))
        self.page.locator("#rebuildEventTimeline").click()
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))
        self.assertFalse(self.evaluate("events.some(e => e.chord === 'A7')"))
        self.assertEqual(self.evaluate("DATA.timeline_reconciliation.reason"), "explicit_user_choice")
        self.page.locator("#undoEdit").click()
        self.assertTrue(self.evaluate("timelineAudit.conflicted"))
        self.assertIsNone(self.evaluate("DATA.timeline_reconciliation || null"))

    def test_roman_quality_and_modulation_are_preserved(self):
        result = self.evaluate("['II7','ii7','Imaj7','6m7','IV','C/E/G'].map(x => parseEditableChord(x)?.chordDisplay || null)")
        self.assertEqual(result, ["D7","Dm7","Cmaj7","Am7","F",None])
        self.assertEqual(self.evaluate("editKeyOverride='G'; const slash=parseEditableChord('C/E'); editKeyOverride=null; [slash.chordDisplay,slash.romanDisplay,slash.bassRoot]"),["C/E","IV/VI","E"])
        self.assertEqual(self.evaluate("editKeyOverride='G'; const slash=parseEditableChord('C/D'); editKeyOverride=null; [slash.chordDisplay,slash.romanDisplay]"),["C/D","IV/V"])
        self.evaluate("DATA.modulations=[{start_bar:4,key:'D',mode:'major'}]; selectEvent(events.findIndex(e=>e.bar===4))")
        self.assertEqual(self.evaluate("parseEditableChord('C').roman"), "bVII")
        self.evaluate("scoreText.value='| 6 |'; scoreStart.value='1'; scoreRepeat.checked=true; applyScoreToTimeline()")
        self.assertEqual(self.evaluate("DATA.bars.map(b=>b.spans[0].chord)"), ["Am","Am","Am","Bm","Bm","Bm"])
        self.evaluate("DATA.modulations=[]; DATA.key_timeline={segments:[{start_bar:1,end_bar:3,key:'Am'},{start_bar:4,end_bar:6,key:'D'}]}; DATA.mode='minor'; selectEvent(events.findIndex(e=>e.bar===4))")
        self.assertEqual(self.evaluate("editingKey()"),"D")
        self.assertEqual(self.evaluate("parseEditableChord('C').roman"),"bVII")

    def test_section_rename_preserves_shared_boundary_bar(self):
        self.evaluate("DATA.structure.sections[0].end=5.5; DATA.structure.sections[1].start=5.5; DATA.structure.sections[1].start_bar=3; render(); recordHistory('fixture')")
        self.page.locator("#tabStructure").click()
        self.page.locator("#sectionName").fill("确认主歌")
        self.page.locator("#saveSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections[0].name"), "确认主歌")
        self.assertEqual(self.evaluate("DATA.structure.sections[0].end"), 5.5)
        self.assertEqual(self.evaluate("DATA.structure.manual_overrides[1].start"), 5.5)

    def assert_structure_can_be_reanalyzed(self):
        from song_structure import apply_manual_sections
        payload = self.evaluate("DATA")
        manual = payload["structure"]["manual_overrides"]
        restored = apply_manual_sections({}, manual, payload)
        self.assertEqual([(row["start"],row["end"],row["name"]) for row in restored["sections"]],
                         [(row["start"],row["end"],row["name"]) for row in manual])

    def test_section_rename_preserves_uncovered_zero_bar(self):
        self.evaluate("DATA.bars[0].bar=0; DATA.bars[0].grid_status='uncovered'; DATA.bars[0].beats=[]; DATA.bars[0].spans.forEach(s=>s.bar=0); DATA.events.filter(e=>e.bar===1).forEach(e=>e.bar=0); DATA.structure.sections[0].start_bar=0; render(); recordHistory('fixture')")
        self.page.locator("#tabStructure").click()
        self.assertEqual(self.page.locator("#sectionStartBar").input_value(), "0")
        self.assertEqual(self.page.locator("#sectionStartBar").get_attribute("min"), "0")
        self.page.locator("#sectionName").fill("未定位前奏")
        self.page.locator("#saveSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections[0].name"), "未定位前奏")
        self.assertEqual(self.evaluate("DATA.structure.sections[0].start"), 0)
        self.assertEqual(self.evaluate("DATA.structure.sections[0].start_bar"), 0)
        self.assert_structure_can_be_reanalyzed()
        self.page.locator("#undoEdit").click()
        self.assertEqual(self.evaluate("DATA.structure.sections[0].name"), "段落 A")

    def test_section_boundary_updates_neighbor_and_keeps_complete_coverage(self):
        self.page.locator("#tabStructure").click()
        self.page.locator("#sectionEndBar").fill("2")
        self.page.locator("#saveSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections.map(s=>[s.start,s.end])"),[[0,4],[4,12]])
        self.assertEqual(self.evaluate("DATA.structure.sections[1].confidence"),.7)
        self.assertEqual(self.evaluate("DATA.structure.boundaries.map(b=>b.time)"),[4])
        self.assertEqual(self.evaluate("DATA.structure.automatic_sections[0].end"),6)
        self.assert_structure_can_be_reanalyzed()
        self.page.locator("#sectionStartBar").fill("2")
        self.page.locator("#saveSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections[0].start"),0)
        self.page.locator("#undoEdit").click()
        self.assertEqual(self.evaluate("DATA.structure.sections.map(s=>[s.start,s.end])"),[[0,6],[6,12]])

    def test_section_partition_and_merge_preserve_neighbors_and_round_trip(self):
        self.page.locator("#tabStructure").click()
        self.page.locator("#sectionStartBar").fill("2")
        self.page.locator("#sectionEndBar").fill("2")
        self.page.locator("#sectionName").fill("人工过门")
        self.page.locator("#addSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections.map(s=>[s.start,s.end,s.name])"),[[0,2,"段落 A"],[2,4,"人工过门"],[4,6,"段落 A"],[6,12,"段落 B"]])
        self.assertEqual(self.evaluate("new Set(DATA.structure.sections.map(s=>s.id)).size"),4)
        self.assertEqual(self.evaluate("DATA.structure.boundaries.map(b=>b.time)"),[2,4,6])
        self.assert_structure_can_be_reanalyzed()
        self.assertEqual(self.page.locator("#deleteSection").inner_text(),"并入上一段")
        self.page.locator("#deleteSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections.map(s=>[s.start,s.end,s.name])"),[[0,4,"段落 A"],[4,6,"段落 A"],[6,12,"段落 B"]])
        self.assert_structure_can_be_reanalyzed()
        self.page.locator("#undoEdit").click()
        self.assertEqual(self.evaluate("DATA.structure.sections.length"),4)

    def test_first_manual_section_leaves_explicit_unassigned_regions(self):
        self.evaluate("DATA.structure.sections=[];render()")
        self.page.locator("#tabStructure").click()
        self.page.locator("#sectionStartBar").fill("2")
        self.page.locator("#sectionEndBar").fill("3")
        self.page.locator("#sectionName").fill("主歌")
        self.page.locator("#addSection").click()
        self.assertEqual(self.evaluate("DATA.structure.sections.map(s=>[s.start,s.end,s.name])"),[[0,2,"未分段"],[2,6,"主歌"],[6,12,"未分段"]])
        self.assert_structure_can_be_reanalyzed()

    def test_restore_merged_range_recovers_original_subdivisions(self):
        self.evaluate("SOURCE_DATA.bars[0].spans[1].chord='Dm'; SOURCE_DATA.bars[0].spans[1].chord_display='Dm'; SOURCE_DATA.events[1].chord='Dm'; SOURCE_DATA.events[1].chord_display='Dm'; scoreText.value='| G |'; scoreStart.value='1'; scoreRepeat.checked=false; applyScoreToTimeline(); restoreSelectedChord()")
        self.assertEqual(self.evaluate("DATA.bars[0].spans.map(s=>[s.start,s.end,s.chord])"), [[0,.25,"C"],[.25,.75,"Dm"],[.75,2,"C"]])
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))

    def test_import_export_and_draft_keep_full_provenance(self):
        incoming = copy.deepcopy(self.payload)
        incoming["bpm"] = 118.5
        incoming["analysis_provenance"] = {"model":"external-fixture-v2", "unfamiliar_nested_field":[1,{"a":"b"}]}
        incoming["structure"]["provenance"] = {"model":"struct-fixture", "audio_sha256":"fixture-hash"}
        incoming["structure"]["sections"][0].update(name="人工主歌",role="verse",source="manual")
        incoming["structure"]["manual_overrides"] = copy.deepcopy(incoming["structure"]["sections"])
        self.page.locator("#tabTools").click()
        self.page.locator("#importAnalysis").set_input_files({"name":"provenance.json", "mimeType":"application/json", "buffer":json.dumps(incoming).encode()})
        self.page.wait_for_function("DATA.bpm === 118.5")
        self.page.on("dialog",lambda dialog:dialog.accept())
        self.page.reload()
        self.page.wait_for_function("DATA.bpm === 118.5")
        with self.page.expect_download() as download:
            self.page.locator("#exportJson").click()
        exported = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
        self.assertEqual(exported["analysis_provenance"], incoming["analysis_provenance"])
        self.assertEqual(exported["structure"], incoming["structure"])
        self.assertEqual(exported["bpm"], 118.5)
        self.assertTrue(exported["corrected"])
        self.assertGreaterEqual(self.evaluate("audio.readyState"),1)

    def test_import_rejects_shifted_or_nonfinite_timebase(self):
        self.assertTrue(self.evaluate("isCompatibleAnalysis(DATA)"))
        for offset in ("invalid", 1.6):
            incoming = copy.deepcopy(self.payload)
            incoming["audio_offset"] = offset
            incoming["bpm"] = 77
            self.page.locator("#tabTools").click()
            self.evaluate("byId('importStatus').textContent=''")
            self.page.locator("#importAnalysis").set_input_files({"name":"wrong-offset.json","mimeType":"application/json","buffer":json.dumps(incoming).encode()})
            self.page.wait_for_function("byId('importStatus').textContent.includes('偏移一致')")
            self.assertEqual(self.evaluate("DATA.audio_offset"),1.5)
            self.assertEqual(self.evaluate("DATA.bpm"),120)

    def test_drafts_do_not_cross_analysis_versions(self):
        self.page.locator("#chordInput").fill("Dm")
        self.page.locator("#saveChord").click()
        original_key = self.evaluate("autosaveStorageKey")
        other = copy.deepcopy(self.payload)
        other["events"][0].update(chord="Em",chord_display="Em")
        other["bars"][0]["spans"][0].update(chord="Em",chord_display="Em")
        other_html = self.directory / "other_baseline.html"
        write_html(other_html,other)
        dialogs = []
        self.page.on("dialog",lambda dialog:(dialogs.append(dialog.message),dialog.dismiss()))
        self.page.goto(other_html.as_uri())
        self.page.wait_for_function("historyIndex>=0")
        self.assertNotEqual(self.evaluate("autosaveStorageKey"), original_key)
        self.assertEqual(self.evaluate("events[0].chord"), "Em")
        self.assertEqual(dialogs, [])
        same_chords_new_model = copy.deepcopy(self.payload)
        same_chords_new_model["analysis_provenance"] = {"model":"a-new-model-version"}
        write_html(other_html,same_chords_new_model)
        self.page.reload()
        self.page.wait_for_function("historyIndex>=0")
        self.assertNotEqual(self.evaluate("autosaveStorageKey"), original_key)
        self.assertEqual(self.evaluate("events[0].chord"), "C")
        self.assertEqual(dialogs, [])

    def test_invalid_audio_does_not_disable_annotation(self):
        missing = copy.deepcopy(self.payload)
        missing.update(source=str(self.directory / "missing-audio.wav"),audio_file="missing-audio.wav")
        missing_html = self.directory / "missing-audio.html"
        write_html(missing_html,missing)
        self.page.goto(missing_html.as_uri())
        self.page.wait_for_function("document.getElementById('sessionStatus').textContent.includes('音频未连接')")
        self.page.locator("#chordInput").fill("Dm")
        self.page.locator("#saveChord").click()
        self.assertEqual(self.evaluate("events[0].chord"), "Dm")
        self.assertEqual(self.page.locator("#duration").inner_text(), "00:12")
        self.assertIn("音频未连接",self.page.locator("#sessionStatus").inner_text())

    def test_keyboard_navigation_keeps_focus_with_selection(self):
        self.page.locator("#bars").focus()
        self.page.keyboard.press("ArrowRight")
        self.assertEqual(self.evaluate("selectedEventIndex"), 1)
        self.assertEqual(self.evaluate("document.activeElement.dataset.bar"), "1")
        self.page.keyboard.press("Enter")
        self.assertEqual(self.evaluate("document.activeElement.id"), "chordInput")
        self.page.keyboard.type("Fmaj7")
        self.page.keyboard.press("Enter")
        self.assertEqual(self.evaluate("events[1].chord"), "Fmaj7")

    def test_cross_bar_events_edit_without_duplicate_intervals(self):
        self.evaluate("""(() => {
          const cross = {...events[0],start:0,end:4,bar:1,retime_original_grid:{bar:99}};
          DATA.events = [cross,...events.filter(e=>e.start>=4)];
          DATA.bars[1].spans.forEach(s=>{s.chord='C';s.chord_display='C'});
          DATA.bars[1].beats.forEach(s=>{s.chord='C';s.chord_display='C'});
          refreshDerived();render();selectEvent(0);recordHistory('fixture');
        })()""")
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))
        self.page.locator('.bar[data-bar="2"] .span-cell').click()
        self.assertEqual(self.evaluate("selectedEventIndex"), 0)
        self.page.locator("#chordInput").fill("Dm")
        self.page.locator("#saveChord").click()
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))
        self.assertEqual(self.evaluate("DATA.bars.slice(0,2).flatMap(b=>b.spans).every(s=>s.chord==='Dm')"), True)
        self.assertEqual(self.evaluate("events[0].retime_original_grid.bar"), 99)
        self.page.locator('.beat[data-bar="2"][data-beat="2"]').click()
        self.page.locator("#chordInput").fill("F")
        self.page.locator("#saveChord").click()
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))
        self.assertEqual(self.evaluate("events.filter(e=>e.start<4).map(e=>[e.start,e.end,e.chord])"), [[0,2,"Dm"],[2,2.5,"Dm"],[2.5,3,"F"],[3,4,"Dm"]])

    def test_uncovered_audio_regions_do_not_invent_beats(self):
        self.evaluate("""(() => {
          const head={bar:0,start:0,end:.25,grid_status:'uncovered',beats:[],spans:[{...events[0],bar:0,beat_start:null,beat_end:null}]};
          DATA.bars[0].start=.25;DATA.bars[0].spans.shift();DATA.bars[0].beats[0].start=.25;
          DATA.bars.unshift(head);DATA.events[0].bar=0;DATA.events[0].beat_start=null;DATA.events[0].beat_end=null;
          refreshDerived();render();selectEvent(0);seekToAnalysis(.01);
        })()""")
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))

        self.assertEqual(self.page.locator('.bar[data-bar="0"] .beat').count(),0)
        self.assertEqual(self.page.locator('.bar[data-bar="0"] .pos').inner_text(),"拍点未定位")
        self.assertEqual(self.page.locator("#currentPosition").inner_text(),"拍点未定位")
        self.assertEqual(self.page.locator("#barCount").inner_text(),"6")
        self.assertTrue(self.page.locator("#applyBoundary").is_disabled())
        self.page.locator("#chordInput").fill("Em")
        self.page.locator("#saveChord").click()
        self.assertEqual(self.evaluate("events[0].chord"),"Em")
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))

    def test_precise_seek_uses_audio_offset_and_validates_time(self):
        self.page.locator("#currentTime").click()
        self.page.locator("#seekTimeInput").fill("00:03.125")
        self.page.locator("#applySeekTime").click()
        self.assertAlmostEqual(self.evaluate("audio.currentTime"),4.625,places=3)
        self.assertTrue(self.evaluate("audio.paused"))
        self.page.locator("#currentTime").click()
        self.page.locator("#seekTimeInput").fill("00:99.999")
        self.page.locator("#applySeekTime").click()
        self.assertEqual(self.page.locator("#seekTimeInput").get_attribute("aria-invalid"),"true")
        self.assertTrue(self.evaluate("document.getElementById('seekDialog').open"))

    def test_time_link_locates_without_playback_or_annotation_changes(self):
        before = self.evaluate("JSON.stringify(DATA)")
        self.evaluate("syncMs=250;localStorage.setItem(syncStorageKey,'250')")
        self.page.goto(self.html.as_uri()+"#t=3.25&end=5.75")
        self.page.reload()
        self.page.wait_for_function("loopRange.a === 3.25 && audio.readyState >= 1")
        self.assertAlmostEqual(self.evaluate("audio.currentTime"),4.5,places=2)
        self.assertAlmostEqual(self.evaluate("analysisTime()"),3.25,places=2)
        self.assertTrue(self.evaluate("audio.paused"))
        self.assertEqual(self.evaluate("loopRange"),{"a":3.25,"b":5.75,"enabled":False})
        self.assertEqual(self.evaluate("JSON.stringify(DATA)"),before)
        self.assertEqual(self.evaluate("historyIndex"),0)
        self.assertEqual(self.evaluate("events[selectedEventIndex].bar"),2)
        self.page.keyboard.press("l")
        self.assertTrue(self.evaluate("loopRange.enabled"))

    def test_time_link_rejects_invalid_coordinates(self):
        self.page.goto(self.html.as_uri()+"#t=-2&end=500")
        self.page.wait_for_function("audio.readyState>=1")
        self.assertEqual(self.evaluate("loopRange"),{"a":None,"b":None,"enabled":False})
        self.assertAlmostEqual(self.evaluate("analysisTime()"),0,places=2)
        self.evaluate("location.hash='t=2.5&end=Infinity'")
        self.page.wait_for_function("Math.abs(analysisTime()-2.5)<.01")
        self.assertIsNone(self.evaluate("loopRange.b"))
        self.assertTrue(self.evaluate("audio.paused"))

    def test_local_audio_waveform_survives_annotation_edits(self):
        self.page.locator("#localAudioFile").set_input_files(str(self.directory / "fixture.wav"))
        self.page.wait_for_function("waveformPeaks !== null && audio.readyState >= 1")
        self.assertGreaterEqual(self.evaluate("waveformPeaks.length"),300)
        self.assertTrue(self.evaluate("waveformPeaks.every(x=>Number.isFinite(x) && x>=0 && x<=1)"))
        self.assertTrue(self.evaluate("audio.src.startsWith('blob:')"))
        self.page.locator("#chordInput").fill("Dm7")
        self.page.locator("#saveChord").click()
        self.page.locator("#undoEdit").click()
        self.assertTrue(self.evaluate("audio.src.startsWith('blob:')"))
        self.assertEqual(self.evaluate("events[0].chord"),"C")

    def test_candidate_prefill_does_not_apply_without_confirmation(self):
        self.evaluate("DATA.extension_suggestions=[{bar:1,start:0,end:.25,suggested_chord:'Cmaj7',score:.8,eligible:true}];render();selectEvent(0)")
        self.page.locator("[data-candidate]").click()
        self.assertEqual(self.page.locator("#chordInput").input_value(),"Cmaj7")
        self.assertEqual(self.evaluate("events[0].chord"),"C")
        self.page.locator("#saveChord").click()
        self.assertEqual(self.evaluate("events[0].chord"),"Cmaj7")

    def test_key_candidates_are_evidence_and_dialog_keeps_keyboard_focus(self):
        self.evaluate("DATA.key_estimation={source:'madmom_cnn_key',audio_offset:1.5,duration:12,candidates:[{key:'Am',score:.2},{key:'C',score:.7},{key:'G',score:.06},{key:'F',score:.03},{key:'Dm',score:.01}]};render();updateActive()")
        self.assertEqual(self.page.locator("#keyCaption").inner_text(),"调性估计")
        self.page.locator("#keyCandidatesButton").click()
        self.assertEqual(self.page.locator("#keyCandidateList li").count(),4)
        self.assertEqual(self.page.locator("#keyCandidateList li strong").first.inner_text(),"C")
        self.assertIn("证据 0.700",self.page.locator("#keyCandidateList").inner_text())
        self.assertNotIn("%",self.page.locator("#keyCandidateList").inner_text())
        self.page.keyboard.press("ArrowRight")
        self.assertEqual(self.evaluate("selectedEventIndex"),0)
        self.page.keyboard.press("Escape")
        self.assertFalse(self.evaluate("document.getElementById('keyCandidatesDialog').open"))
        self.page.set_viewport_size({"width":390,"height":844})
        self.assertLessEqual(self.evaluate("document.documentElement.scrollWidth"),390)
        self.evaluate("DATA.bars[0].local_key='D';DATA.bars[0].local_mode='major';updateActive()")
        self.assertEqual(self.page.locator("#keyCaption").inner_text(),"局部调性")
        self.assertEqual(self.page.locator("#key").inner_text(),"D major")

    def test_review_filter_navigation_stays_in_visible_bars(self):
        self.page.locator("#reviewOnly").check()
        self.assertEqual(self.evaluate("selectedBarNumber()"),3)
        self.assertEqual(self.page.locator(".bar:visible").count(),1)
        self.page.locator("#bars").focus()
        for key in ["ArrowRight","ArrowLeft","ArrowDown","ArrowUp"]:
            self.page.keyboard.press(key)
            self.assertEqual(self.evaluate("selectedBarNumber()"),3)
        self.page.locator('.beat[data-bar="3"][data-beat="1"]').click()
        self.page.locator("#bars").focus()
        for _ in range(6):
            self.page.keyboard.press("ArrowRight")
        self.assertEqual(self.evaluate("selectedBeatTarget"),{"bar":3,"beat":4})
        self.page.keyboard.press("ArrowDown")
        self.assertEqual(self.evaluate("selectedBarNumber()"),3)

    def test_unlocated_long_timeline_uses_readable_time_cards(self):
        self.evaluate("""(() => {
          const spans=Array.from({length:40},(_,i)=>({...events[0],bar:0,start:i*.3,end:(i+1)*.3,beat_start:null,beat_end:null,chord:i%2?'F':'C',chord_display:i%2?'F':'C'}));
          DATA.events=spans;DATA.bars=[{bar:0,start:0,end:12,beats:[],spans,grid_status:'uncovered'}];
          DATA.structure.sections=[];DATA.meter='unknown';DATA.beats_per_bar=null;
          refreshDerived();render();selectEvent(0);updateActive();
        })()""")
        self.assertEqual(self.page.locator(".uncovered-long .span-cell").count(),40)
        self.assertGreaterEqual(self.page.locator(".uncovered-long .span-cell").first.bounding_box()["width"],116)
        self.assertEqual(self.page.locator(".beat").count(),0)
        self.assertEqual(self.page.locator("#beatPulse i").count(),0)
        self.assertIn("未定位区间",self.page.locator("#songPath").inner_text())
        self.assertEqual(self.page.locator("#barCount").inner_text(),"0")
        self.evaluate("seekToAnalysis(10.8,true)")
        self.page.wait_for_timeout(150)
        self.assertEqual(self.page.locator(".span-cell.active").count(),1)
        active=self.page.locator(".span-cell.active").bounding_box()
        header=self.page.locator(".topbar").bounding_box()
        self.assertGreaterEqual(active["y"],header["y"]+header["height"])
        self.assertLess(active["y"]+active["height"],1000)

    def test_display_title_and_expert_grid_review_are_explicit(self):
        self.evaluate("DATA.display_title='示例歌曲 · 专家拍格建议';DATA.structure.grid_review_required=true;DATA.structure.expert_grid={applied_to_display:true};render()")
        self.assertEqual(self.page.locator("#songTitle").inner_text(),"示例歌曲 · 专家拍格建议")
        self.assertEqual(self.evaluate("baseExportName()"),"示例歌曲 · 专家拍格建议")
        self.assertEqual(self.evaluate("DATA.audio_file"),"fixture.wav")
        self.assertTrue(self.page.locator("#gridReviewNotice").is_visible())
        self.assertIn("尚需人工核对",self.page.locator("#gridReviewNotice").inner_text())

    def test_local_meter_controls_pulses_grid_and_sixteenth_splits(self):
        self.evaluate("""(() => {
          const bar=DATA.bars[0];bar.meter='3/8';bar.beats_per_bar=3;
          bar.beats=Array.from({length:3},(_,i)=>({...bar.beats[0],beat:i+1,start:i*2/3,end:(i+1)*2/3}));
          bar.spans=[{...events[0],start:0,end:2,beat_start:1,beat_end:3}];
          DATA.events=[...bar.spans,...events.filter(e=>e.start>=2)];DATA.meter_mode='variable';
          refreshDerived();render();selectEvent(0);updateActive();
        })()""")
        self.assertEqual(self.page.locator("#meter").inner_text(),"3/8")
        self.assertEqual(self.page.locator("#beatPulse i").count(),3)
        self.assertEqual(self.evaluate("getComputedStyle(document.querySelector('.bar[data-bar=\"1\"]')).getPropertyValue('--beats')"),"3")
        self.assertEqual(self.evaluate("getComputedStyle(document.querySelector('.bar[data-bar=\"1\"] .span-lane')).getPropertyValue('--subdivisions')"),"6")
        self.evaluate("splitSelectedEvent('sixteenth')")
        self.assertEqual(self.evaluate("DATA.bars[0].spans.length"),6)
        self.assertAlmostEqual(self.evaluate("DATA.bars[0].spans[0].end"),1/3,places=6)
        self.assertFalse(self.evaluate("timelineAudit.conflicted"))
        self.evaluate("seekToAnalysis(2.1)")
        self.assertEqual(self.page.locator("#meter").inner_text(),"4/4")
        self.assertEqual(self.page.locator("#beatPulse i").count(),4)

    def test_incomplete_grid_is_not_filled_and_score_order_is_preserved(self):
        self.evaluate("DATA.bars[1].grid_status='incomplete';DATA.bars[1].beats.splice(1,1);refreshDerived();render();selectEvent(3);updateActive()")
        self.assertEqual(self.page.locator('.bar[data-bar="2"] .beat').count(),3)
        self.assertEqual(self.page.locator('.bar[data-bar="2"] .span-grid').count(),0)
        self.assertTrue(self.page.locator("#applyBoundary").is_disabled())
        self.assertTrue(self.page.locator("#gridReviewNotice").is_visible())
        self.evaluate("scoreText.value='| Dm | Em | G |';scoreStart.value='1';scoreRepeat.checked=false;applyScoreToTimeline()")
        self.assertEqual(self.evaluate("DATA.bars.slice(0,3).map(b=>b.spans[0].chord)"),["Dm","Am","G"])
        self.assertIn("跳过 1",self.page.locator("#scoreStatus").inner_text())

    def test_unexported_model_confidence_is_not_shown_as_zero_evidence(self):
        self.evaluate("[...DATA.events,...DATA.bars.flatMap(b=>[...b.spans,...b.beats])].forEach(e=>{e.confidence=0;e.confidence_status='not_exported_by_model'});refreshDerived();render();selectEvent(0)")
        self.assertEqual(self.page.locator("#selectionConfidence").inner_text(),"证据未提供")
        self.assertEqual(self.page.locator(".event .conf").first.text_content(),"未提供")
        self.assertTrue(self.page.locator("#missingEvidenceLegend").is_visible())
        self.assertIn("模型未提供证据分值",self.page.locator("#overviewHint").inner_text())
        self.assertEqual(self.evaluate("getComputedStyle(document.querySelector('.bar'),'::before').backgroundColor"),"rgb(152, 166, 160)")
        self.evaluate("DATA.events[0].confidence_status='provided';DATA.events[0].confidence=1;DATA.confidence_kind='weighted_temporal_support';selectEvent(0)")
        self.assertIn("时间段中和弦标签的支持程度",self.page.locator("#selectionConfidence").get_attribute("title"))

    def test_embedded_waveform_validates_timebase_and_yields_to_local_audio(self):
        self.evaluate("DATA.audio_waveform={schema:'peak_envelope_v1',peaks:[0,.5,1,.25],duration:12,audio_offset:1.5};render()")
        self.assertEqual(self.evaluate("Array.from(waveformPeaks)"),[0,.5,1,.25])
        self.assertIn("音频波形",self.page.locator("#overviewHint").inner_text())
        self.evaluate("DATA.audio_waveform={...DATA.audio_waveform,audio_offset:0};render()")
        self.assertIsNone(self.evaluate("waveformPeaks"))
        self.evaluate("DATA.audio_waveform={...DATA.audio_waveform,audio_offset:1.5,peaks:[null,.5,1]};render()")
        self.assertIsNone(self.evaluate("waveformPeaks"))
        self.evaluate("DATA.audio_waveform={...DATA.audio_waveform,peaks:[0,1,0]};render()")
        self.assertEqual(self.evaluate("waveformPeaks.length"),3)
        self.page.locator("#localAudioFile").set_input_files(str(self.directory / "fixture.wav"))
        self.page.wait_for_function("waveformPeaks!==null && waveformPeaks.length>=300")
        count=self.evaluate("waveformPeaks.length")
        self.evaluate("DATA.audio_waveform={...DATA.audio_waveform,peaks:[1,0,1]};render()")
        self.assertEqual(self.evaluate("waveformPeaks.length"),count)

    def test_import_refreshes_embedded_waveform_and_preserves_payload(self):
        incoming=copy.deepcopy(self.payload)
        incoming["audio_waveform"]={"schema":"peak_envelope_v1","peaks":[0,.5,1,.25],"duration":12,"audio_offset":1.5,"source":"decoded_audio"}
        self.page.locator("#tabTools").click()
        self.page.locator("#importAnalysis").set_input_files({"name":"waveform.json","mimeType":"application/json","buffer":json.dumps(incoming).encode()})
        self.page.wait_for_function("waveformPeaks !== null")
        self.assertEqual(self.evaluate("Array.from(waveformPeaks)"),[0,.5,1,.25])
        self.assertEqual(self.evaluate("DATA.audio_waveform"),incoming["audio_waveform"])
        self.page.locator("#undoEdit").click()
        self.assertIsNone(self.evaluate("waveformPeaks"))



if __name__ == "__main__":
    unittest.main()
