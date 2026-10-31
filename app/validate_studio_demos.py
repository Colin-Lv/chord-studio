"""Refresh and inspect the local review collection with a real browser.

Optional development check; requires Playwright and local Edge/Chromium.
It never changes the analysis JSON files. Screenshots and a JSON report are
written beside the supplied session directory's model reports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from chord_studio import write_html
from build_studio_review_index import recommended_primary_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("benchmarks/reports/session_20260926"))
    parser.add_argument("--primary-report", type=Path, help="Override the automatically selected default workflow report")
    parser.add_argument("--only-primary", action="store_true", help="Inspect only the current default workbenches")
    args = parser.parse_args()
    session = args.session.resolve()
    comparisons = json.loads((session / "full_song_demos/report.json").read_text(encoding="utf-8"))
    primary_report = args.primary_report or recommended_primary_report(session, len(comparisons["demos"]))
    report = json.loads(primary_report.read_text(encoding="utf-8"))
    rows = [(Path(row["prediction"]), Path(row["html"])) for row in report["songs"] if row.get("status") == "complete"]
    for suffix in (() if args.only_primary else ("native", "expert_grid_review")):
        for source in sorted((session / "full_song_demos").glob(f"*_{suffix}_analysis.json")):
            rows.append((source, source.with_name(source.name.replace("_analysis.json", ".html"))))
    for source, target in rows:
        write_html(target, json.loads(source.read_text(encoding="utf-8")))

    from playwright.sync_api import sync_playwright
    results = []
    executable = os.environ.get("STUDIO_BROWSER_PATH") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            **({"executable_path": executable} if Path(executable).is_file() else {}),
            headless=True, args=["--allow-file-access-from-files", "--autoplay-policy=no-user-gesture-required"])
        for source, target in rows:
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(target.as_uri())
            page.wait_for_function("historyIndex >= 0 && audio.readyState >= 1")
            result = page.evaluate("""() => ({
                title:byId('songTitle').textContent, audio_ready:audio.readyState,
                timeline_conflicted:timelineAudit.conflicted, events:events.length,
                bars:DATA.bars.length, sections:sections().length,
                waveform_peaks:waveformPeaks?.length || 0,
                review_banner_visible:!byId('gridReviewNotice').hidden,
                width:innerWidth,scroll_width:document.documentElement.scrollWidth
            })""")
            result["html"] = str(target.relative_to(session))
            assert not result["timeline_conflicted"], target
            assert result["waveform_peaks"] == 1600, target
            assert result["scroll_width"] <= result["width"], target
            page.evaluate("syncMs=250; audio.playbackRate=1.25; seekToAnalysis(15,true)")
            page.wait_for_function("!audio.paused && analysisTime()>15", timeout=10000)
            result["plays"] = page.evaluate("!audio.paused && analysisTime()>15")
            assert result["plays"], target
            page.evaluate("audio.pause();syncMs=0;displaySync();audio.playbackRate=1;seekToAnalysis(7.5);const item=activeFor(7.5);if(item)selectEventByObject(item);scrollTo(0,0)")
            if "qimeidi" in target.name:
                if target.name == "qimeidi_studio.html":
                    page.screenshot(path=str(session / "studio_desktop.png"))
                    page.locator("#themeToggle").click()
                    page.wait_for_timeout(250)
                    page.screenshot(path=str(session / "studio_dark.png"))
                    page.locator("#themeToggle").click()
                    page.set_viewport_size({"width":390,"height":844})
                    page.evaluate("scrollTo(0,0)")
                    page.wait_for_timeout(200)
                    page.screenshot(path=str(session / "studio_mobile.png"))
                    result["mobile_no_overflow"] = page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    assert result["mobile_no_overflow"]
                    page.locator("#showInspector").click()
                    page.wait_for_timeout(150)
                    page.screenshot(path=str(session / "studio_mobile_editor.png"))
                    page.keyboard.press("Escape")
                    page.set_viewport_size({"width":1440,"height":1000})
                elif "expert_grid_review" in target.name:
                    page.screenshot(path=str(session / "studio_expert_grid_review.png"))
                else:
                    page.screenshot(path=str(session / "studio_sheetsage_native.png"))
                before = page.evaluate("({name:sections()[0].name,events:events.map(e=>[e.start,e.end,e.chord]),source:DATA.source,expert:DATA.structure.expert_grid})")
                page.locator("#tabStructure").click()
                page.evaluate("selectSection(sections()[0].id,false)")
                page.locator("#sectionName").fill("浏览器往返校验")
                page.locator("#saveSection").click()
                assert page.evaluate("sections()[0].name") == "浏览器往返校验", target
                with page.expect_download() as download:
                    page.locator("#exportJson").click()
                exported = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
                assert [[e["start"],e["end"],e["chord"]] for e in exported["events"]] == before["events"]
                assert exported["source"] == before["source"]
                assert exported["structure"].get("expert_grid") == before.get("expert")
                assert len(exported["structure"]["manual_overrides"]) == result["sections"]
                page.locator("#undoEdit").click()
                assert page.evaluate("sections()[0].name") == before["name"]
                result["structure_export_undo"] = True
            result["page_errors"] = errors
            assert not errors, errors
            results.append(result)
            result_path = session / ("studio_final_primary_acceptance.json" if args.only_primary else "studio_final_demo_acceptance.json")
            result_path.write_text(json.dumps({"complete":len(results)==len(rows),"primary_report":str(primary_report),"pages":results},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            print(f"PASS {target.name}: {result['events']} events, {result['waveform_peaks']} waveform peaks", flush=True)
            context.close()
        browser.close()


if __name__ == "__main__":
    main()
