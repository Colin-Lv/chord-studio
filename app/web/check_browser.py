"""Browser checks against a completed real project; optional reversible edit test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path(".tmp/web_validation"))
    parser.add_argument("--exercise-edits", action="store_true", help="Only use on a disposable validation project")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], []
    with sync_playwright() as driver:
        browser = driver.chromium.launch(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(args.base_url)
        page.evaluate("id => openProject(id)", args.project_id)
        page.locator("#resultPanel").wait_for(state="visible")
        page.wait_for_function("audio.readyState >= 1")
        original = page.evaluate("JSON.stringify(state.result)")
        result = json.loads(original)
        assert len(result["sources"]) == page.locator(".source-card").count()
        assert page.locator(".chord-row").count() > 0
        page.evaluate("audio.muted=true;seek(3)")
        page.locator("#play").click()
        page.wait_for_function("audio.currentTime > 3.25")
        page.locator("#play").click()
        assert page.locator(".chord-row.playing").count() == 1
        checks.append("HTTP audio with seeking, actual playback and synchronized selected-source highlight")

        page.locator("#keySelect").select_option("E")
        page.locator("#displayName").click()
        assert page.locator("#displayName").get_attribute("aria-pressed") == "true"
        page.locator("#displayRoman").click()
        for index in range(len(result["sources"])):
            position = page.evaluate("audio.currentTime")
            page.locator(".source-card").nth(index).click()
            assert page.evaluate("audio.currentTime") == position
            assert page.locator("#keySelect").input_value() == "E"
            assert page.locator(".chord-row.playing .row-main").inner_text() == page.locator("#nowChord").inner_text()
        page.locator(".source-card").first.click()
        checks.append("manual key and name/degree toggle across every source without changing playback time")

        if args.exercise_edits:
            # This CLI requires an explicit flag before writing test corrections.
            page.locator(".chord-row").first.click()
            page.locator("#chordInput").fill("V7")
            page.locator("#applyEdit").click()
            assert page.locator("#chordInput").input_value() == "B7"
            assert page.locator("#nowChord").inner_text() == "V7"
            page.locator("#saveCorrections").click()
            page.wait_for_function("!document.getElementById('saveCorrections').disabled")
            assert "已保存到项目" in page.locator("#saveStatus").inner_text()
            with page.expect_download() as capture:
                page.locator("#exportLab").click()
            text = Path(capture.value.path()).read_text(encoding="utf-8")
            source = result["sources"][0]
            lines = text.splitlines()
            for row, line in zip(source["rows"], lines, strict=True):
                assert line.split("\t")[:2] == [row["start_text"], row["end_text"]]
            assert lines[0].endswith("\tB7")
            fresh = browser.new_context(viewport={"width": 1440, "height": 1100})
            second = fresh.new_page()
            second.on("pageerror", lambda exc: errors.append(str(exc)))
            second.goto(args.base_url)
            second.evaluate("id => openProject(id)", args.project_id)
            second.locator("#resultPanel").wait_for(state="visible")
            second.locator(".chord-row").first.click()
            assert second.locator("#chordInput").input_value() == "B7"
            assert second.locator("#keySelect").input_value() == "E"
            second.locator("#restoreChord").click()
            second.locator("#resetKey").click()
            second.locator("#saveCorrections").click()
            second.wait_for_function("!document.getElementById('saveCorrections').disabled")
            assert "已保存到项目" in second.locator("#saveStatus").inner_text()
            fresh.close()
            checks.append("V7 resolves to B7; saved LAB timestamps; separate browser reload; original chord restored and saved")
        assert page.evaluate("JSON.stringify(state.result)") == original
        checks.append("model JSON and source labels remain unchanged by display or editing")
        page.locator("#resetKey").click()
        page.evaluate("seek(Math.min(57,state.result.duration/2));window.scrollTo(0,0)")
        page.screenshot(path=str(args.output / (args.project_id + "_desktop.png")), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "mobile horizontal overflow"
        page.screenshot(path=str(args.output / (args.project_id + "_mobile.png")), full_page=True)
        checks.append("1440px desktop and 390px mobile, no horizontal overflow")
        assert not errors, errors
        report = dict(project_id=args.project_id, run_id=result["run_id"], audio=result["audio"], models=result["model_identity"],
                      checks=checks, javascript_errors=errors, acoustic_accuracy_test=False,
                      source_intervals={source["id"]:len(source["rows"]) for source in result["sources"]})
        (args.output / (args.project_id + "_browser.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        browser.close()
    print(json.dumps(dict(project_id=args.project_id, passed=len(checks), javascript_errors=errors)))


if __name__ == "__main__":
    main()
