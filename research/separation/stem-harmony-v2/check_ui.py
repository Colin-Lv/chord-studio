"""Headless browser testing only; never controls the user's desktop window."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
here=Path(__file__).resolve().parent
s=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text())
base='http://127.0.0.1:'+str(s['port']);pid='aa5a41797bb343e590abcf4d8b27ae54'
checks=[];errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1000})
    context.add_cookies([dict(name='chord_session',value=s['token'],url=base)])
    page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(base);page.evaluate('id=>openProject(id)',pid)
    page.wait_for_function('state.result?.stem_harmony_v2')
    page.locator('#pipeOpenResult').click()
    page.wait_for_function('document.getElementById("pipeModel").options.length>0')
    page.locator('#pipeChordsOnly').click()
    page.wait_for_function('document.getElementById("pipeChordsOnly").disabled')
    assert '完成。' not in page.locator('#inputStatus').inner_text()
    page.wait_for_function('!document.getElementById("pipeChordsOnly").disabled',timeout=90000)
    assert '完成。请在和弦来源' in page.locator('#inputStatus').inner_text()
    page.wait_for_function('state.result.stem_harmony_v2.implementation_sha256')
    checks.append('actual_chords_only_button_runs_installed_queue_without_early_success')
    page.locator('#sources button').filter(has_text='分轨构成音和弦').last.click()
    assert page.evaluate('state.source.id.startsWith("stem_evidence_v2_")')
    assert page.evaluate('state.source.rows.length')==64
    page.locator('#displayName').click()
    assert 'stem_evidence_v2_' in page.locator('#exportOriginal').get_attribute('href')
    checks.append('new_source_selectable_named_chords_and_export_available')
    page.locator('#sources').scroll_into_view_if_needed()
    page.screenshot(path=str(here/'installed_source.png'))
    # A controlled failed response must not report success or change the default.
    fake='f'*32
    page.route('**/api/projects/'+pid+'/fusion',lambda r:r.fulfill(json=dict(id=fake,profile='fusion'),content_type='application/json'))
    page.route('**/api/jobs/'+fake,lambda r:r.fulfill(json=dict(id=fake,profile='fusion',project_id=pid,status='failed',
        stage_label='测试失败',stages=[],error='模拟解码失败'),content_type='application/json'))
    page.locator('#pipeOpenResult').click()
    page.locator('#pipeChordsOnly').click()
    page.wait_for_function('document.getElementById("inputStatus").textContent.includes("模拟解码失败")')
    checks.append('failed_job_not_reported_as_complete')
    assert not errors,errors
    (here/'ui_checks.json').write_text(json.dumps(dict(checks=checks,page_errors=errors),indent=2))
    print(checks);browser.close()
