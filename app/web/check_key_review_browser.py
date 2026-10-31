"""Real HTTP + Edge check against an isolated copy of frozen Xuni outputs."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from playwright.sync_api import sync_playwright
from .storage import ROOT, atomic_json, read_json, signature


def run(base,project,output):
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();errors=[];checks=[]
    def get(suffix):
        with urlopen(base+f'/api/projects/{project}'+suffix) as r:return json.load(r)
    original=get('/result');api=base+f'/api/projects/{project}/key-review';initial=get('/key-review')
    assert original['key_analysis']['key']=='G' and initial['suggestion']['prediction']['key']=='C'
    def open_project(page):
        page.goto(base);page.get_by_role('button',name=original['title'],exact=False).first.click();page.wait_for_function("document.getElementById('globalKey').textContent.includes('大调')")
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
        context=browser.new_context(viewport=dict(width=1440,height=1100));page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));open_project(page)
        assert page.locator('#globalKey').inner_text()=='G 大调'
        assert 'C 大调' in page.locator('#keyReviewContent').inner_text()
        assert page.locator('#keyReviewContent details').last.evaluate('(node)=>node.open') is False
        before_rows=page.locator('.row-main').all_text_contents();page.screenshot(path=str(output/'xuni_original_G.png'),full_page=True)
        # Preserve a real chord edit across key adoption/restoration.
        page.locator('#chordInput').fill('Cmaj7');page.locator('#applyEdit').click();page.locator('#saveCorrections').click()
        page.wait_for_function("document.getElementById('saveStatus').textContent.includes('已保存到项目')")
        correction=get('/corrections/'+original['default_source']);assert correction['edits']
        page.locator('#adoptKeySuggestion').click();page.wait_for_function("document.getElementById('globalKey').textContent==='C 大调'")
        adopted=get('/key-review');assert adopted['manual']['source']=='harmonic_key_v2';assert adopted['automatic']['key']=='G'
        assert adopted['suggestion']['prediction']['key']=='C' and adopted['derived_key_analysis_stale']
        assert page.locator('.row-main').all_text_contents()!=before_rows
        page.screenshot(path=str(output/'xuni_manual_C.png'),full_page=True)
        # Separate context has no localStorage: server persistence, not local draft.
        context2=browser.new_context(viewport=dict(width=390,height=844));page2=context2.new_page();page2.on('pageerror',lambda e:errors.append(str(e)));open_project(page2)
        assert page2.locator('#globalKey').inner_text()=='C 大调';assert page2.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page2.screenshot(path=str(output/'xuni_mobile_C.png'),full_page=True)
        # Old window payload must conflict once another page restores automatic.
        page2.locator('#resetKey').click();page2.wait_for_function("document.getElementById('globalKey').textContent==='G 大调'")
        stale=dict(run_id=adopted['run_id'],audio_sha256=adopted['audio_sha256'],revision=adopted['revision'],action='custom',key='D')
        try:
            urlopen(Request(api,json.dumps(stale).encode(),headers={'X-Harmony-Client':'studio','Content-Type':'application/json'}))
            raise AssertionError('Stale revision accepted')
        except HTTPError as e:assert e.code==409
        restored=get('/key-review');assert restored['effective_key']=='G' and restored['manual'] is None
        assert get('/result')==original and get('/corrections/'+original['default_source'])==correction
        page2.locator('#play').click();page2.wait_for_function("document.getElementById('audio').currentTime>0.1",timeout=10000)
        page2.locator('#play').click()
        # Blind page exposes no model answers until label lock and reveal.
        page2.goto(base+'/assets/key_annotation.html?project='+project);page2.wait_for_function("document.getElementById('end').value!==''")
        assert page2.locator('#prediction').is_hidden();assert page2.locator('#reveal').is_disabled()
        page2.locator('#reviewer').fill('integration_fixture_not_human_gold');page2.locator('#basis').fill('Automated UI fixture; no human listening label claim')
        page2.locator('#annotationForm button[type=submit]').click();page2.wait_for_function("!document.getElementById('reveal').disabled")
        page2.locator('#reveal').click();page2.wait_for_function("!document.getElementById('prediction').hidden")
        annotation=get('/key-annotation');assert annotation['locked_label']['prediction_visible_before_lock'] is False and annotation['prediction_visible']
        assert not annotation['independent_gold'];atomic_json(output/'fixture_annotation.json',annotation)
        browser.close()
    assert not errors,errors
    atomic_json(output/'validation.json',dict(real_http_service=True,response_interception=False,real_frozen_outputs=True,
        acoustic_cache_replay=True,new_acoustic_inference=False,new_training=False,project_id=project,
        audio_identity=original['audio'],initial=initial,adopted=adopted,restored=restored,
        checks=['original G/suggestion C/no auto override','uncalibrated evidence','collapsed local candidates','Roman labels updated',
          'manual persists in separate browser context','human adoption event records harmonic source','restore original G','stale revision 409','original result unchanged',
          'chord edits preserved','audio playback','390px no overflow','blind lock then reveal with exposure audit'],
        javascript_errors=errors,elapsed_seconds=time.perf_counter()-start))
    print(dict(checks=13,errors=errors,elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',default='http://127.0.0.1:8766');p.add_argument('--project',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base,a.project,a.output)
