"""Browser regressions: mocked jobs never submit real separation work."""
import copy
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:61857'
PROJECT = 'aa5a41797bb343e590abcf4d8b27ae54'

def get(path):
    with urllib.request.urlopen(BASE + path) as response:
        return json.load(response)

base = get(f'/api/projects/{PROJECT}/stems?model=bs_roformer_sw')
parts = get(f'/api/projects/{PROJECT}/stems/vocal-parts?model=bs_roformer_sw')
models = ['htdemucs', 'htdemucs_ft', 'bs_roformer_sw']
labels = ['标准（快）', '精细（四模型）', '六轨精细']

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe', headless=True,
        args=['--autoplay-policy=no-user-gesture-required', '--mute-audio'])
    context = browser.new_context(viewport={'width':1500,'height':1050})
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    saved = {'htdemucs'}
    job = None
    submitted = []
    held = []
    delay_model = None
    fail_read = None

    def manifest(model):
        result = copy.deepcopy(base)
        result.update(model_key=model, model_options=[dict(id=m,label=l,ready=True) for m,l in zip(models, labels)])
        result['model']['label'] = labels[models.index(model)]
        for field in ['urls','wav_urls']:
            result[field] = {k:v.replace('bs_roformer_sw',model) for k,v in result[field].items()}
        return result

    def route_api(route):
        global job
        parsed = urlparse(route.request.url)
        path = parsed.path
        model = parse_qs(parsed.query).get('model',['htdemucs'])[0]
        status, result = 200, {}
        if path == f'/api/projects/{PROJECT}':
            result = dict(title='虚拟 · 界面回归验证', active_job=job['id'] if job and job['status']=='running' else None)
        elif path.endswith('/stems'):
            result = manifest(model)
            if model == delay_model:
                held.append((route,result))
                return
        elif path.endswith('/stems/vocal-parts'):
            if model == fail_read:
                status, result = 503, dict(detail='测试：服务暂时不可用')
            elif model not in saved:
                status, result = 404, dict(detail='尚未细分')
            else:
                result = copy.deepcopy(parts)
                result['model'] = model
                for field in ['urls','wav_urls']:
                    result[field] = {k:v.replace('bs_roformer_sw',model) for k,v in result[field].items()}
        elif path.endswith('/vocal-parts') and route.request.method == 'POST':
            submitted.append(route.request.post_data_json)
            job = dict(id='test-job',profile='vocal_parts',model=submitted[-1]['model'],status='running',stage_label='正在分离主唱与背景和声')
            result = job
        elif path.startswith('/api/jobs/'):
            result = job
        elif path.endswith('/result') or path.endswith('/drums-parts'):
            status, result = 404, dict(detail='not found')
        else:
            route.fulfill(status=204)
            return
        route.fulfill(status=status, json=result)

    page.route('**/api/**',route_api)
    checks = []
    def loaded(model):
        page.wait_for_function('(model) => state.manifest?.model_key === model', arg=model)
    def switch(model):
        page.locator('#modelSelect').select_option(model)
        loaded(model)
    def tracks():
        return page.evaluate('state.tracks.map(t=>t.id)')

    page.goto(f'{BASE}/stems?project={PROJECT}&model=htdemucs')
    loaded('htdemucs')
    assert 'lead_vocals' in tracks() and 'vocals' not in tracks()
    assert '标准' in page.locator('#vocalSource').inner_text()
    checks.append('existing_result_auto_expands_with_source')
    switch('htdemucs_ft')
    assert 'vocals' in tracks()
    assert page.locator('#vocalSavedResults button[data-model="htdemucs"]').count()==1
    page.locator('#vocalSavedResults button[data-model="htdemucs"]').click()
    loaded('htdemucs')
    assert 'lead_vocals' in tracks()
    checks.append('saved_result_accessible_from_other_model')
    page.locator('#vocalPartsBtn').click()
    switch('htdemucs_ft')
    switch('htdemucs')
    assert 'vocals' in tracks()
    page.reload();loaded('htdemucs')
    assert 'vocals' in tracks()
    checks.append('explicit_collapse_preserved_across_switch_and_reload')
    switch('htdemucs_ft')
    page.locator('#vocalPartsBtn').click()
    page.wait_for_function('state.vocalJob?.id === "test-job"')
    assert submitted == [{'model':'htdemucs_ft'}]
    switch('bs_roformer_sw')
    assert page.locator('#vocalPartsBtn').is_disabled()
    assert '精细（四模型）' in page.locator('#vocalJobStatus').inner_text()
    page.reload();loaded('bs_roformer_sw')
    page.wait_for_function('state.vocalJob?.id === "test-job"')
    assert '精细（四模型）' in page.locator('#vocalJobStatus').inner_text()
    saved.add('htdemucs_ft');job['status']='completed'
    page.wait_for_function('state.vocalJob === null && !!state.vocalCatalog.htdemucs_ft?.result')
    assert 'vocals' in tracks(), 'Must not replace the currently selected model with another source'
    page.locator('#vocalSavedResults button[data-model="htdemucs_ft"]').click()
    loaded('htdemucs_ft')
    assert 'lead_vocals' in tracks() and 'vocals' not in tracks()
    urls = page.evaluate('Object.values(state.audios).map(a=>a.src)')
    assert all('model=htdemucs_ft' in url for url in urls)
    assert page.evaluate('document.querySelectorAll("audio.stem-audio").length') == len(tracks())+1
    checks.append('job_survives_switch_and_reload_completion_keeps_source_identity')
    page.reload();loaded('htdemucs_ft')
    assert 'lead_vocals' in tracks()
    checks.append('completed_result_and_model_persist_on_reload')

    # Simulate a task started elsewhere: server active_job must suffice without local job memory.
    saved.remove('bs_roformer_sw') if 'bs_roformer_sw' in saved else None
    job = dict(id='remote-job',profile='vocal_parts',model='bs_roformer_sw',status='running',stage_label='正在分离')
    page.evaluate('savePreference({vocalJobId:null})')
    page.reload();loaded('htdemucs_ft')
    page.wait_for_function('state.vocalJob?.id === "remote-job"')
    job.update(status='failed',error='测试任务失败')
    page.wait_for_function('state.vocalJob === null && state.vocalNotice.includes("测试任务失败")')
    assert page.locator('#vocalSavedResults button[data-model="htdemucs_ft"]').count()==1
    checks.append('server_active_job_resume_and_failure_keeps_saved_results')

    delay_model = 'htdemucs'
    page.locator('#modelSelect').select_option('htdemucs')
    page.wait_for_timeout(150)
    assert held
    page.locator('#modelSelect').select_option('bs_roformer_sw')
    loaded('bs_roformer_sw')
    for route,result in held: route.fulfill(json=result)
    page.wait_for_timeout(250)
    assert page.evaluate('state.manifest.model_key')=='bs_roformer_sw'
    assert page.locator('#modelSelect').input_value()=='bs_roformer_sw'
    delay_model = None
    checks.append('late_model_response_cannot_overwrite_new_selection')

    fail_read = 'bs_roformer_sw'
    page.reload();loaded('bs_roformer_sw')
    assert '读取失败' in page.locator('#vocalStatus').inner_text()
    assert page.locator('#vocalSavedResults button[data-model="htdemucs_ft"]').count()==1
    checks.append('read_failure_distinguished_from_missing_result')
    fail_read = None
    switch('htdemucs_ft')
    page.evaluate("toggleSolo('lead_vocals')")
    assert page.evaluate("Object.entries(state.audios).every(([id,a])=>a.muted === (id !== 'lead_vocals'))")
    assert all('model=htdemucs_ft' in url for url in page.locator('.track-download').evaluate_all('(links)=>links.map(a=>a.href)'))
    checks.append('solo_and_download_still_use_selected_source')
    page.screenshot(path=str(OUT/'mock_flow.png'),full_page=True)
    assert not errors, errors

    # Real cache check is read-only; no model inference is triggered.
    real = context.new_page()
    real.on('pageerror',lambda error:errors.append(str(error)))
    real.goto(f'{BASE}/stems?project={PROJECT}&model=bs_roformer_sw')
    real.wait_for_selector('.track-head[data-track="lead_vocals"]')
    assert real.locator('#vocalSavedResults button[data-model="bs_roformer_sw"]').count()==1
    for href in real.locator('.track-head[data-track="lead_vocals"] a, .track-head[data-track="backing_vocals"] a').evaluate_all('(links)=>links.map(a=>a.href)'):
        response = real.request.get(href,headers={'Range':'bytes=0-43'})
        assert response.ok, (href,response.status)
    real.screenshot(path=str(OUT/'real_results.png'),full_page=True)
    checks.append('real_xuni_cached_children_display_and_wav_download')
    assert not errors, errors
    (OUT/'results.json').write_text(json.dumps(dict(checks=checks,errors=errors,submitted_mock_jobs=submitted),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=checks,errors=errors),ensure_ascii=False,indent=2))
    browser.close()
