"""Isolated browser jobs; never submit actual jobs during interface regression."""
import copy,json,urllib.request
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;BASE='http://127.0.0.1:61858';PID='aa5a41797bb343e590abcf4d8b27ae54'
with urllib.request.urlopen(BASE+f'/api/projects/{PID}/stems?model=bs_roformer_sw') as r:base=json.load(r)
def parts(mode):
    keys=['acoustic_guitar','electric_guitar','guitar_residual'] if mode=='timbre' else ['guitar_one','guitar_two']
    names=['木吉他（估计）','电吉他（估计）','吉他余量'] if mode=='timbre' else ['吉他旋律（估计）','吉他伴奏（估计）']
    tracks=[dict(base['tracks'][4],id=k,name=n,file=f'guitar_parts/{"a"*32}/{k}.mp3',wav=f'guitar_parts/{"a"*32}/{k}.wav') for k,n in zip(keys,names)]
    return dict(model='bs_roformer_sw',mode=mode,tracks=tracks,
        urls={t['id']:f'/api/projects/{PID}/stems/{t["file"]}?model=bs_roformer_sw' for t in tracks},
        wav_urls={t['id']:f'/api/projects/{PID}/stems/{t["wav"]}?model=bs_roformer_sw' for t in tracks})
saved={'timbre':parts('timbre')};job=None;submissions=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--mute-audio'])
    page=browser.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    def route(r):
        global job
        u=urlparse(r.request.url);path=u.path;status=200;data={}
        if path==f'/api/projects/{PID}':data=dict(title='虚拟 · 吉他细分验证',active_job=job['id'] if job and job['status']=='running' else None)
        elif path.endswith('/stems'):
            m=parse_qs(u.query).get('model',['bs_roformer_sw'])[0];data=copy.deepcopy(base);data['model_key']=m
            if m=='htdemucs':data['tracks']=data['tracks'][:4]
        elif path.endswith('/guitar-parts-catalog'):
            data=dict(results=list(saved.values()),active_job=job['id'] if job and job['status']=='running' else None,
              modes=[dict(id='timbre',label='木吉他／电吉他',ready=True,note='独立估计与余量'),dict(id='duet',label='旋律／伴奏吉他',ready=True,note='估计声部')])
        elif path.endswith('/guitar-parts') and r.request.method=='POST':
            body=r.request.post_data_json;submissions.append(body)
            job=dict(id='guitar-job',profile='guitar_parts',model=body['model'],guitar_mode=body['mode'],status='running',stage_label='正在细分吉他');data=job
        elif path.startswith('/api/jobs/'):data=job
        elif path.endswith(('/result','/drums-parts','/vocal-parts')):status=404;data=dict(detail='not generated')
        else:r.fulfill(status=204);return
        r.fulfill(status=status,json=data)
    page.route('**/api/**',route)
    def loaded(model):page.wait_for_function('(m)=>state.manifest?.model_key===m',arg=model)
    def ids():return page.evaluate('state.tracks.map(t=>t.id)')
    page.goto(BASE+f'/stems?project={PID}&model=bs_roformer_sw');loaded('bs_roformer_sw')
    assert 'acoustic_guitar' in ids() and 'guitar' not in ids() and len(ids())==8
    assert '完整吉他轨' in page.locator('#guitarSource').inner_text();checks.append('saved_timbre_auto_expands_and_replaces_parent')
    page.locator('#guitarExpand').click();assert 'guitar' in ids()
    page.locator('#modelSelect').select_option('htdemucs');loaded('htdemucs')
    assert page.locator('#guitarParent').is_visible() and page.locator('#guitarSavedResults button').count()==1
    page.locator('#guitarSavedResults button').click();loaded('bs_roformer_sw')
    assert 'acoustic_guitar' in ids();checks.append('saved_results_accessible_across_parent_versions')
    page.locator('#guitarMode').select_option('duet');assert 'guitar' in ids()
    page.locator('#guitarGenerate').click();page.wait_for_function('!!readPreference().guitarJobId')
    assert submissions==[dict(model='bs_roformer_sw',mode='duet',force=False)]
    page.locator('#guitarMode').select_option('timbre');assert 'acoustic_guitar' in ids()
    page.reload();loaded('bs_roformer_sw');page.wait_for_function('document.getElementById("guitarJobStatus").textContent.includes("正在细分")')
    assert page.locator('#guitarGenerate').is_disabled()
    saved['duet']=parts('duet');job['status']='completed'
    page.wait_for_selector('#guitarSavedResults button[data-mode="duet"]')
    assert 'acoustic_guitar' in ids();checks.append('background_job_survives_mode_switch_and_reload')
    page.locator('#guitarSavedResults button[data-mode="duet"]').click()
    page.wait_for_selector('.track-head[data-track="guitar_one"]')
    assert len(ids())==7 and 'guitar' not in ids() and 'acoustic_guitar' not in ids()
    sources=page.evaluate('Object.entries(state.audios).filter(([k])=>k.startsWith("guitar_")).map(([,a])=>a.src)')
    assert all('model=bs_roformer_sw' in s for s in sources)
    page.evaluate("toggleSolo('guitar_one')")
    assert page.evaluate("Object.entries(state.audios).every(([k,a])=>a.muted===(k!=='guitar_one'))")
    assert page.locator('.track-head[data-track="guitar_one"] .track-download').get_attribute('href').endswith('guitar_one.wav?model=bs_roformer_sw')
    checks.append('duet_source_urls_download_and_solo')
    page.locator('#guitarExpand').click();page.reload();loaded('bs_roformer_sw')
    assert 'guitar' in ids() and page.locator('#guitarMode').input_value()=='duet'
    assert len(ids())+1==page.locator('audio.stem-audio').count();checks.append('collapse_persists_no_orphan_audio')
    page.locator('#guitarGenerate').click();page.wait_for_function('!!readPreference().guitarJobId')
    assert submissions[-1]['force']
    job.update(status='failed',error='模拟失败，旧结果保留')
    page.wait_for_function('document.getElementById("guitarJobStatus").textContent.includes("模拟失败")')
    assert page.locator('#guitarSavedResults button[data-mode="duet"]').count()==1
    page.locator('#guitarSavedResults button[data-mode="duet"]').click();assert 'guitar_one' in ids()
    checks.append('failed_recompute_keeps_previous_result')
    page.screenshot(path=str(ROOT/'ui_mock.png'),full_page=True)
    assert not errors,errors
    (ROOT/'ui_checks.json').write_text(json.dumps(dict(checks=checks,errors=errors,submissions=submissions),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False));browser.close()
