"""Real installed UI, assets and playback; no request mocking and no new jobs."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;PID='aa5a41797bb343e590abcf4d8b27ae54'
session=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text());BASE='http://127.0.0.1:'+str(session['port'])
checks=[];errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--mute-audio'])
    context=browser.new_context(viewport={'width':1500,'height':1100})
    context.add_cookies([dict(name='chord_session',value=session['token'],url=BASE,httpOnly=True,sameSite='Strict')])
    page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+f'/stems?project={PID}&model=bs_roformer_sw')
    page.wait_for_function("state.guitarParts?.mode==='timbre' && state.tracks.some(t=>t.id==='acoustic_guitar')",timeout=45000)
    assert page.locator('#guitarSavedResults button').count()==2
    assert page.locator('#guitarSource').inner_text()=='来源：六轨精细 · 完整吉他轨'
    checks.append('both_real_results_visible_with_correct_parent')
    page.locator('.track-head[data-track="acoustic_guitar"] .solo').click()
    page.locator('#play').click()
    try:
        page.wait_for_function("state.playing && state.audios.acoustic_guitar.currentTime>1 && state.audios.acoustic_guitar.readyState>=2",timeout=30000)
    except Exception:
        print(page.evaluate("({playing:state.playing, solo:state.solo, errors:[...document.querySelectorAll('audio')].map(a=>({track:a.dataset.track,ready:a.readyState,network:a.networkState,time:a.currentTime,paused:a.paused,error:a.error?.message,src:a.getAttribute('src')}))})"),flush=True)
        print('page errors',errors,flush=True)
        page.screenshot(path=str(ROOT/'playback_failure.png'))
        raise
    assert page.evaluate("Object.entries(state.audios).filter(([id,a])=>!a.muted).map(([id])=>id)")==['acoustic_guitar']
    page.locator('#play').click();checks.append('real_acoustic_audio_decodes_plays_and_solos')
    page.locator('#guitarMode').select_option('duet')
    assert page.evaluate("state.tracks.some(t=>t.id==='guitar_one') && !state.tracks.some(t=>t.id==='guitar')")
    assert '吉他旋律' in page.locator('.track-head[data-track="guitar_one"]').inner_text()
    href=page.locator('.track-head[data-track="guitar_one"] .track-download').get_attribute('href')
    response=context.request.get(BASE+href,headers={'Range':'bytes=0-4095'})
    assert response.status==206 and response.body()[:4]==b'RIFF'
    checks.append('role_names_and_downloaded_wav_are_actual_output')
    page.reload();page.wait_for_function("state.guitarParts?.mode==='duet' && state.expandGuitars")
    checks.append('role_selection_survives_reload')
    page.locator('#modelSelect').select_option('htdemucs')
    page.wait_for_function("state.manifest?.model_key==='htdemucs'")
    assert page.locator('#guitarSavedResults button').count()==2
    page.locator('#guitarSavedResults button[data-mode="timbre"]').click()
    page.wait_for_function("state.manifest?.model_key==='bs_roformer_sw' && state.guitarParts?.mode==='timbre'")
    checks.append('cross_parent_navigation_returns_to_saved_real_result')
    page.locator('#guitarExpand').click();assert page.evaluate("state.tracks.some(t=>t.id==='guitar')")
    page.reload();page.wait_for_function("state.manifest && state.guitarParts?.mode==='timbre'")
    assert not page.evaluate('state.expandGuitars')
    page.locator('#guitarExpand').click();checks.append('collapse_and_expand_survive_reload')
    assert page.evaluate("document.querySelectorAll('audio.stem-audio').length===state.tracks.length+1")
    assert not errors,errors
    page.locator('#guitarSource').scroll_into_view_if_needed()
    page.screenshot(path=str(ROOT/'installed_ui.png'),full_page=True)
    (ROOT/'installed_ui_checks.json').write_text(json.dumps(dict(checks=checks,page_errors=errors),indent=2))
    print(json.dumps(checks),flush=True);browser.close()
