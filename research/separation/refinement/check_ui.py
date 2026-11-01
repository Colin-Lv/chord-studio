"""Browser verification against the read-only local preview, no real jobs."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,
                              args=['--autoplay-policy=no-user-gesture-required','--mute-audio'])
    page=browser.new_page(viewport={'width':1500,'height':1050})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:61857/stems?project=aa5a41797bb343e590abcf4d8b27ae54&model=bs_roformer_sw')
    page.wait_for_function("document.querySelectorAll('.track-head').length === 6")
    page.locator('#vocalPartsBtn').click()
    page.wait_for_function("document.querySelectorAll('.track-head').length === 7")
    state=page.evaluate("""() => ({ids:state.tracks.map(t=>t.id), sources:state.tracks.map(t=>state.audios[t.id].src),
       downloads:[...document.querySelectorAll('.track-head a')].map(a=>a.href)})""")
    assert 'vocals' not in state['ids'] and 'lead_vocals' in state['ids'] and 'backing_vocals' in state['ids']
    assert all('model=bs_roformer_sw' in u for u in state['sources'])
    for u in state['downloads']:
        response=page.request.get(u,headers={'Range':'bytes=0-43'})
        assert response.ok, (u,response.status)
    page.evaluate("toggleSolo('lead_vocals')")
    muted=page.evaluate("Object.fromEntries(Object.entries(state.audios).map(([k,a])=>[k,a.muted]))")
    assert muted['lead_vocals'] is False and all(v for k,v in muted.items() if k!='lead_vocals')
    page.locator('#vocalPartsBtn').click()
    page.wait_for_function("document.querySelectorAll('.track-head').length === 6")
    assert page.evaluate('state.solo') is None
    assert page.evaluate('Object.keys(state.audios).length')==6
    page.locator('#vocalPartsBtn').click()
    page.screenshot(path=str(OUT/'vocal_parts_ui.png'),full_page=True)
    assert not errors,errors
    (OUT/'ui_results.json').write_text(json.dumps(dict(state=state,solo_muted=muted,errors=errors,
            checks=['six_to_seven_rows','parent_replaced','selected_model_urls','wav_download','solo','collapse_resets_solo','no_orphan_audio']),indent=2))
    browser.close()
