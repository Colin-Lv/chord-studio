"""Cancel, switch, fail and retry while actual multi-stem media is loading."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;PID='aa5a41797bb343e590abcf4d8b27ae54'
session=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text());BASE='http://127.0.0.1:'+str(session['port'])
checks=[];errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--mute-audio'])
    context=browser.new_context();context.add_cookies([dict(name='chord_session',value=session['token'],url=BASE)])
    page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+f'/stems?project={PID}&model=bs_roformer_sw')
    page.wait_for_function("state.guitarParts?.mode==='timbre'")
    page.evaluate("window.audioTestFetch=window.fetch.bind(window)")
    def transport(mode):
        page.evaluate("""mode => { window.fetch=(url,options={})=>{
          if (!String(url).includes('.mp3') || mode==='real') return window.audioTestFetch(url,options);
          if (mode==='fail') return Promise.resolve(new Response('missing',{status:404}));
          return new Promise((resolve,reject)=>{
            const abort=()=>reject(new DOMException('Cancelled','AbortError'));
            if(options.signal.aborted)abort();else options.signal.addEventListener('abort',abort,{once:true});
          });
        }; }""",mode)
    transport('hold');page.locator('#play').click();page.wait_for_function('state.starting')
    page.locator('#play').click();assert page.evaluate("!state.starting && !state.playing && [...document.querySelectorAll('audio')].every(a=>a.paused)")
    transport('real');checks.append('cancel_pending_downloads_stops_all_media')
    transport('hold');page.locator('#play').click();page.wait_for_function('state.starting')
    page.locator('#guitarMode').select_option('duet');transport('real')
    assert page.evaluate("!state.playing && !state.starting && state.tracks.some(t=>t.id==='guitar_one')")
    checks.append('mode_change_cancels_old_play_request')
    transport('fail')
    page.locator('#play').click();page.wait_for_function("!state.starting && document.getElementById('toast').textContent.includes('失败')")
    assert page.evaluate("!state.playing && [...document.querySelectorAll('audio')].every(a=>a.paused)")
    transport('real');checks.append('missing_media_reports_failure_without_partial_playback')
    page.evaluate('seek(30)');page.locator('#play').click()
    page.wait_for_function("state.playing && state.audios.guitar_one.currentTime>30.5",timeout=45000)
    page.locator('#play').click();checks.append('retry_recovers_and_preserves_seek_before_loading')
    assert not errors,errors
    (ROOT/'audio_lifecycle_checks.json').write_text(json.dumps(dict(checks=checks,page_errors=errors),indent=2))
    print(json.dumps(checks));browser.close()
