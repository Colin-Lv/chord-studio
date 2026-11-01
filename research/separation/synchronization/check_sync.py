"""Actual Chromium audio rendering and real Xuni UI; no separation jobs started."""
import argparse
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--installed', action='store_true')
args = parser.parse_args()
session = json.loads(Path('D:/ChordStudio/data/active-session.json').read_text())
base = 'http://127.0.0.1:' + str(session['port'])
checks, errors = [], []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe', headless=True, args=['--mute-audio'])
    context = browser.new_context(viewport={'width': 1500, 'height': 1000})
    context.add_cookies([dict(name='chord_session', value=session['token'], url=base)])
    page = context.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    if not args.installed:
        for asset in ['stems.js', 'stem-audio.js']:
            page.route('**/assets/' + asset, lambda route, request, name=asset: route.fulfill(path=str(REPO / 'frontend' / name), content_type='text/javascript'))
    page.goto(base + '/stems?project=aa5a41797bb343e590abcf4d8b27ae54&model=bs_roformer_sw')
    page.wait_for_function("state.tracks.some(t=>t.id==='acoustic_guitar')")
    page.add_script_tag(content=(ROOT / 'test_transport.js').read_text())
    checks.extend(page.evaluate('transportRenderChecks()'))
    print('Offline audio renders passed', flush=True)
    start = time.monotonic()
    page.locator('#play').click()
    page.wait_for_function('state.playing && StemAudio.position()>0.1', timeout=90000)
    loaded = page.evaluate('({tracks:StemAudio.sources.length, bytes:[...StemAudio.job.records.values()].reduce((n,r)=>n+r.buffer.length*r.buffer.numberOfChannels*4,0)})')
    loaded.update(name='actual_song_decoded', load_seconds=round(time.monotonic()-start, 3))
    checks.append(loaded)
    page.evaluate('window.originalSources=[...StemAudio.sources];window.originalGeneration=StemAudio.generation')
    for i in range(24):
        track = ['acoustic_guitar', 'drums', 'bass', 'lead_vocals'][i % 4]
        page.locator(f'.track-head[data-track="{track}"] .solo').click()
        page.wait_for_timeout(60)
        page.locator(f'.track-head[data-track="{track}"] button:not(.solo)').click()
        page.wait_for_timeout(60)
    page.evaluate('state.solo=null;state.muted={};applyAudioState()')
    page.locator('#reference').check();page.wait_for_timeout(80)
    assert page.evaluate("[...StemAudio.job.records].filter(([id,r])=>r.gain.gain.value>0.5).map(([id])=>id)") == ['mix']
    page.locator('#reference').uncheck();page.wait_for_timeout(80)
    assert page.evaluate('originalGeneration===StemAudio.generation && originalSources.every((s,i)=>s===StemAudio.sources[i])')
    assert page.evaluate('[...StemAudio.job.records].every(([id,r])=>Math.abs(r.gain.gain.value-(id==="mix"?0:1))<0.001)')
    checks.append(dict(name='48_mute_solo_actions_and_reference_keep_same_sources'))
    page.evaluate('seek(30)');page.wait_for_function('StemAudio.position()>30.2')
    page.locator('#play').click();paused = page.evaluate('StemAudio.position()')
    page.wait_for_timeout(250)
    assert page.evaluate('StemAudio.position()') == paused
    assert page.evaluate('StemAudio.live.size===0')
    page.locator('#play').click();page.wait_for_function('StemAudio.position()>' + str(paused + .15))
    checks.append(dict(name='seek_pause_resume_preserve_common_position'))
    page.locator('#loopBar').check()
    assert page.evaluate('StemAudio.sources.every(s=>s.loop && s.loopStart===StemAudio.range[0] && s.loopEnd===StemAudio.range[1])')
    blocked = page.evaluate('''() => {const start=performance.now();while(performance.now()-start<650){};
        return {position:StemAudio.position(),range:StemAudio.range};}''')
    assert blocked['range'][0] <= blocked['position'] < blocked['range'][1]
    page.locator('#loopBar').uncheck()
    page.locator('#play').click();page.wait_for_timeout(100)
    assert page.evaluate('StemAudio.live.size===0')
    checks.append(dict(name='bar_loop_and_blocked_ui_use_audio_clock'))
    # Pending download cancellation, track change, HTTP failure and recovery.
    page.evaluate('StemAudio.reset(state.manifest.duration);window.audioTestFetch=window.fetch.bind(window)')
    def transport(mode):
        page.evaluate('''mode => {window.fetch=(url,options={})=>{
          if(!String(url).includes('.mp3')||mode==='real')return audioTestFetch(url,options);
          if(mode==='fail')return Promise.resolve(new Response('missing',{status:404}));
          return new Promise((resolve,reject)=>{const abort=()=>reject(new DOMException('Cancelled','AbortError'));
            if(options.signal.aborted)abort();else options.signal.addEventListener('abort',abort,{once:true});});
        };}''', mode)
    transport('hold');page.locator('#play').click();page.wait_for_function('state.starting')
    page.locator('#play').click()
    assert page.evaluate('!state.playing && !state.starting && !StemAudio.job && !StemAudio.live.size')
    transport('hold');page.locator('#play').click();page.wait_for_function('state.starting')
    page.locator('#guitarMode').select_option('duet');transport('real')
    assert page.evaluate('!state.playing && !state.starting && !StemAudio.job')
    transport('fail');page.locator('#play').click()
    page.wait_for_function("!state.starting && document.getElementById('toast').textContent.includes('失败')")
    assert page.evaluate('!state.playing && !StemAudio.job && !StemAudio.live.size')
    transport('real');page.evaluate('seek(30)');page.locator('#play').click()
    page.wait_for_function('state.playing && StemAudio.position()>30.2', timeout=90000)
    page.locator('#play').click();page.wait_for_timeout(100)
    checks.append(dict(name='cancel_change_tracks_failure_retry_before_playback'))
    assert not errors, errors
    result = dict(checks=checks, page_errors=errors, installed=args.installed)
    (ROOT / ('installed_checks.json' if args.installed else 'candidate_checks.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    browser.close()
