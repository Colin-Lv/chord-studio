import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
s=json.loads(Path('D:/ChordStudio/data/active-session.json').read_text());base='http://127.0.0.1:'+str(s['port'])
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--mute-audio'])
    ctx=browser.new_context(viewport={'width':1500,'height':1000});ctx.add_cookies([dict(name='chord_session',value=s['token'],url=base)])
    page=ctx.new_page()
    page.add_init_script("""window.syncTrace=[];window.seekCounts={};
      const d=Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype,'currentTime');
      Object.defineProperty(HTMLMediaElement.prototype,'currentTime',{...d,set(v){
        const k=this.dataset.track;window.seekCounts[k]=(window.seekCounts[k]||0)+1;d.set.call(this,v);
      }});
    """)
    page.goto(base+'/stems?project=aa5a41797bb343e590abcf4d8b27ae54&model=bs_roformer_sw')
    page.wait_for_function("state.tracks.some(t=>t.id==='acoustic_guitar')")
    page.locator('#play').click();page.wait_for_function('state.playing',timeout=45000)
    page.evaluate("""window.traceTimer=setInterval(()=>{
      const times=Object.fromEntries(Object.entries(state.audios).map(([k,a])=>[k,a.currentTime]));
      const values=Object.values(times);syncTrace.push({t:performance.now(),spread:Math.max(...values)-Math.min(...values),times});
    },20)""")
    for i in range(18):
        track=['acoustic_guitar','drums','bass','lead_vocals'][i%4]
        page.locator(f'.track-head[data-track="{track}"] .solo').click()
        page.wait_for_timeout(120)
        page.locator(f'.track-head[data-track="{track}"] button:not(.solo)').click()
        page.wait_for_timeout(120)
    page.evaluate('state.solo=null;state.muted={};applyAudioState()')
    page.wait_for_timeout(1800)
    data=page.evaluate('clearInterval(traceTimer);({trace:syncTrace,seeks:seekCounts})')
    page.locator('#play').click()
    data['max_spread_ms']=max(x['spread'] for x in data['trace'])*1000
    data['last_spread_ms']=data['trace'][-1]['spread']*1000
    (ROOT/'before_measurement.json').write_text(json.dumps(data,indent=2))
    print({k:v for k,v in data.items() if k!='trace'});browser.close()
