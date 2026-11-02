const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
function fixture(){
  class Node{
    constructor(){this.children=[];this.value='';this.textContent='';this.hidden=false;this.currentTime=0;this.duration=9;this.paused=true;this.width=1200;this.height=470;this.checked=false;}
    append(...nodes){this.children.push(...nodes);if(!this.value&&nodes[0]?.value)this.value=nodes[0].value;}
    replaceChildren(){this.children=[];this.value='';}
    get options(){return this.children;}
    querySelector(){return {disabled:false};}
    pause(){this.paused=true;}
    getContext(){return new Proxy({},{get:()=>()=>{},set:()=>true});}
    getBoundingClientRect(){return {left:0,top:0,width:1200,height:470};}
  }
  const nodes={},calls=[];const node=id=>nodes[id]||(nodes[id]=new Node());
  const manifest={generation:'generation',input_kind:'piano',duration:9,notes:1,bars:2,bpm:80,created_at:'2026-11-02T00:00:00Z',limitations:['测试限制'],urls:{'notes.json':'/notes','performance.mid':'/performance','grid.mid':'/grid','preview.wav':'/preview'}};
  const data={bars:[{number:1,start:1},{number:2,start:4}],notes:[{pitch:60,velocity:70,start:1,end:2,bar:1,beat:1,fraction:0,held_quarters:1.33,rhythm:{kind:'subdivision',subdivision:'0'},chord_context:'C:maj'}]};
  const context={document:{getElementById:node,createElement:()=>new Node()},history:{replaceState(){}},location:{search:'?project=song'},URLSearchParams,Date,Math,console,setTimeout,FormData:class{},fetch:async(url,opts)=>{calls.push({url,opts});let value;
    if(url==='/api/projects')value=[{id:'song',title:'曲目',audio:'audio.wav'}];
    else if(url==='/api/projects/song')value={id:'song',audio:'audio.wav'};
    else if(url==='/api/projects/song/piano'&&opts.method==='POST')value={id:'job'};
    else if(url==='/api/projects/song/piano')value=manifest;
    else if(url==='/api/jobs/job')value={status:'completed'};
    else if(url==='/notes')value=data;
    else throw new Error('unexpected '+url);
    return {ok:true,json:async()=>value};
  }};
  vm.createContext(context);vm.runInContext(fs.readFileSync('frontend/piano.js','utf8'),context);
  return {context,nodes,calls};
}
test('actual piano entry loads saved results with stable downloads and correct input identity',async()=>{
  const f=fixture();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.nodes.result.hidden,false);assert.equal(f.nodes.performance.href,'/performance');assert.equal(f.nodes.grid.href,'/grid');assert.equal(f.nodes.inputKind.value,'piano');
  assert.match(f.nodes.sourceNote.textContent,/直接输入/);assert.equal(f.nodes.audio.src,'/preview');
  f.nodes.inputKind.value='mix';assert.equal(f.nodes.performance.href,'/performance'); // preference changes retain results
  await vm.runInContext('generate()',f.context);
  const call=f.calls.find(c=>c.opts.method==='POST');assert.equal(JSON.parse(call.opts.body).input,'mix');
  assert.equal(f.nodes.generate.disabled,false);assert.equal(f.nodes.result.hidden,false);
});
test('audition switching and note inspection use the original audio clock',async()=>{
  const f=fixture();await new Promise(resolve=>setImmediate(resolve));
  f.nodes.audio.currentTime=4;f.nodes.listen.value='source';await f.nodes.listen.onchange();
  assert.equal(f.nodes.audio.src,'/api/projects/song/audio');f.nodes.audio.onloadedmetadata();assert.equal(f.nodes.audio.currentTime,4);
  f.nodes.next.onclick();assert.match(f.nodes.windowLabel.textContent,/00:01.00/);
  const rect=vm.runInContext('state.notes[0]',f.context);assert.ok(rect);
  f.nodes.roll.onclick({currentTarget:f.nodes.roll,clientX:rect.x+1,clientY:rect.y+1});
  assert.match(f.nodes.noteDetail.textContent,/力度 70/);assert.equal(f.nodes.audio.currentTime,1);
});
