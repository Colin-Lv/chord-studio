const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function fixture(){
 class Node{
  constructor(tag){this.tag=tag;this.textContent='';this.children=[];this.style={};}
  append(...items){this.children.push(...items)}
  replaceChildren(...items){this.children=items}
  setAttribute(){}
 }
 const panel=new Node('section'),seeks=[];
 const c=vm.createContext({window:{HarmonyNotation:{parseInKey:()=>({romanDisplay:'I'})}},document:{getElementById:()=>panel,createElement:t=>new Node(t)}});
 vm.runInContext(fs.readFileSync('frontend/harmonic-key.js','utf8'),c);
 const all=()=>{const walk=n=>[n,...n.children.flatMap(walk)];return walk(panel)};
 const render=(result,source={id:'a',sha256:'first'})=>c.window.HarmonicKeyTimeline.render(result,source,t=>seeks.push(t));
 return {render,all,seeks,text:()=>all().map(n=>n.textContent).join(' ')};
}
function data(){return {run_id:'run',source_sha256:'first',status:'computed',main_key:'C',start_key:'C',end_key:'G',source_name:'当前和弦',changes:[{time:4,from_key:'C',to_key:'G',boundary_interval:[3,4]}],
 segments:[{start:0,end:4,key:'C',candidates:[],evidence:[]},{start:4,end:8,key:'G',candidates:[],evidence:[{start:4,chord:'G:maj',role:'调内',tonic:true}]}],tonicizations:[]};}
test('user mode exposes inference, change times and seekable chord evidence',()=>{
 const html=fs.readFileSync('frontend/index.html','utf8');assert(!html.match(/id="harmonicKeyTimeline"[^>]*audit-only/));
 const f=fixture();f.render({run_id:'run',harmonic_keys:{a:data()}});
 assert.match(f.text(),/和弦推理主调/);assert.match(f.text(),/00:04.00　C 大调 → G 大调/);assert.match(f.text(),/查看推理依据/);assert.match(f.text(),/G:maj/);
 f.all().find(n=>n.tag==='button'&&n.textContent.includes('→')).onclick();assert.deepEqual(f.seeks,[4]);
});
test('changing source clears previous inference and rejects stale identities',()=>{
 const f=fixture(),result={run_id:'run',harmonic_keys:{a:data()}};f.render(result);f.render(result,{id:'other'});
 assert.match(f.text(),/尚无调性时间线/);assert(!f.text().includes('C 大调'));
 f.render({...result,run_id:'new'});assert.match(f.text(),/不匹配/);assert(!f.text().includes('C 大调'));
});
test('single-key analysis states no inferred modulation rather than inventing one',()=>{
 const f=fixture(),d=data();d.changes=[];d.segments=d.segments.slice(0,1);d.end_key='C';f.render({run_id:'run',harmonic_keys:{a:d}});
 assert.match(f.text(),/未推断出独立转调段落/);assert(!f.all().some(n=>n.tag==='button'&&n.textContent.includes('→')));
});
