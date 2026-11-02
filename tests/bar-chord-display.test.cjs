const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {barChordLayout,barIndexAt}=require('../frontend/bar-editor.js');

test('one chord fills the whole bar, not four fixed beats',()=>{
  const e={start:0,end:8,label:'C:maj'};
  const parts=barChordLayout([e],{start:2,end:6});
  assert.equal(parts.length,1);assert.equal(parts[0].left,0);assert.equal(parts[0].width,100);
  assert.equal(parts[0].event,e);
});
test('off-beat changes and short chords keep exact positions',()=>{
  const parts=barChordLayout([{start:10,end:10.05,label:'C:maj'},
    {start:10.05,end:11.3,label:'F:min'},{start:11.3,end:14,label:'G:7'}],{start:10,end:14});
  assert.ok(Math.abs(parts[0].width-1.25)<1e-10);
  assert.ok(Math.abs(parts[1].left-1.25)<1e-10);
  assert.ok(Math.abs(parts[2].left-32.5)<1e-10);
  assert.ok(Math.abs(parts.reduce((sum,p)=>sum+p.width,0)-100)<1e-10);
});
test('a sustained chord is labelled again across unequal bar lengths',()=>{
  const events=[{start:0,end:5,label:'C#:maj7'},{start:5,end:7,label:'C:min'}];
  const first=barChordLayout(events,{start:0,end:4});
  const second=barChordLayout(events,{start:4,end:7});
  assert.equal(first.length,1);assert.equal(second[0].event.label,'C#:maj7');
  assert.ok(Math.abs(second[0].width-100/3)<1e-10);assert.ok(Math.abs(second[1].left-100/3)<1e-10);
});
test('uncovered time and explicit no-chord remain distinct',()=>{
  const parts=barChordLayout([{start:1,end:2,label:'N'}],{start:0,end:4});
  assert.deepEqual(parts.map(p=>p.event?.label??null),[null,'N',null]);
  assert.deepEqual(parts.map(p=>p.width),[25,25,50]);
});
test('playback highlights exactly one block and clears on seek/end',()=>{
  const source=fs.readFileSync(require.resolve('../frontend/bar-editor.js'),'utf8');
  const paint=source.slice(source.indexOf('  function paintPlayback('),source.indexOf('  function playFrame('));
  const make=(start,end)=>({dataset:{start:String(start),end:String(end)},classes:new Set(),attributes:{},
    classList:{add(c){this.owner.classes.add(c)},remove(c){this.owner.classes.delete(c)}},
    setAttribute(k,v){this.attributes[k]=v},removeAttribute(k){delete this.attributes[k]}});
  const cells=[make(0,1.3),make(1.3,4),make(4,8)];cells.forEach(c=>c.classList.owner=c);
  const bars=[{start:0,end:4},{start:4,end:8}];
  const grid={children:[{querySelector:()=>({children:cells.slice(0,2)})},{querySelector:()=>({children:cells.slice(2)})}]};
  const s={doc:{bars},playingIndex:null,playingCell:null};
  const context={current:s,$:()=>grid,barIndexAt};vm.createContext(context);vm.runInContext(paint,context);
  const highlighted=()=>cells.map((c,i)=>c.classes.has('playing')?i:-1).filter(i=>i>=0);
  context.paintPlayback(s,1.299999);assert.deepEqual(highlighted(),[0]);
  context.paintPlayback(s,1.3);assert.deepEqual(highlighted(),[1]);
  context.paintPlayback(s,4);assert.deepEqual(highlighted(),[2]);
  context.paintPlayback(s,.1);assert.deepEqual(highlighted(),[0]);
  context.paintPlayback(s,8);assert.deepEqual(highlighted(),[]);
  assert.ok(cells.every(c=>!('aria-current' in c.attributes)));
});
