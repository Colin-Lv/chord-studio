const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const grid=require('../frontend/chord-grid.js');
test('main grid splits by real time, repeats sustained chords and preserves uncovered time',()=>{
  const bars=[{bar:1,start:1,end:5},{bar:2,start:5,end:8}];
  const rows=[{start:0,end:2.25,label:'C:maj'},{start:2.25,end:7,label:'G:maj'}];
  const actual=grid.layout(bars,rows,8);
  assert.deepEqual(actual.map(b=>b.number),[0,1,2]);
  assert.deepEqual(actual[1].parts.map(p=>p.width),[31.25,68.75]);
  assert.deepEqual(actual[2].parts.map(p=>p.sourceIndex),[1,-1]);
  assert.deepEqual(actual[2].parts.map(p=>[p.start,p.end]),[[5,7],[7,8]]);
});
test('without beats it does not invent four-beat bars',()=>{
  const actual=grid.layout([],[{start:0,end:10,label:'C:maj'}],10);
  assert.equal(actual.length,1);assert.equal(actual[0].number,null);assert.equal(actual[0].parts[0].width,100);
});
test('actual app renderRows renders user-mode blocks and playback follows across bars',()=>{
  class Node{
    constructor(tag,cls='',text=''){this.tag=tag;this.classes=new Set(cls.split(' '));this.textContent=text;this.children=[];this.style={};this.dataset={};this.attrs={};this.scrollTop=0;this.offsetHeight=82;this.clientHeight=560;
      this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),toggle:(c,on)=>on?this.classes.add(c):this.classes.delete(c)};}
    append(...children){this.children.push(...children)}
    replaceChildren(fragment){this.children=fragment.children}
    setAttribute(k,v){this.attrs[k]=v}removeAttribute(k){delete this.attrs[k]}
    getBoundingClientRect(){return {top:0}}
  }
  const nodes=Object.fromEntries(['chordRows','rowCount','editCount','follow'].map(k=>[k,new Node('div')]));nodes.follow.checked=false;
  const rows=[{start:0,end:1.5,label:'C:maj'},{start:1.5,end:6,label:'G:maj'}];
  const state={result:{bars:[{bar:1,start:0,end:4},{bar:2,start:4,end:8}]},source:{rows},selected:0,active:-1};
  let seekTo=null,selected=null;
  const context={state,$:id=>nodes[id],document:{createDocumentFragment:()=>new Node('fragment')},
    element:(...args)=>new Node(...args),window:{ChordGrid:grid,HarmonyTime:{range:(a,b)=>`${a}-${b}`}},
    duration:()=>8,fmt:String,currentDraft:()=>({edits:new Map()}),chord:(s,i)=>s.rows[i].label,
    setLabel:(main,sub,label)=>main.textContent=label,seek:t=>seekTo=t,selectRow:i=>selected=i,
    at:(parts,t)=>parts.findIndex(p=>p.start<=t&&t<p.end)};
  const source=fs.readFileSync(require.resolve('../frontend/app.js'),'utf8');
  vm.createContext(context);vm.runInContext(source.slice(source.indexOf('function renderRows(){'),source.indexOf('function selectRow(')),context);
  context.renderRows();assert.equal(nodes.chordRows.children.length,2);assert.equal(state.rows.length,4);
  assert.equal(state.rows[0].style.width,'37.5%');assert.equal(state.rows[1].style.left,'37.5%');
  state.rows[2].onclick();assert.equal(selected,1);assert.equal(seekTo,4);
  context.paintChordGrid(3.9);assert.equal(state.rows[1].classes.has('playing'),true);
  context.paintChordGrid(4);assert.equal(state.rows[1].classes.has('playing'),false);assert.equal(state.rows[2].classes.has('playing'),true);
  context.paintChordGrid(8);assert.ok(state.rows.every(n=>!n.classes.has('playing')));
  assert.ok(source.includes('paintChordGrid(time);'));
  const html=fs.readFileSync(require.resolve('../frontend/index.html'),'utf8');
  assert.ok(html.includes('/assets/chord-grid.js'));assert.ok(html.includes('/assets/chord-grid.css'));
  assert.ok(!html.includes('<div class="table-heading">'));
});
