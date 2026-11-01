// DOM contract check without launching a browser or controlling the desktop.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Node {
  constructor(tag){this.tag=tag;this.children=[];this.style={};this.attributes={};this.textContent='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=[...nodes];}
  setAttribute(k,v){this.attributes[k]=v;}
}
const panel=new Node('section');global.document={createElement:tag=>new Node(tag),getElementById:id=>id==='harmonicKeyTimeline'?panel:null};global.window={};
vm.runInThisContext(fs.readFileSync(process.argv[2],'utf8'));
vm.runInThisContext(fs.readFileSync('frontend/harmonic-key.js','utf8'));
const data=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
let selected=null;
const source=data.sources.find(s=>s.id===data.default_source);
window.HarmonicKeyTimeline.render(data,source,t=>selected=t);
function nodes(n){return [n,...n.children.flatMap(nodes)];}
assert(nodes(panel).some(n=>n.textContent.includes('局部调性与转调')));
const button=nodes(panel).find(n=>n.tag==='button');assert(button);button.onclick();assert.strictEqual(selected,data.harmonic_keys[source.id].segments[0].start);
assert(nodes(panel).some(n=>n.tag==='table'));
window.HarmonyNotation.setKey('C');window.HarmonyNotation.parseInKey('B:maj','B');assert.strictEqual(window.HarmonyNotation.parse('C:maj').romanDisplay,'I');
window.HarmonicKeyTimeline.render(data,{id:'missing'},()=>{});
assert(!nodes(panel).some(n=>n.tag==='table'));assert(nodes(panel).some(n=>n.textContent.includes('尚无调性时间线')));
console.log('PASS: source binding, timeline seek, local Roman notation, manual-key isolation, stale content clearing');
