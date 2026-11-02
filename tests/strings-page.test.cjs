const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
function fixture(){
 const nodes={},calls=[];const node=id=>nodes[id]??=( {value:'',hidden:false,checked:false,textContent:''} );
 const c=vm.createContext({URL,URLSearchParams,location:{search:'?project=song',href:'http://localhost/stems?project=song'},
  document:{getElementById:node,querySelectorAll:()=>[]},localStorage:{getItem:()=>null,setItem:()=>{}},
  window:{confirm:()=>{throw Error('Cached model must not prompt')}},setTimeout,clearTimeout,console,
  StemAudio:{setMix(m){this.mix=m}},history:{replaceState(){}},GuitarParts:{}});
 vm.runInContext(fs.readFileSync('frontend/stems.js','utf8').replace(/boot\(\);\s*$/,''),c);
 return {c,nodes,calls,run:s=>vm.runInContext(s,c)};
}
test('seven-track view has one strings track and replaces expanded parents',()=>{
 const f=fixture();f.run(`state.manifest={tracks:['vocals','drums','bass','other','guitar','piano','strings'].map(id=>({id}))};state.expandVocals=true;state.vocalParts={tracks:[{id:'lead_vocals'},{id:'backing_vocals'}]};`);
 const ids=Array.from(f.run('composedTracks().map(t=>t.id)'));
 assert.equal(ids.filter(id=>id==='strings').length,1);assert(!ids.includes('vocals'));assert.equal(ids.length,8);
});
test('cached version switching performs reads only and retains both options',async()=>{
 const f=fixture();f.run(`state.manifest={model_key:'strings_v2',model_options:[{id:'strings_v1',ready:true},{id:'strings_v2',ready:true}]};globalThis.reads=[];api=async u=>{reads.push(u);return {model_key:u.endsWith('strings_v1')?'strings_v1':'strings_v2',model_options:state.manifest.model_options}};start=async m=>{state.manifest=m};`);
 await f.run(`switchModel('strings_v1')`);await f.run(`switchModel('strings_v2')`);
 assert.equal(f.run('reads.length'),2);assert.equal(f.run('state.manifest.model_options.length'),2);
});
test('a song without previous stems can start strings directly',async()=>{
 const f=fixture();f.run(`state.manifest=null;globalThis.selected=null;runSeparation=async (force,model)=>{selected={force,model}};`);
 await f.run(`switchModel('strings_v2')`);
 assert.equal(f.run('selected.model'),'strings_v2');assert.equal(f.run('selected.force'),false);
});
test('sub-splits share preview level; mute and solo do not reset audio',()=>{
 const f=fixture();f.run(`state.manifest={preview_gain:.6,tracks:[{id:'strings'},{id:'vocals'}]};state.tracks=[{id:'strings'},{id:'lead_vocals'}];state.audios={strings:{id:'strings'},lead_vocals:{id:'lead_vocals'}};state.mix={id:'mix'};state.master=.5;state.position=23;applyAudioState();`);
 assert.equal(f.run('state.audios.strings.volume'),.5);assert.equal(f.run('state.audios.lead_vocals.volume'),.3);
 f.run(`state.solo='strings';applyAudioState()`);assert.equal(f.run('state.audios.lead_vocals.muted'),true);assert.equal(f.run('state.position'),23);
});
