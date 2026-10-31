/* Derived authored diagnostics only. No storage or application API writes. */
(function(){'use strict';
const names={uncovered:'无预测覆盖',invalid_or_no_chord:'无和弦或无效',root:'根音',base_triad:'基本性质',seventh:'七音',upper_extension:'九音/更高扩展',correct:'符合规格'};
const finite=n=>typeof n==='number'&&Number.isFinite(n),digest=s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s);
const core={
  validate(data){
    if(!data||data.schema!=='paired_timbre_listening_review_v1'||!digest(data.source_report_sha256)||!Array.isArray(data.pairs)||!Array.isArray(data.segments)||!Array.isArray(data.summary))throw Error('诊断记录无效');
    const pairs=new Map();for(const pair of data.pairs){
      if(pairs.has(pair.id))throw Error('作曲配对重复');
      for(const entry of [pair.warm,pair.piano])if(!entry||!/^[a-z0-9_]+$/.test(entry.id)||!digest(entry.sha256)||!finite(entry.duration)||entry.duration<=0)throw Error('样例身份无效');
      if(pair.warm.family!==pair.piano.family||pair.warm.transpose!==pair.piano.transpose||pair.warm.duration!==pair.piano.duration)throw Error('配对规格不同');pairs.set(pair.id,pair);
    }
    for(const row of data.segments){const pair=pairs.get(row.pair);if(!pair||!Object.hasOwn(data.engines,row.engine)||!['piano_loses','piano_wins','both_mismatch'].includes(row.kind)||!finite(row.start)||!finite(row.end)||row.start<0||row.end<=row.start||row.end>pair.warm.duration+1e-8||!Object.hasOwn(names,row.warm_error)||!Object.hasOwn(names,row.piano_error))throw Error('差异区间无效');}
    return data;
  },
  link(entry,row){if(!entry||!digest(entry.sha256)||!finite(row.start)||!finite(row.end)||row.start<0||row.end<=row.start||row.end>entry.duration+1e-8)throw Error('区间链接无效');const p=new URLSearchParams({case:entry.id,case_sha256:entry.sha256,start:String(row.start),end:String(row.end),context:'.5'});return '/assets/synthetic-review.html?'+p;},
  filter(data,engine,pair,kind){return data.segments.filter(row=>row.engine===engine&&(pair==='all'||row.pair===pair)&&(kind==='all'||row.kind===kind)).sort((a,b)=>(b.end-b.start)-(a.end-a.start)||a.pair.localeCompare(b.pair)||a.start-b.start);}
};
if(typeof module==='object'&&module.exports){module.exports=core;return;}
const $=id=>document.getElementById(id),make=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
let data,page=0;const pageSize=12;
function render(){
  const engine=$('engine').value,rows=core.filter(data,engine,$('pair').value,$('kind').value),pages=Math.max(1,Math.ceil(rows.length/pageSize));page=Math.max(0,Math.min(page,pages-1));
  const summary=data.summary.find(x=>x.engine===engine);$('metrics').replaceChildren();
  for(const [key,label] of [['root','根音'],['triad','基本三和弦'],['tetrad','包含七音'],['full_harmony','完整和声含九音']]){const box=make('article',undefined,'gap-metric');box.append(make('h3',label),make('strong',summary.piano.pct[key].toFixed(2)+'%'),make('small','钢琴 · 程序音色 '+summary.warm.pct[key].toFixed(2)+'%'));$('metrics').append(box);}
  $('count').textContent=rows.length+' 段';$('segments').replaceChildren();
  for(const row of rows.slice(page*pageSize,(page+1)*pageSize)){
    const pair=data.pairs.find(p=>p.id===row.pair),article=make('article',undefined,'gap-row'),text=make('div'),labels=make('p',undefined,'gap-labels'),links=make('div',undefined,'gap-links');
    text.append(make('h3',pair.title+' · '+row.start.toFixed(3)+'–'+row.end.toFixed(3)+' 秒'),make('small','作曲规格 '+row.reference+' · 钢琴判断：'+names[row.piano_error]));
    labels.append(make('span','程序 '+(row.warm??'无覆盖')),make('span','钢琴 '+(row.piano??'无覆盖')));text.append(labels);
    for(const [entry,label] of [[pair.warm,'听程序音色 ↗'],[pair.piano,'听钢琴 ↗']]){const a=make('a',label);a.href=core.link(entry,row);a.target='_blank';a.rel='noopener';links.append(a);}article.append(text,links);$('segments').append(article);
  }
  if(!rows.length)$('segments').append(make('p','此筛选下没有待复核片段。可换一个来源或查看其他差异类型。','gap-empty'));
  $('pageInfo').textContent=(page+1)+' / '+pages;$('previous').disabled=page===0;$('next').disabled=page>=pages-1;
  $('status').textContent='已载入 '+data.pairs.length+' 组配对；当前显示 '+data.engines[engine]+'。未自动播放。';
}
async function start(){try{const response=await fetch('/assets/diagnostic-data/timbre-gap-v1.json',{cache:'no-store'});if(!response.ok)throw Error('诊断数据暂不可用');data=core.validate(await response.json());
  for(const [value,name] of Object.entries(data.engines)){$('engine').append(new Option(name,value));}$('engine').value='v1';for(const pair of data.pairs)$('pair').append(new Option(pair.title,pair.id));
  for(const id of ['engine','pair','kind'])$(id).addEventListener('change',()=>{page=0;render();});
  $('previous').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};$('sourceScope').textContent=data.model_scope+'。报告身份 '+data.source_report_sha256.slice(0,16)+'…';$('content').hidden=false;render();
}catch(error){$('content').hidden=true;$('status').textContent='无法打开复核记录：'+error.message;}}
start();
})();
