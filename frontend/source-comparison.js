/* Pairwise source differences locate listening work; they do not estimate accuracy. */
(function(root){
  "use strict";
  const labels={root:"根音不同",harmony:"和弦音或扩展不同",bass:"低音标注差异",chord_vs_no_chord:"和弦与 N 不同",unknown_label:"未知或无法解析",missing_prediction:"未覆盖"};
  function matches(payload,doc,left,right){
    if(!payload||payload.schema!=="source_comparison_v1"||!["project_id","run_id","audio_sha256","duration"].every(k=>doc[k]!==undefined&&payload[k]===doc[k]))return false;
    if(!["left","right"].every((side,i)=>payload[side]?.id===[left,right][i].id&&payload[side]?.sha256===[left,right][i].sha256))return false;
    if(!Array.isArray(payload.segments)||!payload.summary||payload.summary.intervals!==payload.segments.length
      ||!Number.isFinite(payload.summary.flagged_seconds)||!Number.isFinite(payload.summary.matching_label_seconds)
      ||payload.summary.flagged_seconds<0||payload.summary.matching_label_seconds<0
      ||Math.abs(payload.summary.flagged_seconds+payload.summary.matching_label_seconds-doc.duration)>1e-6)return false;
    let end=0;
    return payload.segments.every(row=>{const valid=row&&Number.isFinite(row.start)&&Number.isFinite(row.end)&&row.start>=end&&row.end>row.start&&row.end<=doc.duration&&Array.isArray(row.reasons)&&row.reasons.length>0&&row.reasons.every(k=>Object.hasOwn(labels,k))&&[row.left,row.right].every(side=>side&&(side.label===null||typeof side.label==="string"));end=row?.end;return valid;});
  }
  function filtered(payload,reason){return payload.segments.filter(row=>!reason||row.reasons.includes(reason));}
  function auditionRange(start,end,context,duration){
    if(![start,end,context,duration].every(Number.isFinite)||start<0||end<=start||end>duration||context<0||context>5)throw Error("试听范围或前后时长无效");
    return {start:Math.max(0,start-context),end:Math.min(duration,end+context)};
  }
  const logic={matches,filtered,auditionRange};if(typeof module!=="undefined"&&module.exports){module.exports=logic;return;}
  const entries=new WeakMap();let view=null;
  const node=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  function show(session){view=session;if(!entries.has(session))entries.set(session,{right:session.result.sources.find(source=>source.id!==session.source.id)?.id||"",payload:null,loading:false,error:"",filter:"",index:0,context:.5,request:0});draw();}
  async function load(session,entry){
    const right=session.result.sources.find(source=>source.id===entry.right);if(!right)return;
    const token=++entry.request;entry.loading=true;entry.error="";entry.payload=null;draw();
    try{
      const query=new URLSearchParams({left:session.source.id,right:right.id}),response=await fetch(`/api/projects/${session.doc.project_id}/source-comparison?${query}`),payload=await response.json();
      if(!response.ok)throw Error(typeof payload.detail==="string"?payload.detail:"无法读取来源比较");
      if(!matches(payload,session.doc,session.source,right))throw Error("比较结果与当前来源或录音身份不一致，请重新读取");
      if(token===entry.request){entry.payload=payload;entry.index=0;}
    }catch(error){if(token===entry.request)entry.error=error.message;}
    finally{if(token===entry.request)entry.loading=false;if(view===session)draw();}
  }
  function draw(){
    const host=document.getElementById("beSourceComparison"),session=view;if(!host||!session||root.BarEditor.getContext()!==session)return;
    const entry=entries.get(session);host.replaceChildren();
    host.append(node("p","muted","比较两条原始预测，按时间定位可听辨的差异、未知和缺口。来源可能相关；一致不保证正确，差异也不说明哪一方正确。"));
    const options=session.result.sources.filter(source=>source.id!==session.source.id);
    if(!options.length){host.append(node("p","status","当前只有一个来源；可继续按小节试听与修订。"));return;}
    const tools=node("div","be-comparison-tools"),select=node("select");select.id="beCompareSource";select.setAttribute("aria-label","与当前来源比较");
    for(const source of options)select.append(new Option(source.name||source.id,source.id));select.value=entry.right;select.disabled=entry.loading;
    select.onchange=()=>{entry.right=select.value;entry.payload=null;entry.error="";entry.index=0;entry.request++;draw();};
    const button=node("button","secondary",entry.loading?"读取中…":"查找差异区间");button.id="beCompareLoad";button.disabled=entry.loading;button.onclick=()=>load(session,entry);
    tools.append(node("span","",`${session.source.name} ↔`),select,button);host.append(tools);
    if(entry.error){host.append(node("p","status",entry.error));return;}if(!entry.payload)return;
    const payload=entry.payload;
    host.append(node("p","muted",`共 ${payload.summary.intervals} 个定位区间，合计 ${Number(payload.summary.flagged_seconds).toFixed(3)} 秒（含未知与未覆盖）；这是标注差异，不是识别错误时长。`));
    if(payload.identical_source_bytes)host.append(node("p","muted","两条来源文件内容完全相同，不能视为两份独立证据。"));
    const filter=node("select");filter.id="beCompareFilter";filter.setAttribute("aria-label","差异类型");filter.append(new Option("全部定位类型",""));
    for(const [key,label] of Object.entries(labels))filter.append(new Option(label,key));filter.value=entry.filter;filter.onchange=()=>{entry.filter=filter.value;entry.index=0;draw();};host.append(filter);
    const rows=filtered(payload,entry.filter);
    if(!rows.length){host.append(node("p","status",entry.filter?"此类型没有定位区间。":"这两条来源未检出标注差异或缺口；仍需听辨验证。"));return;}
    entry.index=Math.max(0,Math.min(entry.index,rows.length-1));const row=rows[entry.index];
    const nav=node("div","be-comparison-tools"),previous=node("button","secondary","上一处"),next=node("button","secondary","下一处"),index=node("input");
    previous.id="beComparePrevious";next.id="beCompareNext";previous.disabled=entry.index===0;next.disabled=entry.index===rows.length-1;
    previous.onclick=()=>{entry.index--;draw();};next.onclick=()=>{entry.index++;draw();};
    index.id="beCompareIndex";index.type="number";index.min="1";index.max=String(rows.length);index.step="1";index.value=String(entry.index+1);index.setAttribute("aria-label","定位区间编号");
    index.onchange=()=>{const n=Number(index.value);if(Number.isInteger(n)&&n>=1&&n<=rows.length)entry.index=n-1;draw();};nav.append(previous,index,node("span","",`/ ${rows.length}`),next);host.append(nav);
    const card=node("div","be-comparison-card");card.id="beCompareCurrent";
    card.append(node("strong","",`${row.start}–${row.end} 秒`),node("p","",row.reasons.map(k=>labels[k]).join(" · ")));
    for(const side of ["left","right"]){const label=row[side].label;card.append(node("p","",`${payload[side].name}：${label===null?"未覆盖":label==="N"?"N（来源标为无和弦）":label}`));}
    const actions=node("div","be-comparison-tools"),locate=node("button","secondary","定位到小节"),listen=node("button","secondary","带上下文试听"),padding=node("input"),field=node("label","","前后各（秒）");
    locate.id="beCompareLocate";listen.id="beCompareListen";padding.id="beCompareContext";padding.type="number";padding.min="0";padding.max="5";padding.step="0.1";padding.value=String(entry.context);field.append(padding);
    padding.onchange=()=>{const n=Number(padding.value);if(padding.value.trim()&&Number.isFinite(n)&&n>=0&&n<=5)entry.context=n;padding.value=String(entry.context);};
    locate.onclick=()=>{if(root.BarEditor.getContext()===session)root.BarEditor.focusInterval(row.start,row.end);};
    listen.onclick=()=>{if(root.BarEditor.getContext()===session)root.BarEditor.focusInterval(row.start,row.end,{listen:true,context:entry.context});};
    actions.append(locate,listen,field);card.append(actions,node("p","muted","定位与试听不会填入或采纳任一来源标签；原始候选仍可在所选范围逐一核对。"));host.append(card);
  }
  root.SourceComparison={show,logic};
})(typeof window!=="undefined"?window:globalThis);
