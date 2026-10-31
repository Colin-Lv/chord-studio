/* Read-only comparison to one explicitly selected human listening fragment. */
(function(root){
  "use strict";
  const history=typeof module!=="undefined"&&module.exports?require("./key-listening-history.js"):root.KeyListeningHistory;
  function identityMatches(record,doc){return !!record&&["project_id","run_id","audio_sha256","duration"].every(k=>doc[k]!==undefined&&record[k]===doc[k])&&Number.isSafeInteger(record.revision)&&record.revision>=0&&Array.isArray(record.history);}
  function scope(label,window){const start=Math.max(label.excerpt_start,window.start),end=Math.min(label.excerpt_end,window.end);return Number.isFinite(start)&&Number.isFinite(end)&&start<end?{start,end}:null;}
  function localKey(label){
    if(label.annotation_status!=="single_key"||!["major","minor"].includes(label.mode)||typeof label.tonic!=="string"||!/^(?:C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B)$/.test(label.tonic))return null;
    return label.tonic+(label.mode==="minor"?"m":"");
  }
  function choices(record,doc,window){if(!identityMatches(record,doc))throw Error("片段听辨记录与当前项目、分析、音频或时长不一致");return history.fragmentEntries(record.history,doc.duration).entries.filter(event=>scope(event.label,window));}
  const logic={identityMatches,scope,localKey,choices};if(typeof module!=="undefined"&&module.exports){module.exports=logic;return;}
  const node=(tag,cls,text)=>{const element=document.createElement(tag);if(cls)element.className=cls;if(text!==undefined)element.textContent=text;return element;};
  const cache=new Map();let view=null;
  const key=doc=>[doc.project_id,doc.run_id,doc.audio_sha256,doc.duration].join(":");
  async function refresh(entry,doc){
    if(entry.loading)return;entry.loading=true;entry.error="";draw();
    try{const response=await fetch(`/api/projects/${doc.project_id}/key-annotation`);const record=await response.json();if(!response.ok)throw Error(typeof record.detail==="string"?record.detail:"无法读取片段听辨记录");if(!identityMatches(record,doc))throw Error("片段听辨身份与当前录音不一致，未提供对照");entry.record=record;}
    catch(error){entry.record=null;entry.error=error.message;}
    finally{entry.loading=false;if(view&&key(view.session.doc)===entry.key)draw();}
  }
  function draw(){
    const host=document.getElementById("beLocalKeyReview");if(!host||!view||root.BarEditor.getContext()!==view.session)return;
    const {session,window}=view,entry=cache.get(key(session.doc));host.replaceChildren();
    const heading=node("div","section-heading"),caption=node("p","muted","选择一条已保存的听辨记录，仅在其覆盖范围内对照当前编辑轨。记录未独立验证，重叠记录不会自动合并或裁决。");
    const reload=node("button","secondary","刷新听辨记录");reload.type="button";reload.id="beKeyReviewRefresh";reload.disabled=entry.loading;reload.onclick=()=>refresh(entry,session.doc);heading.append(node("strong","","当前所选范围的片段记录"),reload);host.append(heading,caption);
    if(entry.loading){host.append(node("p","muted","正在读取听辨记录…"));return;}
    if(entry.error){host.append(node("p","status",entry.error+"。原始和弦与人工编辑不受影响。"));return;}
    const options=choices(entry.record,session.doc,window);
    if(!options.length){host.append(node("p","muted","此范围尚无已保存的听辨记录。可使用上方“记录所选片段调性”入口，听辨后刷新。"));return;}
    const select=node("select");select.id="beKeyFragment";select.setAttribute("aria-label","选择片段听辨记录");select.append(new Option("选择一条记录进行对照", ""));
    for(const event of options)select.append(new Option(`修订 ${event.revision} · ${event.label.excerpt_start}–${event.label.excerpt_end} 秒 · ${history.summary(event.label)}`,String(event.revision)));
    const chosen=options.find(event=>event.revision===entry.selected);if(!chosen)entry.selected=null;select.value=chosen?String(chosen.revision):"";
    select.onchange=()=>{entry.selected=select.value?Number(select.value):null;draw();};host.append(select);if(!chosen)return;
    const span=scope(chosen.label,window),tonality=localKey(chosen.label),info=node("p","be-local-key-context");info.id="beLocalKeyContext";
    info.textContent=`本次对照 ${span.start}–${span.end} 秒 · ${history.summary(chosen.label)} · 听辨修订 ${chosen.revision} · ${tonality?"级数依据该片段人工判断":"调性未明确，保留和弦名称"} · ${session.dirty?"当前未保存编辑草稿":"当前编辑轨"}`;host.append(info);
    const audition=node("button","secondary","试听此交集");audition.id="beLocalKeyListen";audition.type="button";audition.onclick=()=>{if(root.BarEditor.getContext()===session)root.BarEditor.auditionSpan(span.start,span.end);};host.append(audition);
    const table=node("table","be-local-key-table"),thead=node("thead"),tr=node("tr");for(const value of ["时间（秒）","编辑轨和弦","片段级数","来源"])tr.append(node("th","",value));thead.append(tr);table.append(thead);const tbody=node("tbody");
    const parts=root.BarEditor.logic.timelineParts(session.doc.events,span);
    for(const part of parts.slice(0,120)){
      const row=node("tr"),label=part.event?.label,parsed=label&&root.HarmonyNotation.parseInKey(label,tonality||"");
      row.append(node("td","",`${part.start}–${part.end}`),node("td","",label===undefined?"无覆盖":parsed?.chordDisplay||label),node("td","be-local-degree",tonality&&parsed?parsed.romanDisplay:"—"),node("td","",part.event?({model:"模型",manual:"人工",score:"曲谱采纳"}[part.event.origin]||part.event.origin):"无预测"));tbody.append(row);
    }
    table.append(tbody);const scroll=node("div","be-local-key-scroll");scroll.append(table);host.append(scroll);
    if(parts.length>120)host.append(node("p","muted",`当前共 ${parts.length} 个区间，只展示前 120 个；缩小所选小节范围可继续查看。`));
    host.append(node("p","muted","这里只读展示，不改写整曲主调、和弦内容或听辨标签；N 与未覆盖区间分开保留。"));
  }
  function show(session,window){view={session,window};const identity=key(session.doc);if(!cache.has(identity)){const entry={key:identity,record:null,error:"",loading:false,selected:null};cache.set(identity,entry);refresh(entry,session.doc);}else draw();}
  root.LocalKeyReview={show,logic};
})(typeof window!=="undefined"?window:globalThis);
