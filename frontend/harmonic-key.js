"use strict";
// This panel is bound to the selected immutable model source. It never edits
// global/manual keys, chord corrections or the independent bar-editor review.
window.HarmonicKeyTimeline = (() => {
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const name=k=>k?k.replace(/m$/,"")+(k.endsWith("m")?" 小调":" 大调"):"未确定";
  const time=t=>`${String(Math.floor(t/60)).padStart(2,"0")}:${(t%60).toFixed(2).padStart(5,"0")}`;
  function render(result,source,seek){
    const panel=document.getElementById("harmonicKeyTimeline");if(!panel)return;
    panel.replaceChildren(el("h3","主调与转调解析"));
    const data=result?.harmonic_keys?.[source?.id];
    if(data && ((data.run_id && data.run_id!==result.run_id) || (data.source_sha256 && data.source_sha256!==source?.sha256))){
      panel.append(el("p","调性记录与当前和弦来源不匹配，请重新分析后查看。","muted"));return;
    }
    if(!data||data.status!=="computed"){
      panel.append(el("p",data?.reason||"此和弦来源尚无调性时间线。重新分析后生成。","muted"));return;
    }
    const summary=el("div",undefined,"tonal-summary");
    for(const [label,value] of [["和弦推理主调",name(data.main_key)],["开始调性",name(data.start_key)],["结束调性",name(data.end_key)],["转调候选",`${data.changes.length} 处`]]){
      const item=el("div");item.append(el("span",label,"field-label"),el("strong",value));summary.append(item);
    }
    panel.append(summary);
    panel.append(el("p",`依据：${data.source_name||source.name||source.id}。全曲主调按有效调区时长汇总；逐段比较和弦的调内关系、主和弦与属到主进行。`,"muted"));
    panel.append(el("p","点击区间定位试听，展开下方段落查看和弦依据。以下为模型推断，转调位置仍可结合原曲校对。","muted"));
    const strip=el("div",undefined,"tonal-strip");strip.setAttribute("aria-label","调性段落时间线");
    for(const s of data.segments){
      const button=el("button",`${time(s.start)}–${time(s.end)}　${name(s.key)}${s.status==="ambiguous"?" · 存在竞争解释":""}`,"secondary tonal-region");
      button.style.flexGrow=String(Math.max(1,s.end-s.start));button.onclick=()=>seek(s.start);strip.append(button);
    }
    panel.append(strip);
    if(!data.changes.length)panel.append(el("p","当前和弦序列未推断出独立转调段落。","tonal-change-note"));
    else{
      const changes=el("div",undefined,"tonal-changes");
      for(const change of data.changes){
        const jump=el("button",`${time(change.time)}　${name(change.from_key)} → ${name(change.to_key)}`,"text-button");
        jump.onclick=()=>seek(change.time);changes.append(jump);
      }
      panel.append(changes);
    }
    for(const [index,s] of data.segments.entries()){
      const details=el("details",undefined,"tonal-detail");
      details.append(el("summary",`${index+1}. ${time(s.start)}–${time(s.end)}　${name(s.key)}${s.start_bar!=null?` · 小节 ${s.start_bar}–${s.end_bar??"?"}`:""} · 查看推理依据`));
      const change=data.changes.find(c=>Math.abs(c.time-s.start)<1e-6);
      if(change)details.append(el("p",`${name(change.from_key)} → ${name(change.to_key)}：候选位置 ${time(change.time)}；相邻和弦范围 ${time(change.boundary_interval[0])}–${time(change.boundary_interval[1])}。这个范围不是统计置信区间。`));
      if(change?.first_distinctive_chord!=null)details.append(el("p",`首个新调调内、旧调非调内的和弦：${time(change.first_distinctive_chord)}${change.first_authentic_arrival!=null?`；属到主建立：${time(change.first_authentic_arrival)}`:"；尚无明确属到主到达"}。`));
      details.append(el("p",`竞争候选：${s.candidates.map(c=>`${name(c.key)}（${c.score.toFixed(2)}）`).join("、")}。分数仅用于排序，不是概率。`,"muted"));
      for(const kept of s.keep_neighbour_key||[]){
        const roles=[...new Set(kept.explanations.filter(c=>c.role!=="调内").map(c=>`${c.chord}：${c.role}`))];
        details.append(el("p",`若延续邻段 ${name(kept.key)}：${roles.join("；")||"和弦均可作调内解释，仍须比较主音建立"}。`));
      }
      const table=el("table",undefined,"key-experiment-table"),head=el("tr");
      for(const title of ["时间","模型和弦","此段级数","解释"])head.append(el("th",title));table.append(head);
      for(const item of s.evidence){
        const row=el("tr"),jump=el("button",time(item.start),"text-button");jump.onclick=()=>seek(item.start);
        const cell=el("td");cell.append(jump);row.append(cell,el("td",item.chord),el("td",window.HarmonyNotation.parseInKey(item.chord,s.key)?.romanDisplay||"—"),el("td",item.role+(item.arrival?" · 属到主":item.tonic?" · 主和弦":"")));table.append(row);
      }
      details.append(table);panel.append(details);
    }
    if(data.tonicizations.length){
      const details=el("details");details.append(el("summary",`局部主音化线索 · ${data.tonicizations.length} 处（未拆成独立调区）`));
      for(const t of data.tonicizations)details.append(el("p",`${time(t.start)}–${time(t.end)}：${t.chords.join(" → ")}，指向 ${name(t.target)}；上下文 ${name(t.context)}。`));panel.append(details);
    }
    if(data.excursions?.length){
      const details=el("details");details.append(el("summary",`借用与短段转调的竞争解释 · ${data.excursions.length} 处`));
      for(const e of data.excursions){
        const jump=el("button",`${time(e.start)}–${time(e.end)}　${e.chords.join(" → ")}`,"text-button");jump.onclick=()=>seek(e.start);details.append(jump);
        details.append(el("p",`当前延续 ${name(e.context)}；${e.independent_progression?"局部比较以下解释，不自动另立调区":"单个和弦不足以独立证明转调"}。`));
        for(const c of e.candidates)details.append(el("p",`${name(c.key)}（${c.score.toFixed(2)}）：${c.explanations.map(x=>`${x.chord} ${x.role}`).join("；")}`,"muted"));
      }
      panel.append(details);
    }
    if(data.uncovered?.length)panel.append(el("p",`无有效和弦证据的区间：${data.uncovered.map(g=>`${time(g.start)}–${time(g.end)}`).join("、")}；这些时长不计入主调汇总。`,"muted"));
    panel.append(el("p","此面板级数按各段调性解释。上方全曲级数及人工主调设置仍独立保留；修改和弦后，本面板仍代表原始模型来源。","muted"));
  }
  return {render};
})();
