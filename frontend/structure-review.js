/* Read-only original-analysis loop evidence; never changes the edit track. */
(function(root){
  "use strict";
  function sourceInfo(result){
    const binding=result.structure?.harmony_source;
    if(binding){
      const source=result.sources?.find(item=>item.id===binding.source_id);
      const valid=!!source&&[binding.source_sha256,binding.run_id,binding.audio_sha256].every(value=>typeof value==="string"&&value.length>0)&&binding.source_sha256===source.sha256&&binding.run_id===result.run_id&&binding.audio_sha256===result.audio?.sha256;
      return {valid,name:source?.name||binding.source_id||"未知来源",text:valid?`循环来源：${source.name} · 原始预测，未随来源切换或人工修改重算。`:"循环来源身份无法核对；保留结果供查看，定位与试听已暂停。"};
    }
    const source=result.sources?.find(item=>item.id===result.default_source);
    return {valid:true,legacy:true,name:source?.name||"未知来源",text:`旧结果未保存循环来源绑定；原默认来源：${source?.name||"未知"}。切换来源或编辑和弦不会重算这些线索。`};
  }
  function loopInfo(loop,result){
    const valid=Number.isFinite(loop.start)&&Number.isFinite(loop.end)&&loop.start>=0&&loop.end>loop.start&&loop.end<=result.duration&&Number.isInteger(loop.period_bars)&&loop.period_bars>=1&&Number.isInteger(loop.repetitions)&&loop.repetitions>=2;
    const declaredFull=loop.match_basis==="full_bar_relative_harmonic_timeline";
    const patterns=Array.isArray(loop.bar_patterns)&&loop.bar_patterns.length===loop.period_bars&&loop.bar_patterns.every(rows=>Array.isArray(rows)&&rows.length&&rows[0].start_fraction===0&&rows.at(-1).end_fraction===1&&rows.every((span,i)=>typeof span.label==="string"&&Number.isFinite(span.start_fraction)&&Number.isFinite(span.end_fraction)&&span.start_fraction>=0&&span.end_fraction>span.start_fraction&&span.end_fraction<=1&&(!i||span.start_fraction===rows[i-1].end_fraction)))?loop.bar_patterns:null;
    const full=declaredFull&&!!patterns;
    const basis=full?"完整小节内进行匹配":declaredFull?"完整进行记录不完整":result.structure?.version==="1.0"?"旧版主和弦摘要匹配":"匹配依据未记录";
    let cycle=null;
    if(valid){
      const bars=result.bars||[],index=bars.findIndex(bar=>bar.bar===loop.start_bar&&bar.start===loop.start),selected=bars.slice(index,index+loop.period_bars);
      if(index>=0&&selected.length===loop.period_bars&&selected.every((bar,i)=>Number.isFinite(bar.start)&&Number.isFinite(bar.end)&&bar.end>bar.start&&(!i||bar.start===selected[i-1].end))&&selected.at(-1).end<=loop.end)cycle={start:loop.start,end:selected.at(-1).end};
    }
    return {valid,full,basis,patterns,cycle};
  }
  const logic={sourceInfo,loopInfo};
  if(typeof module!=="undefined"&&module.exports){module.exports=logic;return;}
  const node=(tag,cls,text)=>{const value=document.createElement(tag);if(cls)value.className=cls;if(text!==undefined)value.textContent=text;return value;};
  const seconds=t=>Number(t).toFixed(3).replace(/\.?0+$/,"");
  function render(result){
    const host=document.getElementById("loops"),note=document.getElementById("loopSourceNote"),loops=result.structure?.loops||[],source=sourceInfo(result);
    host.replaceChildren();note.hidden=!loops.length;note.textContent=source.text;
    for(const loop of loops){
      const info=loopInfo(loop,result),card=node("details","loop-evidence"),summary=node("summary"),heading=node("strong","",`${loop.period_bars} 小节 × ${loop.repetitions} 次`);
      summary.append(heading,node("span","loop-basis",info.basis),node("span","loop-time",`${seconds(loop.start)}–${seconds(loop.end)} 秒`));card.append(summary);
      const body=node("div","loop-evidence-body");
      body.append(node("p","muted",info.full?"相同的是原始标注中的完整进行和小节相对位置；仍需听辨和核对拍格。":"该线索未证明每个小节内的全部和弦相同，请逐段听辨核对。"));
      if(info.patterns){
        const list=node("ol","loop-patterns");
        for(const spans of info.patterns){const item=node("li");item.append(node("strong","",spans.map(span=>span.label).join(" → ")),node("small","",spans.map(span=>`${(span.start_fraction*100).toFixed(2).replace(/\.?0+$/,"")}–${(span.end_fraction*100).toFixed(2).replace(/\.?0+$/,"")}%`).join(" · ")));list.append(item);}
        body.append(list,node("p","muted","百分比表示每小节内的相对位置，不假定拍号或每拍等长。"));
      }else if(Array.isArray(loop.chords))body.append(node("p","loop-pattern-summary",loop.chords.join(" · ")));
      const actions=node("div","loop-actions"),locate=node("button","secondary loop-locate","定位整个范围"),listen=node("button","secondary loop-listen","试听一个周期"),status=node("p","status");status.setAttribute("role","status");
      const focus=(range,play)=>{const context=root.BarEditor?.getContext();if(!context||context.result!==result){status.textContent="当前编辑轨尚未就绪，请稍后重试。";return;}root.BarEditor.focusInterval(range.start,range.end,{listen:play});};
      locate.disabled=!source.valid||!info.valid;listen.disabled=!source.valid||!info.cycle;
      locate.onclick=()=>focus(loop,false);listen.onclick=()=>focus(info.cycle,true);
      actions.append(locate,listen);body.append(actions,status,node("p","muted","定位和试听只改变选区，不填入和弦，也不采纳循环。"));card.append(body);host.append(card);
    }
  }
  root.StructureReview={render,logic};
})(typeof window!=="undefined"?window:globalThis);
