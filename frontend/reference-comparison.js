/* Display imported score annotations against the unchanged prediction track. */
(function(root){
  "use strict";
  const node=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  function label(value){
    const p=root.HarmonyNotation?.parse(value);
    const roman=document.getElementById("displayRoman")?.getAttribute("aria-pressed")==="true";
    return (roman&&document.getElementById("keySelect")?.value?p?.romanDisplay:p?.chordDisplay)||p?.chordDisplay||value;
  }
  function decorate(){
    const s=root.BarEditor?.getContext(),grid=document.getElementById("beGrid");
    if(!grid)return;
    const imported=s?.doc.bars.some(b=>Number.isInteger(b.reference_score_bar));
    let banner=document.getElementById("referenceComparisonNotice");
    if(!imported){if(banner)banner.hidden=true;return;}
    if(!document.getElementById("referenceComparisonStyles")){
      const style=node("style");style.id="referenceComparisonStyles";
      style.textContent=".xr-notice{margin:10px 0;padding:12px 15px;border:1px solid #a28b5d;border-radius:9px;background:#302b20;color:#eddbb7;font-size:12px;line-height:1.7}.xr-notice p{margin:3px 0}.xr-notice strong{font-size:14px}.xr-tracks{flex:1;min-width:0;padding:3px 0}.xr-track-label{font-size:10px;color:#c5aa76;margin:1px 0 3px}.xr-model-label{color:#8aab98;margin-top:7px}.xr-model{display:flex;gap:3px;min-height:30px}.xr-model-cell{min-width:0;flex-basis:0;padding:4px 2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border:1px solid #3e5d4b;border-radius:4px;background:#15251c;color:#b5c7bb;font-size:12px;cursor:pointer}.xr-model-cell:hover{background:#294335}.xr-model-gap{border-style:dashed;color:#82968b}.xr-original-grid{display:block;font-size:9px;color:#b7a578;padding:2px}.xr-tracks>.be-chords{width:100%;box-sizing:border-box}.xr-unannotated{font-size:10px;color:#b7a578}";
      document.head.append(style);
    }
    if(!banner){banner=node("div","xr-notice");banner.id="referenceComparisonNotice";grid.before(banner);}
    banner.hidden=false;
    const count=s.doc.bars.filter(b=>Number.isInteger(b.reference_score_bar)).length;
    banner.replaceChildren(node("strong","",`曲谱参考已载入 · ${count} 小节 · 与模型逐小节对照`),
      node("p","","上排：谱面参考／当前修订　下排：原始模型输出。小节编号已按原曲合并，每小节四拍。"),
      node("p","","时间按原界面两格合一投影，待听校；开头与尾部未标注区间留空。点击和弦可定位试听并继续修订。"));
    [...grid.children].forEach((cell,i)=>{
      if(cell.querySelector(".xr-tracks"))return;
      const bar=s.doc.bars[i],strip=cell.querySelector(".be-chords");if(!bar||!strip)return;
      const tracks=node("div","xr-tracks");strip.before(tracks);
      tracks.append(node("div","xr-track-label",bar.reference_score_bar?`参考／当前修订 · 谱面 ${bar.reference_score_bar} · 第 ${bar.reference_page} 页`:"参考未标注"),strip);
      const ids=bar.original_ui_bars;
      if(ids)cell.querySelector(".be-bar-head")?.append(node("small","xr-original-grid",`原格 ${ids[0]}–${ids[1]}`));
      tracks.append(node("div","xr-track-label xr-model-label","模型 · "+s.source.name));
      const comparison=node("div","xr-model");comparison.setAttribute("aria-label",`小节 ${bar.number} 原始模型预测`);
      const parts=root.BarEditor.logic.timelineParts(s.source.rows,bar);
      for(const part of parts){
        const r=part.event,a=part.start,b=part.end;
        const button=node("button","xr-model-cell"+(r?"":" xr-model-gap"),r?label(r.label):"未覆盖");
        button.type="button";button.style.flexGrow=String(b-a);
        button.title=`原始模型 ${r?.label||"未覆盖"} · ${a.toFixed(3)}–${b.toFixed(3)} 秒；点击定位，不修改标签`;
        button.onclick=()=>root.BarEditor.focusInterval(a,b);
        comparison.append(button);
      }
      tracks.append(comparison);
    });
  }
  root.addEventListener("harmony-editor-updated",decorate);
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",decorate);else decorate();
})(window);
