/* Bar editor: the model timeline stays immutable; this document is a separate review track. */
(function (root) {
  "use strict";
  const clone = value => JSON.parse(JSON.stringify(value));
  let nextId = 0;
  const uid = prefix => prefix + "-" + Date.now().toString(36) + "-" + (++nextId);
  const overlap = (a,b) => a.start < b.end - 1e-7 && a.end > b.start + 1e-7;
  // Input values retain source precision; rounded labels are display-only.
  const timeInput = value => String(value);
  function timelineParts(events,window) {
    const result=[];
    let cursor=window.start;
    for(const event of events.filter(e=>e.start<window.end&&e.end>window.start)){
      const start=Math.max(event.start,window.start),end=Math.min(event.end,window.end);
      if(start>cursor)result.push({start:cursor,end:start,event:null});
      result.push({start,end,event});cursor=end;
    }
    if(cursor<window.end)result.push({start:cursor,end:window.end,event:null});
    return result;
  }
  function barIndexAt(bars,time) {
    if(!Number.isFinite(time))return -1;
    let low=0,high=bars.length;
    while(low<high){const middle=(low+high)>>>1;if(bars[middle].start<=time)low=middle+1;else high=middle;}
    const index=low-1;
    return index>=0&&time<bars[index].end?index:-1;
  }
  function rangeSourceEvidence(sources,window) {
    if(!Number.isFinite(window.start)||!Number.isFinite(window.end)||window.start<0||window.end<=window.start)throw new Error("来源对照范围无效");
    return sources.map(source=>{
      let previous=-Infinity;
      const valid=Array.isArray(source.rows)&&source.rows.every(row=>{
        const ok=Number.isFinite(row.start)&&Number.isFinite(row.end)&&row.start>=0&&row.end>row.start&&row.start>=previous&&typeof row.label==="string"&&row.label.length>0;
        previous=row.end;return ok;
      });
      if(!valid)return {source_id:source.id,name:source.name,valid:false,parts:[],covered_seconds:0};
      const indices=new Map(source.rows.map((row,i)=>[row,i]));
      const parts=timelineParts(source.rows,window).map(({start,end,event})=>({start,end,label:event?.label??null,source_index:event?indices.get(event):null}));
      return {source_id:source.id,name:source.name,valid:true,parts,covered_seconds:parts.reduce((total,p)=>total+(p.label===null?0:p.end-p.start),0)};
    });
  }
  function intervalSelection(bars,start,end){
    if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start||!bars.length)return null;
    const first=barIndexAt(bars,start);let lo=0,hi=bars.length;
    while(lo<hi){const mid=(lo+hi)>>>1;if(bars[mid].start<end)lo=mid+1;else hi=mid;}
    const last=lo-1;return first>=0&&last>=first&&end<=bars[last].end?[first,last]:null;
  }
  function replaceInterval(events, start, end, label, origin="manual", extra={}) {
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) throw new Error("起止时间无效");
    const updated = [];
    for (const event of events) {
      if (!overlap(event,{start,end})) { updated.push(clone(event)); continue; }
      if (event.start < start) updated.push({...clone(event),id:uid("event"),end:start});
      if (event.end > end) updated.push({...clone(event),id:uid("event"),start:end});
    }
    updated.push({id:uid("event"),start,end,label,origin,...extra});
    return updated.sort((a,b)=>a.start-b.start);
  }
  function moveBoundary(doc,index,value) {
    const result=clone(doc), bars=result.bars;
    if(index<=0||index>=bars.length||!Number.isFinite(value)||value<=bars[index-1].start||value>=bars[index].end) throw new Error("边界必须位于相邻两小节之间");
    bars[index-1].end=value;bars[index].start=value;
    bars[index-1].grid_origin="manual";bars[index].grid_origin="manual";
    // Beat ownership follows its time. This operation never shifts chords or audio.
    const times=[...bars[index-1].beats,...bars[index].beats];
    bars[index-1].beats=times.filter(t=>t>=bars[index-1].start&&t<value);
    bars[index].beats=times.filter(t=>t>=value&&t<bars[index].end);
    return result;
  }
  function reviseGrid(doc,index,fields) {
    const old=doc.bars[index];
    if(!old)throw new Error("请选择有效小节");
    if(String(fields.boundary).trim()===""||String(fields.number).trim()==="")throw new Error("小节起点和编号不能为空");
    const boundary=Number(fields.boundary),number=Number(fields.number);
    if(!Number.isFinite(boundary))throw new Error("小节起点需为有效秒数");
    if(index===0&&boundary!==old.start)throw new Error("音频起点固定，不能移动");
    const result=boundary!==old.start?moveBoundary(doc,index,boundary):clone(doc),bar=result.bars[index];
    if(!Number.isSafeInteger(number)||number<0)throw new Error("小节编号需为有效的非负整数");
    if(!["measure","pickup","unconfirmed"].includes(fields.kind))throw new Error("无效小节类型");
    bar.number=number;bar.kind=fields.kind;bar.grid_origin="manual";
    const meter=String(fields.meter).trim();
    if(meter){const match=meter.match(/^(\d{1,2})\/(1|2|4|8|16|32)$/);if(!match||Number(match[1])<1||Number(match[1])>32)throw new Error("拍号格式例如 3/4、6/8；未知请留空");bar.meter={numerator:Number(match[1]),denominator:Number(match[2])};}else bar.meter=null;
    const text=String(fields.beats).trim(),tokens=text?text.split(/[,，]/):[];
    if(tokens.some(t=>!t.trim()))throw new Error("拍点不能包含空值");
    const entered=tokens.map(Number),unchanged=entered.length===old.beats.length&&entered.every((t,i)=>t===old.beats[i]);
    // Moving a boundary automatically transfers beat ownership. Untouched input
    // must not restore the old beat list and discard those transferred beats.
    const beats=unchanged?bar.beats:entered;
    if(beats.length>128||beats.some((t,i)=>!Number.isFinite(t)||t<bar.start||t>=bar.end||(i>0&&t<=beats[i-1])))throw new Error("拍点需递增并位于此小节内，使用绝对秒数");
    bar.beats=beats;
    return result;
  }
  function renumberBars(doc,from) {
    const number=doc.bars[from].number;
    if(!Number.isSafeInteger(number)||number<0||!Number.isSafeInteger(number+doc.bars.length-from-1))throw new Error("小节编号超出有效范围");
    for(let i=from;i<doc.bars.length;i++)doc.bars[i].number=number+i-from;
  }
  function splitBar(doc,index,time,renumber=true) {
    const old=doc.bars[index];
    if(!old||!Number.isFinite(time)||time<=old.start||time>=old.end)throw new Error("新边界必须位于所选小节内部");
    if(doc.bars.length>=10000)throw new Error("小节数已达到上限");
    const result=clone(doc),left=result.bars[index],right={...clone(left),id:uid("bar"),start:time,kind:"measure"};
    left.end=time;left.beats=old.beats.filter(t=>t<time);right.beats=old.beats.filter(t=>t>=time);
    left.meter=null;right.meter=null;left.grid_origin="manual";right.grid_origin="manual";
    result.bars.splice(index+1,0,right);
    for(const section of result.sections)if(section.end_bar_id===old.id)section.end_bar_id=right.id;
    if(renumber)renumberBars(result,index);
    return result;
  }
  function mergeBars(doc,from,to,renumber=true) {
    if(!Number.isInteger(from)||!Number.isInteger(to)||from<0||to<=from||to>=doc.bars.length)throw new Error("请选择至少两个相邻小节");
    const order=new Map(doc.bars.map((bar,i)=>[bar.id,i]));
    const blocked=doc.sections.filter(section=>{
      const start=order.get(section.start_bar_id),end=order.get(section.end_bar_id);
      return (start>from&&start<=to)||(end>=from&&end<to);
    });
    if(blocked.length)throw new Error("所选范围跨越段落边界："+blocked.map(s=>s.name).join("、")+"。请先调整或移除这些段落标注。");
    const result=clone(doc),selected=result.bars.slice(from,to+1),first=selected[0],ids=new Set(selected.map(b=>b.id));
    first.end=selected[selected.length-1].end;first.beats=selected.flatMap(b=>b.beats);first.meter=null;first.grid_origin="manual";
    result.bars.splice(from,to-from+1,first);
    for(const section of result.sections){if(ids.has(section.start_bar_id))section.start_bar_id=first.id;if(ids.has(section.end_bar_id))section.end_bar_id=first.id;}
    if(renumber)renumberBars(result,from);
    return result;
  }
  function copyAnchors(source,dest,mode){
    if(mode==="relative")return [[source.start,dest.start],[source.end,dest.end]];
    if(mode!=="beats")throw Error("无效的复制对齐方式");
    for(const bar of [source,dest]){
      if(bar.grid_origin!=="manual"||bar.kind!=="measure"||!bar.meter)throw Error("按拍点复制需要先人工校正两侧完整小节的拍点和拍号；弱起或待确认区间不能直接对应");
      if(!Array.isArray(bar.beats)||!bar.beats.length||bar.beats[0]!==bar.start||bar.beats.some((t,i)=>!Number.isFinite(t)||t<bar.start||t>=bar.end||i>0&&t<=bar.beats[i-1]))throw Error("两侧拍点须递增、从小节起点开始，并位于各自小节内");
    }
    if(source.meter.numerator!==dest.meter.numerator||source.meter.denominator!==dest.meter.denominator||source.beats.length!==dest.beats.length)throw Error("两侧拍号与拍点个数须一致；不能自动推断跨拍号或不同拍层级的对应");
    return [...source.beats.map((time,index)=>[time,dest.beats[index]]),[source.end,dest.end]];
  }
  function copyTime(time,anchors){
    for(let i=0;i<anchors.length;i++){
      if(time===anchors[i][0])return anchors[i][1];
      if(time<anchors[i][0]&&i>0){const left=anchors[i-1],right=anchors[i];return left[1]+(time-left[0])/(right[0]-left[0])*(right[1]-left[1]);}
    }
    throw Error("复制区间超出已核对的时间锚点");
  }
  function copyBars(doc,from,to,target,mode="relative") {
    const count=to-from+1;
    if(![from,to,target].every(Number.isInteger)||from<0||to<from||to>=doc.bars.length||target<0||target+count>doc.bars.length)throw new Error("目标范围超出歌曲小节");
    if(target<=to&&target+count>from)throw new Error("复制目标不能与源范围重叠");
    if(!["relative","beats"].includes(mode))throw Error("无效的复制对齐方式");
    const anchors=Array.from({length:count},(_,i)=>copyAnchors(doc.bars[from+i],doc.bars[target+i],mode));
    const snapshot=clone(doc.events),result=clone(doc),changed=[];
    for(let i=0;i<count;i++){
      const source=doc.bars[from+i],dest=doc.bars[target+i];
      // Replace the full target bar, including explicit gaps in the copied source.
      const outside=[];
      for(const event of result.events){
        if(!overlap(event,dest)){outside.push(event);continue;}
        if(event.start<dest.start)outside.push({...event,id:uid("event"),end:dest.start});
        if(event.end>dest.end)outside.push({...event,id:uid("event"),start:dest.end});
      }
      result.events=outside;
      const scale=(dest.end-dest.start)/(source.end-source.start);
      const mapped=time=>mode==="relative"?dest.start+(time-source.start)*scale:copyTime(time,anchors[i]);
      for(const event of snapshot.filter(e=>overlap(e,source))){
        const start=mapped(Math.max(event.start,source.start));
        const end=mapped(Math.min(event.end,source.end));
        result.events=replaceInterval(result.events,start,end,event.label,"manual");
        changed.push({start,end,label:event.label});
      }
    }
    return {doc:result,changed,alignment:{mode,anchors}};
  }
  function mergeSavedDocument(doc,saved,unchanged){
    if(unchanged)return clone(saved);
    const result=clone(doc);result.revision=saved.revision;result.saved_at=saved.saved_at;
    for(const event of result.events){
      const stored=saved.events.find(e=>e.id===event.id);
      const keys=["start","end","label","origin","reference_id","reference_event_id","alignment_confirmed"];
      if(stored&&stored.reference_revision!==undefined&&keys.every(k=>stored[k]===event[k]))event.reference_revision=stored.reference_revision;
    }
    return result;
  }
  function readBrowserDraft(raw,saved) {
    const result={raw,doc:null,updated_at:"时间未知",problem:"草稿格式无法读取"};
    let wrapper;try{wrapper=JSON.parse(raw);}catch(_){return result;}
    const doc=wrapper?.doc;if(!doc||typeof doc!=="object")return result;
    result.doc=doc;if(typeof wrapper.updated_at==="string")result.updated_at=wrapper.updated_at;
    const identity=["schema","project_id","run_id","source_id","source_sha256","audio_sha256","duration"];
    if(identity.some(key=>saved[key]===undefined||doc[key]!==saved[key])){result.problem="草稿的项目、分析、来源、音频或时长与当前记录不一致";return result;}
    if(!Number.isSafeInteger(doc.revision)||doc.revision<0)return result;
    if(!Array.isArray(doc.bars)||!doc.bars.length||doc.bars.length>10000||!Array.isArray(doc.events)||doc.events.length>20000||!Array.isArray(doc.sections)||doc.sections.length>1000)return result;
    const token=value=>typeof value==="string"&&/^[A-Za-z0-9_-]{1,100}$/.test(value);
    const barIds=new Map();let previous=0;
    for(const bar of doc.bars){
      if(!bar||!token(bar.id)||barIds.has(bar.id)||!Number.isFinite(bar.start)||!Number.isFinite(bar.end)||Math.abs(bar.start-previous)>1e-6||bar.end<=bar.start||bar.end>doc.duration+1e-6||!Number.isSafeInteger(bar.number)||bar.number<0||!["pickup","measure","unconfirmed"].includes(bar.kind)||!Array.isArray(bar.beats)||bar.beats.length>128)return result;
      if(bar.beats.some((beat,i)=>!Number.isFinite(beat)||beat<bar.start||beat>=bar.end||i>0&&beat<=bar.beats[i-1]))return result;
      if(bar.meter!=null&&(!Number.isInteger(bar.meter.numerator)||bar.meter.numerator<1||bar.meter.numerator>32||![1,2,4,8,16,32].includes(bar.meter.denominator)))return result;
      barIds.set(bar.id,barIds.size);previous=bar.end;
    }
    if(Math.abs(previous-doc.duration)>1e-6)return result;
    const sectionIds=new Set();
    for(const section of doc.sections){
      if(!section||!token(section.id)||sectionIds.has(section.id)||typeof section.name!=="string"||!section.name.trim()||section.name.length>80||!barIds.has(section.start_bar_id)||!barIds.has(section.end_bar_id)||barIds.get(section.start_bar_id)>barIds.get(section.end_bar_id))return result;
      sectionIds.add(section.id);
    }
    const eventIds=new Set();previous=0;
    for(const event of doc.events){
      if(!event||!token(event.id)||eventIds.has(event.id)||!Number.isFinite(event.start)||!Number.isFinite(event.end)||event.start<0||event.start<previous-1e-6||event.end<=event.start||event.end>doc.duration+1e-6||typeof event.label!=="string"||!event.label.length||event.label.length>160||/\s/.test(event.label)||!["model","manual","score"].includes(event.origin))return result;
      eventIds.add(event.id);previous=event.end;
    }
    result.problem="";return result;
  }
  const logic={replaceInterval,copyBars,moveBoundary,reviseGrid,splitBar,mergeBars,overlap,mergeSavedDocument,timeInput,timelineParts,rangeSourceEvidence,barIndexAt,readBrowserDraft,intervalSelection};
  if(typeof module!=="undefined"&&module.exports){module.exports=logic;return;}
  const $=id=>document.getElementById(id);
  const node=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  const seconds=timeInput;
  const time=n=>`${Math.floor(n/60).toString().padStart(2,"0")}:${(n%60).toFixed(2).padStart(5,"0")}`;
  const name=label=>root.HarmonyNotation?.parse(label)?.chordDisplay||label;
  const displayMode=()=>{const b=$("displayRoman");return b&&b.getAttribute("aria-pressed")==="true"?"roman":"name";};
  function chordLabel(label){
    if(!label)return"";
    const parsed=root.HarmonyNotation?.parse(label);
    if(!parsed)return label;
    if(displayMode()==="roman"&&parsed.romanDisplay&&$("keySelect")?.value)return parsed.romanDisplay;
    return parsed.chordDisplay||label;
  }
  function setDetailVisible(on){
    for(const id of["beEvents","beEditFields","beModelCandidates"]){const el=$(id);if(el)el.classList.toggle("be-detail-hidden",!on);}
    const hint=$("beDetailHint");if(hint)hint.classList.toggle("be-detail-hidden",on);
  }
  const origins={model:"模型",manual:"人工",score:"曲谱采纳"};
  const sessions=new Map();let current=null,opened=0,frame=null;
  async function request(url,options={}){
    const r=await fetch(url,{...options,headers:{"X-Harmony-Client":"studio",...options.headers}});
    if(!r.ok){let detail;try{detail=(await r.json()).detail;}catch(_){detail=r.statusText;}const e=new Error(typeof detail==="string"?detail:JSON.stringify(detail));e.status=r.status;throw e;}return r.json();
  }
  function mount(){
    if($("barWorkspace"))return;
    const host=node("section","panel bar-workspace");host.id="barWorkspace";
    host.innerHTML=`<div class="be-heading"><div><div class="eyebrow">BAR BY BAR · HUMAN REVIEW</div><h3>按小节听辨与修正</h3><p id="beTrack" class="muted"></p></div><div class="be-history"><button id="beUndo" class="secondary" title="撤销 Ctrl+Z">↶ 撤销</button><button id="beRedo" class="secondary" title="重做 Ctrl+Shift+Z">↷ 重做</button></div></div>
      <div id="beSave" class="be-save" role="status" aria-live="polite">正在读取小节编辑记录…</div>
      <div class="be-export"><button id="beExportJson" class="text-button">导出编辑轨 JSON ↓</button><button id="beExportLab" class="text-button">导出编辑轨 LAB ↓</button><span class="muted">导出当前编辑轨草稿 · 不代表服务器已保存或已验证准确</span></div>
      <div id="beBrowserBackups" class="be-browser-backups"></div><div id="beRecovery" class="be-recovery" hidden><p id="beRecoveryText"></p><button id="beRestoreDraft" class="secondary">恢复此浏览器草稿</button><button id="beExport" class="secondary">导出当前草稿</button><button id="beReload" class="secondary">重新读取服务器版本</button></div>
      <div class="be-toolbar"><strong id="beRange">选择小节</strong><button id="beListen" class="secondary">试听范围</button><button id="beLoop" class="secondary" aria-pressed="false">循环范围</button><a id="beKeyListen" class="text-link" target="_blank" rel="noopener">记录所选片段调性 ↗</a><button id="beDisplayToggle" class="secondary" aria-pressed="true" title="切换和弦标记：级数 / 和弦名">级数</button><span id="beNow" class="muted"></span><span class="be-legend"><i class="model"></i>模型 <i class="manual"></i>人工 <i class="score"></i>曲谱参考</span></div>
      <div id="beSections" class="be-sections" aria-label="人工命名歌曲段落"></div>
      <div class="be-section-form"><label>段落名称<input id="beSectionName" placeholder="例如：主歌 1 / 副歌 / 间奏" maxlength="80"></label><button id="beSectionAdd" class="secondary">标注所选范围</button><label>复制到<select id="beCopyTarget" aria-label="复制目标起始小节"></select></label><label>对齐方式<select id="beCopyMode" aria-label="复制和声对齐方式"><option value="relative">小节相对时间</option><option value="beats">人工校正的对应拍点</option></select></label><button id="beCopy" class="secondary">复制和声…</button></div>
      <p class="be-help">点击左侧小节号选择小节（Shift + 点击扩展范围）；点击任一和弦方框查看该和弦的详情并修改。方框按真实和弦变化切分，长短与持续时长成正比；播放时只高亮当前和弦，页面不会自动滚动。</p>
      <div id="beGrid" class="be-grid" role="group" aria-label="歌曲小节编辑网格"></div>
      <div class="be-inspector"><section id="beChordDetail"><div class="section-heading"><h4 id="beEditHeading">所选小节</h4><span class="muted">局部替换，保留前后和弦</span></div><p id="beDetailHint" class="be-detail-hint">点击上方任一和弦方框，查看它的时间、来源与可编辑内容。</p><div id="beEvents" class="be-events"></div><div id="beEditFields" class="be-edit-fields"><label>起点（秒）<input id="beStart" type="number" step="0.001" min="0"></label><label>终点（秒）<input id="beEnd" type="number" step="0.001" min="0"></label><label>和弦<input id="beChord" placeholder="C、Am7、G/B" autocomplete="off"></label><button id="beListenEdit" class="secondary">试听此区间</button><button id="beApply" class="primary">应用局部修改</button></div><p id="beEditStatus" class="status" role="status"></p><div id="beModelCandidates" class="be-model-candidates"></div></section>
      <details class="be-timing"><summary>校正小节 / 起拍与拍号</summary><p class="muted">自动边界需听辨核对；第一个强拍之前可能是前奏、自由节奏或弱起，先保留待确认。修改边界不移动和弦。</p><div class="be-edit-fields"><label>小节起点（秒）<input id="beBoundary" type="number" step="0.001"></label><label>小节编号<input id="beNumber" type="number" step="1"></label><label>类型<select id="beKind"><option value="unconfirmed">待确认区间</option><option value="measure">完整小节</option><option value="pickup">弱起</option></select></label><label>拍号<input id="beMeter" placeholder="待确认；例如 3/4 或 6/8"></label><label class="be-beats-field">拍点（绝对秒，逗号分隔）<input id="beBeats" placeholder="0.5, 1.0, 1.5"></label><button id="beTimingApply" class="secondary">应用网格校正</button></div><div class="be-grid-actions"><label>新小节边界（秒）<input id="beSplitTime" type="number" step="0.001"></label><button id="beUsePlayhead" class="secondary">使用播放位置</button><button id="beSplit" class="secondary">拆分小节…</button><button id="beMerge" class="secondary">合并所选小节…</button><label class="be-renumber"><input id="beRenumber" type="checkbox" checked>从所选起点重新顺序编号</label></div><p class="muted">拆分、合并保留和弦和拍点的绝对时间，新小节拍号需重新确认。合并不能跨越已标注的段落边界。</p></details></div>
      <details class="be-source-comparison"><summary>来源差异定位 · 双来源听辨</summary><div id="beSourceComparison"></div></details><details class="be-local-key"><summary>片段调性与和声对照 · 人工记录</summary><div id="beLocalKeyReview"></div></details><details class="be-score-ref"><summary>曲谱参考候选 · 单独核对与对齐</summary><p class="muted">印刷吉他谱需要核对和弦名、小节顺序、变调夹、指法调和实音调；曲谱候选不自动写入人工轨。</p><div id="beScoreTools"></div><div id="beScoreStatus" class="status" role="status"></div><div id="beScoreCandidates"></div></details>
      <dialog id="beCopyDialog"><h3>确认复制和声</h3><p id="beCopyPreview"></p><details id="beCopyAlignmentDetails" hidden><summary>查看对应的时间点（源 → 目标）</summary><pre id="beCopyAnchors" class="be-copy-anchors"></pre></details><p id="beCopyMethodNote" class="muted">按每小节相对时间复制；节拍和版本不同需要听辨复核。目标人工修改会被替换，操作可撤销。</p><div class="be-dialog-actions"><button id="beCopyCancel" class="secondary">取消</button><button id="beCopyConfirm" class="primary">确认复制</button></div></dialog><dialog id="beGridDialog"><h3 id="beGridTitle"></h3><p id="beGridPreview"></p><p class="muted">和弦时间与已标注段落范围保持不变。受影响小节拍号设为待确认，首小节类型保留，可继续校正。操作可撤销。</p><div class="be-dialog-actions"><button id="beGridCancel" class="secondary">取消</button><button id="beGridConfirm" class="primary">确认网格修改</button></div></dialog>`;
    $("sources").after(host);
    const setup=node("details","be-analysis-setup");setup.id="beAnalysisSetup";setup.open=true;setup.append(node("summary","","音频与分析设置 · 更换音乐 / 重新分析"));$("inputPanel").before(setup);setup.append($("inputPanel"));
    const reference=node("details","be-analysis-reference");reference.append(node("summary","","原始时间轴、自动段落与主调辅助证据"));host.after(reference);
    for(const content of [document.querySelector(".overview"),document.querySelector(".key-review"),$("keyEvidence")])if(content)reference.append(content);
    // Keep the original interval comparison available without competing with the bar workspace.
    const old=document.querySelector(".review-layout"),details=node("details","be-original");details.append(node("summary","","原始区间与来源比较（独立旧修订记录）"));old.before(details);details.append(old);
    $("beUndo").onclick=()=>history(-1);$("beRedo").onclick=()=>history(1);
    $("beListen").onclick=()=>playRange(false);$("beLoop").onclick=()=>{if(current?.loop)stopLoop();else playRange(true);};
    $("beApply").onclick=applyChord;$("beTimingApply").onclick=applyTiming;
    $("beListenEdit").onclick=auditionEdit;
    $("beUsePlayhead").onclick=()=>{if(current)$("beSplitTime").value=seconds(current.audio.currentTime);};
    $("beDisplayToggle").onclick=()=>{const roman=displayMode()==="roman",target=$(roman?"displayName":"displayRoman");if(target)target.click();render();};
    $("beSplit").onclick=()=>prepareGrid("split");$("beMerge").onclick=()=>prepareGrid("merge");
    $("beGridCancel").onclick=()=>$("beGridDialog").close();$("beGridConfirm").onclick=confirmGrid;
    $("beSectionAdd").onclick=addSection;$("beCopy").onclick=prepareCopy;
    $("beCopyTarget").onchange=()=>{if(current)current.copyTargetId=$("beCopyTarget").value?current.doc.bars[Number($("beCopyTarget").value)]?.id||null:null;};
    $("beCopyMode").onchange=()=>{if(current)current.copyMode=$("beCopyMode").value;};
    $("beCopyCancel").onclick=()=>$("beCopyDialog").close();$("beCopyConfirm").onclick=confirmCopy;
    $("beExport").onclick=()=>exportDraft(current);$("beReload").onclick=reload;
    $("beExportJson").onclick=()=>exportTrack(false);$("beExportLab").onclick=()=>exportTrack(true);
    $("beRestoreDraft").onclick=restoreDraft;
    document.addEventListener("keydown",e=>{if(!current||!e.target.closest("#barWorkspace")||e.target.closest("input,textarea,select"))return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){e.preventDefault();history(e.shiftKey?1:-1);}});
    window.addEventListener("beforeunload",e=>{if([...sessions.values()].some(s=>s.dirty)){e.preventDefault();e.returnValue="";}});
  }
  const storageKey=s=>`harmony-bar-editor:${s.project.id}:${s.doc.run_id}:${s.doc.source_id}:${s.doc.source_sha256}:${s.doc.audio_sha256}`;
  function message(text){$("beEditStatus").textContent=text;}
  function draftText(s){return JSON.stringify({doc:s.doc,version:s.version,updated_at:new Date().toISOString()});}
  function archiveRaw(s,raw,kind="preserved"){
    if(raw===null||raw===undefined)return;
    if(kind==="memory"&&s.archivedVersion===s.version&&s.archivedRevision===s.doc.revision)return;
    if(kind==="preserved"&&s.archivedRaw===raw)return;
    const key=storageKey(s)+":backup:"+new Date().toISOString()+":"+kind+":"+crypto.randomUUID();
    localStorage.setItem(key,raw);if(localStorage.getItem(key)!==raw)throw Error("浏览器备份无法验证，请先导出当前草稿。");
    if(kind==="memory"){s.archivedVersion=s.version;s.archivedRevision=s.doc.revision;}else s.archivedRaw=raw;
  }
  function observeDraft(s,raw){
    if(raw===s.browserRaw)return false;
    clearTimeout(s.timer);s.conflict=true;
    try{if(s.dirty)archiveRaw(s,draftText(s),"memory");if(s.cached&&s.cached.raw!==raw)archiveRaw(s,s.cached.raw);}catch(error){s.backupError=error.message;}
    if(raw!==null)s.cached=readBrowserDraft(raw,s.doc);
    else if(!s.cached)s.cached={raw:"",problem:"浏览器暂存已在另一窗口移除，需要重新读取项目版本。"};
    status(s,"另一窗口改变了浏览器草稿 · 自动保存已暂停；当前内容与保留原文可分别导出。");
    if(s===current)renderRecovery();return true;
  }
  function storageCurrent(s){try{if(observeDraft(s,localStorage.getItem(storageKey(s))))return false;}catch(_){}return !s.cached;}
  function stash(s){
    try{if(!storageCurrent(s))return false;const raw=draftText(s);localStorage.setItem(storageKey(s),raw);s.browserRaw=raw;return true;}
    catch(_){if(s===current)status(s,"浏览器暂存失败，请导出草稿；服务器保存仍会尝试。");return false;}
  }
  function renderBackups(s){
    const host=$("beBrowserBackups");if(!host||s!==current)return;host.replaceChildren();
    try{const prefix=storageKey(s)+":backup:",keys=[];for(let i=0;i<localStorage.length;i++){const key=localStorage.key(i);if(key?.startsWith(prefix))keys.push(key);}keys.sort().reverse();if(!keys.length)return;
      const details=node("details"),summary=node("summary","",`此编辑轨的浏览器备份（${keys.length}）`);details.append(summary);
      for(const key of keys){const button=node("button","text-button",key.slice(prefix.length,prefix.length+24)+" · 导出备份");button.onclick=()=>downloadBarText(localStorage.getItem(key)||"","bar_editor_backup.json");details.append(button);}host.append(details);
    }catch(_){}
  }
  function downloadBarText(text,filename){const url=URL.createObjectURL(new Blob([text],{type:"application/json"})),a=node("a");a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  function status(s,text){s.status=text;if(s===current)$("beSave").textContent=text;}
  function editable(doc){return {bars:clone(doc.bars),sections:clone(doc.sections),events:clone(doc.events)};}
  function snapshot(s){return {document:editable(s.doc),selection:[...s.selection],anchor:s.anchor};}
  function fitSelection(s){const last=s.doc.bars.length-1;s.selection=s.selection.map(i=>Math.max(0,Math.min(last,i)));s.anchor=Math.max(0,Math.min(last,s.anchor));}
  function commit(next,label,selection){const s=current;if(!s)return false;if(!storageCurrent(s)||s.cached){message("请先恢复浏览器草稿，或导出后重新读取服务器版本；原草稿尚未被覆盖。");return false;}s.undo.push(snapshot(s));if(s.undo.length>80)s.undo.shift();s.redo=[];Object.assign(s.doc,editable(next));if(selection){s.selection=selection;s.anchor=selection[0];}fitSelection(s);changed(s,label);render();return true;}
  function changed(s,label){s.version++;s.dirty=true;const retained=stash(s);status(s,s.conflict?"保存冲突 · 当前草稿与另一窗口原文保留；请导出或重新载入":retained?"已暂存 · 将自动保存："+label:"浏览器暂存失败 · 将尝试保存到项目，请及时导出草稿");clearTimeout(s.timer);if(!s.conflict)s.timer=setTimeout(()=>save(s),800);}
  function history(direction){const s=current;if(!s)return;if(!storageCurrent(s)||s.cached)return message("请先处理待恢复的浏览器草稿；原草稿尚未被覆盖。");const src=direction<0?s.undo:s.redo,dest=direction<0?s.redo:s.undo;if(!src.length)return;dest.push(snapshot(s));const previous=src.pop();Object.assign(s.doc,previous.document);s.selection=previous.selection;s.anchor=previous.anchor;fitSelection(s);stopLoop();changed(s,direction<0?"撤销":"重做");render();}
  async function save(s){
    if(s.saving||!s.dirty||s.conflict||!storageCurrent(s))return;s.saving=true;const version=s.version,payload=clone(s.doc);status(s,"正在保存小节修订…");
    try{const saved=await request(`/api/projects/${s.project.id}/editor`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
      s.dirty=s.version!==version;s.doc=mergeSavedDocument(s.doc,saved,!s.dirty);
      if(s.dirty){
        const retained=stash(s);status(s,s.cached?"上一轮修改已保存 · 另一窗口草稿保留，自动保存暂停":retained?"已保存上一轮修改 · 新修改待保存":"上一轮已保存，新的浏览器暂存失败；请导出草稿");
      }else{let incoming=false;try{const raw=localStorage.getItem(storageKey(s));incoming=observeDraft(s,raw)||!!s.cached;if(!incoming){localStorage.removeItem(storageKey(s));s.browserRaw=null;}}catch(_){}
        status(s,incoming?`当前修订 ${saved.revision} 已保存 · 另一窗口草稿仍保留，请核对后恢复或重新读取`:`已自动保存 · 修订 ${saved.revision} · 人工复核尚未独立验证`);}
    }catch(e){if(e.status===409){s.conflict=true;status(s,"保存冲突："+e.message+"。当前草稿保留，自动保存已暂停。");}else{status(s,"保存失败："+e.message+"。草稿保留；点击任意修改会重试。");}stash(s);
    }finally{s.saving=false;if(s===current)renderRecovery();if(s.dirty&&!s.conflict&&s.version!==version)s.timer=setTimeout(()=>save(s),800);}
  }
  async function open(context){
    mount();const token=++opened;stopLoop();current=null;$("beGrid").replaceChildren();$("beSave").textContent="正在读取小节编辑记录…";$("barWorkspace").hidden=false;
    const key=`${context.project.id}:${context.result.run_id}:${context.source.id}`;
    let s=sessions.get(key);
    if(!s){try{const doc=await request(`/api/projects/${context.project.id}/editor/${context.source.id}`);s={...context,doc,undo:[],redo:[],version:0,dirty:false,conflict:false,saving:false,browserRaw:null,selection:[0,0],anchor:0,status:`已读取修订 ${doc.revision} · 模型与人工轨分别保存`};sessions.set(key,s);
      try{const raw=localStorage.getItem(storageKey(s));s.browserRaw=raw;if(raw!==null){s.cached=readBrowserDraft(raw,doc);s.status=s.cached.problem?"浏览器草稿未恢复："+s.cached.problem:"发现此浏览器的未保存草稿；请选择恢复或导出。";}}catch(_){}
    }catch(e){if(token===opened)$("beSave").textContent="小节工作区读取失败："+e.message;return;}}
    if(token!==opened)return;current=s;Object.assign(s,context);document.body.classList.add("has-bar-editor");$("beAnalysisSetup").open=false;$("loop").textContent="循环所选小节";render();
    if(!s.audio.dataset.barEditorListening){s.audio.dataset.barEditorListening="true";s.audio.addEventListener("play",()=>{if(current&&!frame)frame=requestAnimationFrame(playFrame);});s.audio.addEventListener("timeupdate",()=>{if(current){paintPlayback(current,current.audio.currentTime);const i=barIndexAt(current.doc.events,current.audio.currentTime),e=i>=0?current.doc.events[i]:null;$("beNow").textContent=e?`编辑轨：${name(e.label)} · ${origins[e.origin]||e.origin}`:"编辑轨：无覆盖";}});s.audio.addEventListener("seeked",()=>{if(current)paintPlayback(current,current.audio.currentTime);});}
    if(!s.audio.paused&&!frame)frame=requestAnimationFrame(playFrame);root.BarEditor.onOpen?.(context);
  }
  function close(){++opened;stopLoop();current=null;document.body.classList.remove("has-bar-editor");$("loop").textContent="循环原区间";if($("barWorkspace"))$("barWorkspace").hidden=true;if($("beAnalysisSetup"))$("beAnalysisSetup").open=true;}
  function selected(){const s=current;return s.doc.bars.slice(s.selection[0],s.selection[1]+1);}
  function range(){const bars=selected();return {start:bars[0].start,end:bars[bars.length-1].end};}
  function select(index,extend=false){const s=current;stopLoop();if(extend)s.selection=[Math.min(index,s.anchor),Math.max(index,s.anchor)];else{s.anchor=index;s.selection=[index,index];}renderSelection();renderInspector();}
  function render(){const s=current;if(!s)return;fitSelection(s);
    const roman=displayMode()==="roman",toggle=$("beDisplayToggle");if(toggle){toggle.textContent=roman?"级数":"和弦名";toggle.setAttribute("aria-pressed",String(roman));toggle.title=roman?"当前显示级数标记 · 点击切换为和弦名":"当前显示和弦名 · 点击切换为级数标记";}
    $("beTrack").textContent=`修订基于 ${s.source.name} · 小节边界可校正 · ${s.doc.bars.length} 小节`;
    $("beUndo").disabled=!s.undo.length;$("beRedo").disabled=!s.redo.length;$("beSave").textContent=s.status;
    const frag=document.createDocumentFragment();s.doc.bars.forEach((bar,index)=>{
      const cell=node("div","be-bar");cell.dataset.index=index;
      const header=node("button","be-bar-head");header.type="button";
      header.title=`${bar.kind==="pickup"?"弱起":bar.kind==="unconfirmed"?"待确认（拍号未知）":"完整小节"} · ${bar.meter?`${bar.meter.numerator}/${bar.meter.denominator}`:"拍号待确认"} · 点击选择，Shift + 点击扩展范围`;
      header.append(node("b","be-bar-no",bar.kind==="pickup"?`弱起 ${bar.number}`:bar.kind==="unconfirmed"?`${bar.number}?`:`${bar.number}`),
                    node("small","be-bar-time",time(bar.start)));
      header.onclick=e=>{s.activeChord=null;select(index,e.shiftKey);if(!e.shiftKey)s.audio.currentTime=bar.start;};
      cell.append(header);
      const strip=node("div","be-chords");strip.setAttribute("role","group");strip.setAttribute("aria-label",`第 ${bar.number} 小节和弦`);
      for(const part of timelineParts(s.doc.events,bar)){
        const {start:a,end:b,event}=part,duration=Math.max(.001,b-a);
        const button=node("button",event?`be-chord-cell ${event.origin||"model"}`:"be-chord-cell be-chord-gap");
        // Width follows the real duration; a small floor keeps 0.05 s fragments readable.
        button.style.flexGrow=String(Math.max(.3,duration));
        button.append(node("span","be-chord-name",event?chordLabel(event.label):"—"));
        button.dataset.start=String(a);button.dataset.end=String(b);
        if(event?.label==="N")button.dataset.n="1";
        button.title=`${time(a)}–${time(b)} · ${duration.toFixed(2)} 秒`+(event?` · ${origins[event.origin]||event.origin} · ${event.label}`:" · 该区间没有和弦预测（不等同于 N）");
        button.setAttribute("aria-label",button.title);
        button.onclick=ev=>{
          s.activeChord={start:a,end:b,label:event?.label||"",origin:event?.origin||null,index};
          select(index,ev.shiftKey);
          if(!ev.shiftKey)s.audio.currentTime=a;
          for(const other of document.querySelectorAll(".be-chord-cell.active"))other.classList.remove("active");button.classList.add("active");
          // Only an explicit click may move the page; playback never does.
          const detail=$("beChordDetail");
          if(detail&&detail.getBoundingClientRect().bottom>innerHeight)detail.scrollIntoView({block:"nearest"});
        };
        strip.append(button);
      }
      const active=s.activeChord;
      if(active&&active.index===index){for(const el of strip.children){if(Number(el.dataset.start)===active.start&&Number(el.dataset.end)===active.end){el.classList.add("active");break;}}}
      cell.append(strip);frag.append(cell);
    });$("beGrid").replaceChildren(frag);s.playingIndex=null;s.playingCell=null;paintPlayback(s,s.audio.currentTime);
    $("beCopyTarget").replaceChildren(new Option("请选择目标小节",""),...s.doc.bars.map((b,i)=>new Option((b.kind==="pickup"?"弱起 ":"小节 ")+b.number,String(i))));
    const copyTarget=s.doc.bars.findIndex(bar=>bar.id===s.copyTargetId);$("beCopyTarget").value=copyTarget>=0?String(copyTarget):"";$("beCopyMode").value=s.copyMode||"relative";
    const sections=s.doc.sections.map(section=>{const start=s.doc.bars.findIndex(b=>b.id===section.start_bar_id),end=s.doc.bars.findIndex(b=>b.id===section.end_bar_id),wrap=node("span","be-section-chip"),button=node("button","",`${section.name} · ${s.doc.bars[start]?.number}–${s.doc.bars[end]?.number}`),remove=node("button","be-section-remove","×");button.onclick=()=>{s.anchor=start;s.selection=[start,end];$("beSectionName").value=section.name;renderSelection();renderInspector();};remove.title="移除此段落标注";remove.setAttribute("aria-label","移除段落 "+section.name);remove.onclick=()=>{const next=clone(s.doc);next.sections=next.sections.filter(x=>x.id!==section.id);commit(next,"移除段落");};wrap.append(button,remove);return wrap;});
    $("beSections").replaceChildren(...sections);if(!sections.length)$("beSections").append(node("span","muted","选择主歌或副歌的小节范围，然后标注段落名称。"));renderSelection();renderInspector();renderRecovery();
    const playingIndex=barIndexAt(s.doc.events,s.audio.currentTime),playingEvent=playingIndex>=0?s.doc.events[playingIndex]:null;$("beNow").textContent=playingEvent?`编辑轨：${name(playingEvent.label)} · ${origins[playingEvent.origin]||playingEvent.origin}`:"编辑轨：无覆盖";
    root.dispatchEvent(new Event("harmony-editor-updated"));
  }
  function renderSelection(){if(!current)return;const s=current,bars=selected();$("beKeyListen").href=root.KeyListeningHandoff.url(s.doc,bars[0].start,bars[bars.length-1].end);root.LocalKeyReview?.show(s,range());root.SourceComparison?.show(s);$("beRange").textContent=bars.length===1?`已选小节 ${bars[0].number}`:`已选 ${bars[0].number}–${bars[bars.length-1].number} · ${bars.length} 小节`;for(const cell of $("beGrid").children){const i=Number(cell.dataset.index),on=i>=s.selection[0]&&i<=s.selection[1];cell.classList.toggle("selected",on);cell.querySelector("button").setAttribute("aria-pressed",String(on));}}
  function renderInspector(){const s=current;if(!s)return;const bars=selected(),first=bars[0],window=range();$("beEditHeading").textContent=`小节 ${bars.map(b=>b.number).join("、")} · ${time(window.start)}–${time(window.end)}`;
    const active=s.activeChord;setDetailVisible(!!active);
    if(active){$("beStart").value=seconds(active.start);$("beEnd").value=seconds(active.end);$("beChord").value=active.label||"";}
    else{$("beStart").value=seconds(window.start);$("beEnd").value=seconds(window.end);$("beChord").value="";}
    const chips=s.doc.events.filter(e=>overlap(e,window)).map(event=>{const a=Math.max(event.start,window.start),b=Math.min(event.end,window.end),button=node("button",`be-event-chip ${event.origin||"model"}`,`${name(event.label)} · ${time(a)}–${time(b)} · ${origins[event.origin]||event.origin}`);button.onclick=()=>{$("beStart").value=seconds(a);$("beEnd").value=seconds(b);$("beChord").value=event.label;};return button;});$("beEvents").replaceChildren(...chips);
    $("beBoundary").value=seconds(first.start);$("beBoundary").disabled=s.selection[0]===0;$("beNumber").value=first.number;$("beKind").value=first.kind;$("beMeter").value=first.meter?`${first.meter.numerator}/${first.meter.denominator}`:"";$("beBeats").value=first.beats.map(seconds).join(", ");
    $("beSplitTime").value=seconds((first.start+first.end)/2);$("beSplit").disabled=bars.length!==1;$("beMerge").disabled=bars.length<2;
    renderSourceEvidence(s,window);message("修改起止秒可只替换半个小节或某一拍；和弦轨与小节边界独立保存。");
  }
  function renderSourceEvidence(s,window){const host=$("beModelCandidates");host.replaceChildren(node("h5","","比较所选范围的原始预测"),node("p","muted","点选只填入该来源与所选范围的交集，可先试听，再应用为人工修订。来源之间可能相关；这里没有置信概率，也不按全曲主调删除候选。"));
    for(const source of rangeSourceEvidence(s.result.sources,window)){
      const line=node("div","be-source-line"),heading=node("div","be-source-heading");heading.append(node("strong","",source.name||source.source_id),node("small","muted",source.valid?`预测覆盖 ${source.covered_seconds.toFixed(3)} / ${(window.end-window.start).toFixed(3)} 秒${source.source_id===s.source.id?" · 当前来源":""}`:"时间数据异常，暂停提供该来源候选"));line.append(heading);
      const list=node("div","be-source-parts");for(const part of source.parts){const gap=part.label===null,button=node("button",gap?"be-source-gap":"be-source-choice",gap?"无覆盖":name(part.label));button.dataset.source=source.source_id;button.dataset.label=part.label??"";button.dataset.start=String(part.start);button.dataset.end=String(part.end);button.append(node("small","",`${part.start.toFixed(3)}–${part.end.toFixed(3)} 秒`));button.title=gap?"此来源在该区间没有预测；不同于 N":`${source.name||source.source_id} · 原标签 ${part.label} · 不是置信度`;button.onclick=()=>{$("beStart").value=seconds(part.start);$("beEnd").value=seconds(part.end);$("beChord").value=part.label||"";s.audio.currentTime=part.start;message(gap?"已选择未覆盖区间；可试听后自行标注，未自动填入 N。":`已填入 ${source.name||source.source_id} 的 ${part.label} 候选；试听后点击应用才会写入人工轨。`);};list.append(button);}line.append(list);host.append(line);
    }
  }
  function applyChord(){try{const s=current,label=$("beChord").value.trim(),direct=root.HarmonyNotation.parse(label),parsed=direct||root.HarmonyNotation.parseEditable(label);if(!parsed)throw new Error("请输入可解析的和弦名称；级数记法需要已确定的主调");const normalized=direct?label:parsed.chordDisplay,start=Number($("beStart").value),end=Number($("beEnd").value),r=range();if(start<r.start-1e-6||end>r.end+1e-6)throw new Error("修改区间必须位于所选小节范围内");const next=clone(s.doc);next.events=replaceInterval(next.events,start,end,normalized);s.activeChord={start,end,label:normalized,origin:"manual",index:s.selection[0]};if(commit(next,"和弦局部修改"))message(`已应用 ${name(normalized)} · ${time(start)}–${time(end)}；可以撤销。`);}catch(e){message(e.message);}}
  function applyTiming(){try{const s=current,next=reviseGrid(s.doc,s.selection[0],{boundary:$("beBoundary").value,number:$("beNumber").value,kind:$("beKind").value,meter:$("beMeter").value,beats:$("beBeats").value});stopLoop();if(commit(next,"网格校正"))message("已校正网格；拍点按绝对时间归属相邻小节。和弦时间保持原位置，请重新试听边界。");}catch(e){message(e.message);}}
  let pendingGrid=null;
  function prepareGrid(kind){try{const s=current,[from,to]=s.selection,renumber=$("beRenumber").checked;let doc,description,selection;
    if(kind==="split"){if(from!==to)throw new Error("拆分时只选择一个小节");const value=$("beSplitTime").value;if(!value.trim())throw new Error("请输入新边界的秒数");const time=Number(value);doc=splitBar(s.doc,from,time,renumber);description=`将小节 ${s.doc.bars[from].number} 在 ${seconds(time)} 秒拆为两个小节。`;selection=[from,from+1];}
    else{doc=mergeBars(s.doc,from,to,renumber);description=`将所选 ${to-from+1} 个小节合并为一个，范围 ${time(s.doc.bars[from].start)}–${time(s.doc.bars[to].end)}。`;selection=[from,from];}
    pendingGrid={session:s,version:s.version,doc,selection};$("beGridTitle").textContent=kind==="split"?"确认拆分小节":"确认合并小节";$("beGridPreview").textContent=description+(renumber?` 从小节 ${s.doc.bars[from].number} 起重新顺序编号；后续段落引用随小节 ID 保留。`:kind==="split"?" 保留原编号；新拆出的两个小节暂用同一编号，可随后手动调整。":" 保留首小节编号，后续编号保持原值。");$("beGridDialog").showModal();
  }catch(e){message(e.message);}}
  function confirmGrid(){if(!pendingGrid)return;const p=pendingGrid;pendingGrid=null;$("beGridDialog").close();if(p.session!==current||p.version!==current.version)return message("草稿已变化，请重新预览网格修改");stopLoop();if(commit(p.doc,"拆分或合并小节",p.selection))message("小节网格已更新；请试听并确认拍号。原和弦和段落时间已保留，可撤销。");}
  function addSection(){const s=current,name=$("beSectionName").value.trim();if(!name)return message("请先填写主歌、副歌或其他段落名称");const bars=selected(),next=clone(s.doc),existing=next.sections.find(x=>x.start_bar_id===bars[0].id&&x.end_bar_id===bars[bars.length-1].id);if(existing)existing.name=name;else next.sections.push({id:uid("section"),name,start_bar_id:bars[0].id,end_bar_id:bars[bars.length-1].id});if(commit(next,"段落标注"))message(`已标注“${name}”；点击段落带可重新选中范围。`);}
  let pendingCopy=null;
  function prepareCopy(){try{if(!$("beCopyTarget").value)throw Error("请明确选择复制目标小节");const s=current,[from,to]=s.selection,target=Number($("beCopyTarget").value),mode=$("beCopyMode").value,proposal=copyBars(s.doc,from,to,target,mode),dest=s.doc.bars.slice(target,target+to-from+1),window={start:dest[0].start,end:dest[dest.length-1].end},conflicts=s.doc.events.filter(e=>e.origin!=="model"&&overlap(e,window));pendingCopy={session:s,version:s.version,proposal};
    const meter=b=>b.meter?`${b.meter.numerator}/${b.meter.denominator}`:"拍号未确认",differences=[];
    dest.forEach((b,i)=>{const a=s.doc.bars[from+i],lengthA=a.end-a.start,lengthB=b.end-b.start;if(meter(a)!==meter(b)||Math.abs(lengthA-lengthB)>.001)differences.push(`${a.number}→${b.number}：${meter(a)} / ${lengthA.toFixed(3)}秒 → ${meter(b)} / ${lengthB.toFixed(3)}秒`);});
    $("beCopyPreview").textContent=`${mode==="beats"?"按人工校正的对应拍点分段复制":"按小节内相对时间复制"}：小节 ${s.doc.bars[from].number}–${s.doc.bars[to].number} → ${dest[0].number}–${dest[dest.length-1].number}；复制 ${proposal.changed.length} 个区间。目标已有 ${conflicts.length} 个人工或曲谱采纳区间。${differences.length?" 拍号或时长差异："+differences.join("；")+"。需要试听复核。":" 拍点对应仍需试听复核。"}`;
    $("beCopyAlignmentDetails").hidden=mode!=="beats";
    $("beCopyAnchors").textContent=proposal.alignment.anchors.map((pairs,i)=>`小节 ${s.doc.bars[from+i].number} → ${dest[i].number}：\n`+pairs.map(([a,b])=>`${seconds(a)} → ${seconds(b)} 秒`).join("\n")).join("\n\n");
    $("beCopyMethodNote").textContent=(mode==="beats"?"只在对应拍点之间线性映射和弦边界；拍点对应不证明两段和声相同，需试听目标段落。":"按每小节相对时间复制；节拍和版本不同需要听辨复核。")+"目标人工修改会被替换，操作可撤销。";
    $("beCopyDialog").showModal();}catch(e){message(e.message);}}
  function confirmCopy(){if(!pendingCopy)return;const p=pendingCopy;pendingCopy=null;$("beCopyDialog").close();if(p.session!==current||p.version!==current.version)return message("草稿已变化，请重新预览复制");if(commit(p.proposal.doc,"复制段落和声"))message("已复制所选小节的和声；请听辨复核目标范围。");}
  function stopLoop(){if(current)current.loop=null;if(frame)cancelAnimationFrame(frame);frame=null;if($("beLoop"))$("beLoop").setAttribute("aria-pressed","false");if($("loop"))$("loop").setAttribute("aria-pressed","false");if(current&&!current.audio.paused)frame=requestAnimationFrame(playFrame);}
  function playWindow(r,repeat){const s=current;if(!s)return;stopLoop();s.onRangePlayback?.();s.loop={...r,repeat};s.audio.currentTime=r.start;$("beLoop").setAttribute("aria-pressed",String(repeat));$("loop").setAttribute("aria-pressed",String(repeat));s.audio.play().then(()=>{if(current===s&&!frame)frame=requestAnimationFrame(playFrame);}).catch(e=>message("播放失败："+e.message));}
  function playRange(repeat){if(current)playWindow(range(),repeat);}
  function focusInterval(start,end,options={}){
    const s=current;if(!s)return false;const selection=intervalSelection(s.doc.bars,start,end);if(!selection)return false;
    let audition;try{audition=root.SourceComparison.logic.auditionRange(start,end,options.context??0,s.doc.duration);}catch(e){message(e.message);return false;}
    stopLoop();s.anchor=selection[0];s.selection=selection;s.activeChord={start,end,label:"",origin:"来源比较",index:selection[0]};renderSelection();renderInspector();
    $("beStart").value=seconds(start);$("beEnd").value=seconds(end);$("beChord").value="";s.audio.currentTime=start;
    message(`已定位来源比较区间 ${seconds(start)}–${seconds(end)} 秒；未填入和弦。`+(options.listen?` 试听 ${seconds(audition.start)}–${seconds(audition.end)} 秒。`:""));
    if(options.listen)playWindow(audition,false);return true;
  }
  function auditionEdit(){try{if(!current)return;const a=$("beStart").value,b=$("beEnd").value,start=Number(a),end=Number(b),r=range();if(!a.trim()||!b.trim()||!Number.isFinite(start)||!Number.isFinite(end)||start<r.start||end>r.end||end<=start)throw new Error("试听区间须位于所选小节内，并有有效起止秒数");playWindow({start,end},false);}catch(e){message(e.message);}}
  // Highlight only: the grid never scrolls itself, so reading another bar is not interrupted.
  function paintPlayback(s,time){
    if(s!==current)return;const grid=$("beGrid");if(!grid)return;
    const index=barIndexAt(s.doc.bars,time);
    if(index!==s.playingIndex){
      if(Number.isInteger(s.playingIndex)&&s.playingIndex>=0)grid.children[s.playingIndex]?.classList.remove("playing");
      s.playingIndex=index;
      if(index>=0)grid.children[index]?.classList.add("playing");
    }
    const strip=index>=0?grid.children[index]?.querySelector(".be-chords"):null;
    let cell=null;
    for(const el of strip?.children||[]){if(time>=Number(el.dataset.start)-1e-6&&time<Number(el.dataset.end)){cell=el;break;}}
    if(cell===s.playingCell)return;
    s.playingCell?.classList.remove("playing");s.playingCell=cell;cell?.classList.add("playing");
  }
  function playFrame(){frame=null;const s=current;if(!s)return;const t=s.audio.currentTime,loop=s.loop;if(loop&&t>=loop.end-.008){if(loop.repeat){s.audio.currentTime=loop.start;if(s.audio.paused)s.audio.play().catch(e=>message(e.message));}else{s.audio.pause();s.loop=null;}}
    paintPlayback(s,s.audio.currentTime);
    if(!s.audio.paused)frame=requestAnimationFrame(playFrame);
  }
  function renderRecovery(){const s=current;renderBackups(s);$("beRecovery").hidden=!(s.conflict||s.cached||s.dirty&&s.status.startsWith("保存失败"));$("beRestoreDraft").hidden=!s.cached;$("beRestoreDraft").disabled=!!s.cached?.problem;$("beExport").textContent=s.cached?"导出保留的浏览器草稿":"导出当前草稿";$("beRecoveryText").textContent=s.cached?.problem?s.cached.problem+"。当前编辑轨保持不变；该原文保留，可完整导出后重新读取。":s.conflict?"跨窗口或服务器修订发生变化，自动保存已暂停。当前编辑轨与保留原文可分别导出；重新读取会先保留浏览器备份。":s.cached?`浏览器草稿：${s.cached.updated_at} · 修订 ${s.cached.doc.revision}。恢复或重新读取之前暂停修改，避免覆盖未处理草稿。`:"未保存的修改仍保留在此浏览器。";if(s.backupError)$("beRecoveryText").textContent+=" 浏览器备份失败，请分别导出当前编辑轨与保留原文。";}
  function restoreDraft(){
    const s=current,cached=s.cached;if(!cached)return;if(cached.problem)return message(cached.problem+"；请先导出保留的草稿。");
    try{const raw=localStorage.getItem(storageKey(s));if(raw!==cached.raw){observeDraft(s,raw);throw Error("浏览器草稿又发生变化，请先核对保留内容。");}
      if(s.dirty)archiveRaw(s,draftText(s),"memory");archiveRaw(s,cached.raw);
    }catch(error){message(error.message);return;}
    s.browserRaw=cached.raw;s.cached=null;
    if(cached.doc.revision!==s.doc.revision){s.conflict=true;s.doc=clone(cached.doc);s.dirty=true;s.version++;stash(s);status(s,"草稿与服务器修订不同 · 自动保存暂停；请导出并人工对照");fitSelection(s);render();return;}
    s.conflict=false;commit(cached.doc,"恢复浏览器草稿");
  }
  function exportDraft(s){if(s)downloadBarText(s.cached?s.cached.raw:JSON.stringify(s.doc,null,2),"bar_editor_draft.json");}
  function exportTrack(lab){const s=current;if(!s)return;const doc={...clone(s.doc),human_edited_unverified:true,export_state:s.dirty?"unsaved_current_draft":"saved_document_copy",exported_at:new Date().toISOString()};const text=lab?doc.events.map(e=>`${e.start.toFixed(6)}\t${e.end.toFixed(6)}\t${e.label}`).join("\n")+"\n":JSON.stringify(doc,null,2),url=URL.createObjectURL(new Blob([text],{type:lab?"text/plain;charset=utf-8":"application/json"})),a=node("a");a.href=url;a.download="edited_unverified."+(lab?"lab":"json");a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message("已导出当前编辑轨草稿；导出不会代替服务器保存，当前人工结果未经独立验证。");}
  async function reload(){
    const s=current;if(!s)return;if((s.dirty||s.cached)&&!confirm("先保留浏览器备份，再重新读取服务器版本。界面将显示服务器内容。继续？"))return;
    try{
      if(s.saving)throw Error("当前保存请求尚未结束，请稍后重载");
      const key=storageKey(s),raw=localStorage.getItem(key);
      if(s.dirty)archiveRaw(s,draftText(s),"memory");if(s.cached)archiveRaw(s,s.cached.raw);archiveRaw(s,raw);
      const doc=await request(`/api/projects/${s.project.id}/editor/${s.source.id}`);
      if(s.saving)throw Error("当前保存请求尚未结束，请稍后重载");
      const actual=localStorage.getItem(key);if(actual!==raw){observeDraft(s,actual);throw Error("重新读取期间另一窗口又改变了草稿，原文与备份保留，请核对。");}
      clearTimeout(s.timer);localStorage.removeItem(key);s.browserRaw=null;s.doc=doc;s.undo=[];s.redo=[];s.cached=null;s.dirty=false;s.conflict=false;s.backupError="";s.version++;
      status(s,`已读取服务器修订 ${doc.revision} · 浏览器备份可在上方导出`);if(current===s)render();
    }catch(e){message(e.message);if(current===s)renderRecovery();}
  }
  root.addEventListener("storage",event=>{for(const s of sessions.values())if(event.key===storageKey(s)||event.key===null){try{observeDraft(s,localStorage.getItem(storageKey(s)));}catch(_){}}});
  root.BarEditor={open,close,logic,focusInterval,refresh(){if(current)render();},getContext:()=>current,toggleLoop(){if(current?.loop)stopLoop();else playRange(true);},stopLoop,auditionSpan(start,end){if(!current)return false;const r=range();if(!Number.isFinite(start)||!Number.isFinite(end)||start<r.start||end>r.end||end<=start)return false;playWindow({start,end},false);return true;},playbackAt(time){if(!current)return null;const index=barIndexAt(current.doc.events,time),event=index>=0?current.doc.events[index]:null;return event?{label:event.label,origin:origins[event.origin]||event.origin}:{label:null,origin:"编辑轨无覆盖"};},adoptCandidate(candidate){const s=current;if(!s)return false;try{const next=clone(s.doc);next.events=replaceInterval(next.events,candidate.start,candidate.end,candidate.label,"score",{reference_id:candidate.reference_id,reference_event_id:candidate.reference_event_id,alignment_confirmed:true});return commit(next,"显式采纳曲谱候选");}catch(e){message(e.message);return false;}},setScoreContent(renderFn){if($("beScoreTools"))renderFn($("beScoreTools"),current);}};
})(typeof window!=="undefined"?window:globalThis);
