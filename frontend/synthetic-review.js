/* Read-only audio/specification comparison. No application API or storage writes. */
(function () {
  "use strict";
  const isNumber = value => typeof value === "number" && Number.isFinite(value);
  const digest = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
  const core = {
    validateCase(item, entry) {
      if (!item || item.schema !== "harmony_listening_case_v1" || item.id !== entry.id ||
          !isNumber(item.duration) || item.duration <= 0 || item.duration > 1800 || item.duration !== entry.duration ||
          !digest(item.audio_sha256) || item.audio !== item.audio_sha256 + ".wav" ||
          !Number.isInteger(item.sample_rate) || item.sample_rate < 8000 || item.sample_rate > 192000 ||
          item.domain !== entry.domain || item.split !== entry.split || item.truth_kind !== entry.truth_kind ||
          !Array.isArray(item.tracks) || !item.tracks.length) throw Error("样例身份或音频信息不一致");
      const ids = new Set();
      for (const track of item.tracks) {
        if (ids.has(track.id) || !Array.isArray(track.rows)) throw Error("来源记录重复或缺失");
        ids.add(track.id); let end = 0;
        for (const row of track.rows) {
          if (!isNumber(row.start) || !isNumber(row.end) || row.start < end - 1e-8 || row.end <= row.start ||
              row.end > item.duration + 1e-8 || typeof row.label !== "string") throw Error("和弦时间轴无效");
          end = row.end;
        }
        if (item.truth_kind === "none" && (track.kind === "specification" || "spec_match_pct" in track))
          throw Error("无真值样例不能包含规格评分");
      }
      return item;
    },
    range(start, end, context, duration) {
      if (![start,end,context,duration].every(isNumber) || start < 0 || end <= start || end > duration + 1e-8 || context < 0 || context > 2)
        throw Error("请输入录音内有效的起止时间");
      return {start:Math.max(0,start-context), end:Math.min(duration,end+context)};
    },
    selectionFromURL(search,index) {
      if(typeof search!=="string"||search.length>2000)throw Error("区间链接过长或无效");
      const params=new URLSearchParams(search),keys=["case","case_sha256","start","end","context"];
      if(!keys.some(key=>params.has(key)))return null;
      if(keys.some(key=>params.getAll(key).length>1))throw Error("区间链接包含重复参数");
      const id=params.get("case"),entry=index.cases.find(row=>row.id===id);
      if(!entry)throw Error("区间链接指向未知样例");
      if(!digest(params.get("case_sha256"))||params.get("case_sha256")!==entry.sha256)throw Error("区间链接的样例版本不一致，未恢复范围");
      function seconds(key,fallback){const value=params.get(key);if(value===null&&fallback!==undefined)return fallback;
        if(value===null||!/^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(value))throw Error("区间链接的时间无效");
        const number=Number(value);if(!isNumber(number))throw Error("区间链接的时间无效");return number;}
      const start=seconds("start"),end=seconds("end"),context=seconds("context",0);
      if(![0,.5,1,2].includes(context))throw Error("区间链接的上下文长度无效");
      core.range(start,end,context,entry.duration);
      return {id,case_sha256:entry.sha256,start,end,context};
    },
    selectionURL(base,entry,start,end,context) {
      if(!entry||!digest(entry.sha256))throw Error("样例版本尚未核验");
      core.range(start,end,context,entry.duration);
      if(![0,.5,1,2].includes(context))throw Error("上下文长度无效");
      const url=new URL(base);url.search="";url.hash="";
      for(const [key,value] of Object.entries({case:entry.id,case_sha256:entry.sha256,start,end,context}))url.searchParams.set(key,String(value));
      return url.href;
    },
    pairedEntries(item,index) {
      if(item?.truth_kind!=="authored_compositional_specification"||!Number.isInteger(item.transpose)||!item.family)return [];
      return index.cases.filter(entry=>entry.id!==item.id&&entry.timbre!==item.timbre&&entry.family===item.family&&
        entry.transpose===item.transpose&&entry.split===item.split&&entry.duration===item.duration&&
        entry.truth_kind===item.truth_kind&&["procedural","sampled_piano_holdout"].includes(entry.domain));
    },
    sameComposition(a,b) {
      const domains=["procedural","sampled_piano_holdout"];
      if(!a||!b||a.truth_kind!=="authored_compositional_specification"||b.truth_kind!==a.truth_kind||
        !domains.includes(a.domain)||!domains.includes(b.domain)||!a.family||a.family!==b.family||
        !Number.isInteger(a.transpose)||a.transpose!==b.transpose||a.split!==b.split||
        !["train","validation","test"].includes(a.split)||a.duration!==b.duration||a.sample_rate!==b.sample_rate)return false;
      function canonical(value){if(Array.isArray(value))return value.map(canonical);
        if(value&&typeof value==="object")return Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])]));return value;}
      function specification(item){
        if(!["bars","notes","local_keys","sections"].every(key=>Array.isArray(item[key]))||!item.notes.length||!item.bars.length)return null;
        const tracks=item.tracks?.filter(track=>track.kind==="specification");
        if(tracks?.length!==1||!Array.isArray(tracks[0].rows)||!tracks[0].rows.length)return null;
        return canonical({bars:item.bars,notes:item.notes,local_keys:item.local_keys,sections:item.sections,harmony:tracks[0].rows});
      }
      const left=specification(a),right=specification(b);
      return left!==null&&right!==null&&JSON.stringify(left)===JSON.stringify(right);
    },
    validateRhythm(data,item,entry) {
      if(!data||data.schema!=="harmony_rhythm_review_v1"||data.id!==item.id||data.audio_sha256!==item.audio_sha256||
        data.duration!==item.duration||data.case_sha256!==entry.case_sha256||entry.audio_sha256!==item.audio_sha256||item.truth_kind==="none"||
        data.local_key_is_decoded_timeline!==false||!Array.isArray(data.key_windows)||!Array.isArray(data.predicted_beats))throw Error("调性/节拍记录与当前样例身份不一致");
      for(const window of data.key_windows)if(!isNumber(window.start)||!isNumber(window.end)||window.start<0||window.end<=window.start||window.end>item.duration+1e-5)throw Error("调性窗口范围无效");
      for(const beat of data.predicted_beats)if(!isNumber(beat.time)||beat.time<0||beat.time>item.duration+.100001||!Number.isInteger(beat.beat)||beat.beat<1)throw Error("拍点记录无效");
      return data;
    },
    validateLocalBeats(data,item,entry) {
      if(!data||data.schema!=="harmony_local_beat_review_v1"||data.id!==item.id||data.audio_sha256!==item.audio_sha256||
        entry.audio_sha256!==item.audio_sha256||data.case_sha256!==entry.case_sha256||data.duration!==item.duration||item.truth_kind==="none"||
        data.default_changed!==false||data.best_origin_selected!==false||data.local_meter_verified!==false||
        !Array.isArray(data.variants)||data.variants.length!==2||!Array.isArray(data.baseline_beats))throw Error("局部节拍记录与当前样例身份不一致");
      function beats(values){let previous=-1;for(const beat of values){if(!isNumber(beat.time)||beat.time<0||beat.time>=item.duration||beat.time<=previous||!Number.isInteger(beat.beat)||beat.beat<1||beat.beat>6)throw Error("局部拍点时间轴无效");previous=beat.time;}}
      function metrics(value){for(const level of ["written_denominator","compound_dotted_quarter","downbeats"])if(!isNumber(value?.[level]?.f1)||value[level].f1<0||value[level].f1>1)throw Error("局部节拍指标无效");}
      beats(data.baseline_beats);metrics(data.baseline_metrics);
      for(const [index,variant] of data.variants.entries()){
        if(variant.id!==["origin_0s","origin_2s"][index]||variant.origin_seconds!==index*2||variant.applied!==false||variant.continuity_verified!==false||
          !Array.isArray(variant.beats)||!Array.isArray(variant.windows)||!variant.windows.length||!Array.isArray(variant.seams))throw Error("局部窗口范围或试验状态无效");
        beats(variant.beats);metrics(variant.metrics);
        for(const window of variant.windows)if(!isNumber(window.start)||!isNumber(window.end)||window.start<0||window.end<=window.start||window.end>item.duration||![3,4,6].includes(window.model_count))throw Error("局部解码窗口越界");
        for(const seam of variant.seams)if(!isNumber(seam.time)||seam.time<=0||seam.time>=item.duration||seam.boundary_verified!==false)throw Error("局部接缝不能视为确认边界");
      }
      return data;
    }
  };
  if (typeof module !== "undefined") module.exports = core;
  if (typeof document === "undefined") return;
  const $ = id => document.getElementById(id);
  const audio = $("audio"), prefix = "/assets/diagnostic-data/v1/";
  if (window.matchMedia("(max-width: 640px)").matches) $("windowSize").value="8";
  const state = {index:null,item:null,epoch:0,controller:null,blob:null,ready:false,windowStart:0,selected:null,playingRange:null,rhythmIndex:null,rhythm:null,localBeatIndex:null,localBeats:null,pendingSelection:null,pairEpoch:0};
  const number = value => Number(value).toFixed(3);
  const splitNames = {train:"辅助模型训练模板",validation:"合成验证模板",test:"合成留出模板",unlabeled:"无确认和弦真值"};
  const domainNames = {procedural:"程序合成音色",sampled_piano_holdout:"录音钢琴音源留出",public_midi:"公开 MIDI · 钢琴重渲染"};
  const timbreNames = {warm:"温暖音色",bright:"明亮音色",kw_piano:"KW 录音钢琴"};
  function node(tag, text, className) {
    const result = document.createElement(tag); if (text !== undefined) result.textContent = text;
    if (className) result.className = className; return result;
  }
  function status(message, error=false) { $("status").textContent=message; $("status").classList.toggle("error",error); }
  async function sha256(bytes) { return [...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))].map(v=>v.toString(16).padStart(2,"0")).join(""); }
  async function fetchBytes(name, signal, maximum) {
    if (!/^(?:index|[a-z0-9_]+)\.json$/.test(name) && !/^[a-f0-9]{64}\.wav$/.test(name)) throw Error("本地资源路径无效");
    const response=await fetch(prefix+name,{signal,cache:"no-store"});
    if (!response.ok) throw Error("本地诊断数据尚未生成，或文件无法读取");
    const bytes=await response.arrayBuffer(); if (bytes.byteLength>maximum) throw Error("诊断文件超出读取上限"); return bytes;
  }
  function stop() { state.playingRange=null; audio.pause(); if(state.ready)status("已暂停 · 可选择另一个区间继续试听。"); }
  function selectedRange() { return core.range(Number($("start").value),Number($("end").value),Number($("context").value),state.item.duration); }
  function rangeLabel() {
    $("rangeLink").hidden=true;$("rangeLink").removeAttribute("href");
    try { const range=selectedRange(); $("playRange").textContent=`实际试听 ${number(range.start)}–${number(range.end)} s`;
      if(state.ready){const entry=state.index.cases.find(row=>row.id===state.item.id);
        $("rangeLink").href=core.selectionURL(location.href,entry,Number($("start").value),Number($("end").value),Number($("context").value));$("rangeLink").hidden=false;}}
    catch (error) { $("playRange").textContent=error.message; }
  }
  function choose(start,end,label,trackId) {
    stop(); $("start").value=start; $("end").value=end; state.selected={start,end,label,trackId};
    $("selected").textContent=`${label} · ${number(start)}–${number(end)} s · ${number(end-start)} 秒`;
    rangeLabel(); renderNotes(); renderTracks(); drawWaveform();
    status(`已选择 ${label} · ${number(start)}–${number(end)} s`);
  }
  function windowRange() {
    const duration=state.item.duration, setting=$("windowSize").value;
    const length=setting==="all" ? duration : Math.min(Number(setting),duration);
    state.windowStart=Math.max(0,Math.min(state.windowStart,duration-length));
    return {start:state.windowStart,end:Math.min(duration,state.windowStart+length),length};
  }
  function renderTracks() {
    if (!state.item) return;
    const focusTitle=document.activeElement?.classList.contains("segment") ? document.activeElement.title : null;
    const view=windowRange(); $("windowLabel").textContent=`${number(view.start)}–${number(view.end)} s`;
    $("previous").disabled=view.start<=0; $("next").disabled=view.end>=state.item.duration;
    $("ruler").replaceChildren();
    for (let i=0;i<=4;i++) { const mark=node("span",(view.start+view.length*i/4).toFixed(1)); mark.style.left=(i*25)+"%"; $("ruler").append(mark); }
    const tracks=state.item.tracks.slice();
    if (state.item.bars.length) tracks.unshift({id:"bars",kind:"bars",name:"作曲小节",rows:state.item.bars.map(row=>({start:row.start_sample/state.item.sample_rate,end:row.end_sample/state.item.sample_rate,label:`${row.pickup?"弱起":"小节 "+row.bar} · ${row.numerator}/${row.denominator}`}))});
    $("tracks").replaceChildren();
    for (const track of tracks) {
      const row=node("div",undefined,"track "+track.kind), heading=node("div",track.name,"track-title");
      row.dataset.track=track.id;
      if (isNumber(track.spec_match_pct)) heading.append(node("strong",`规格匹配 ${track.spec_match_pct.toFixed(2)}%`));
      else if (track.kind==="specification") heading.append(node("strong","生成意图 · 非听辨真值"));
      const lane=node("div",undefined,"track-lane");
      for (const segment of track.rows) {
        const start=Math.max(view.start,segment.start),end=Math.min(view.end,segment.end); if (end<=start) continue;
        const button=node("button",segment.label,"segment");
        button.style.left=((start-view.start)/view.length*100)+"%"; button.style.width=((end-start)/view.length*100)+"%";
        button.title=`${track.name}: ${segment.label} · ${number(segment.start)}–${number(segment.end)} s`;
        button.setAttribute("aria-label",button.title);
        if (state.selected && state.selected.trackId===track.id && state.selected.start===segment.start && state.selected.end===segment.end) button.classList.add("chosen");
        button.onclick=()=>choose(segment.start,segment.end,track.name+" · "+segment.label,track.id); lane.append(button);
      }
      row.append(heading,lane); $("tracks").append(row);
    }
    if(focusTitle) [...$("tracks").querySelectorAll("button")].find(button=>button.title===focusTitle)?.focus({preventScroll:true});
    renderRhythm();renderLocalBeats();
  }
  const keyName=label=>label.replace(/ major$/," 大调").replace(/ minor$/," 小调");
  function scopeText(scope) {
    const pcs=["C","C♯","D","E♭","E","F","F♯","G","A♭","A","B♭","B"];
    return scope.intents.map(([pc,mode])=>`${pcs[pc]}${mode==="major"?"大调":"小调"}`).join(" / ");
  }
  function renderRhythm() {
    const data=state.rhythm;if(!data||!state.item)return;
    $("rhythmContent").hidden=false;const global=data.global_key;
    $("globalKeySummary").textContent=`全段 CNN：${keyName(global.prediction)}。`+(global.scope.status==="multiple"?`作曲规格包含 ${scopeText(global.scope)}，不设单一主调答案。`:`作曲规格：${scopeText(global.scope)}；${global.exact_match?"标签匹配":"标签不一致，需听辨"}。`);
    $("rhythmMetrics").replaceChildren();
    for(const [key,title] of [["written_denominator","书面分母拍 F1"],["compound_dotted_quarter","复合拍层级 F1"],["downbeats","完整小节重拍 F1"]]) {
      const metric=node("div",undefined,"rhythm-metric");metric.append(node("strong",data.rhythm[key].f1.toFixed(3)),node("span",title));$("rhythmMetrics").append(metric);
    }
    const view=windowRange(),tracks=[{title:"作曲书面拍",values:data.reference_levels.written_denominator},
      {title:"作曲复合拍",values:data.reference_levels.compound_dotted_quarter},
      {title:"作曲重拍",values:data.reference_levels.downbeats},
      {title:"模型拍点",values:data.predicted_beats.map(b=>b.time),prediction:true},
      {title:"模型重拍",values:data.predicted_beats.filter(b=>b.beat===1).map(b=>b.time),prediction:true}];
    $("rhythmWindowLabel").textContent=`当前 ${number(view.start)}–${number(view.end)} s`;
    const focus=document.activeElement?.classList.contains("rhythm-tick")?document.activeElement.title:null;
    const windowFocus=document.activeElement?.classList.contains("key-window")?document.activeElement.title:null;
    $("rhythmTracks").replaceChildren();
    for(const track of tracks){const row=node("div",undefined,"track "+(track.prediction?"prediction":"")),lane=node("div",undefined,"track-lane");
      row.dataset.rhythmTrack=track.title;
      for(const time of track.values){if(time<view.start||time>=view.end)continue;const button=node("button",undefined,"rhythm-tick");
        button.style.left=`calc(${(time-view.start)/view.length*100}% - ${time===view.start?0:3}px)`;button.title=`${track.title} · ${number(time)} s`;button.setAttribute("aria-label",button.title);
        button.onclick=()=>choose(Math.max(0,time-.25),Math.min(state.item.duration,time+.25),button.title,"rhythm");lane.append(button);}
      row.append(node("div",track.title,"track-title"),lane);$("rhythmTracks").append(row);}
    if(focus)[...$("rhythmTracks").querySelectorAll("button")].find(button=>button.title===focus)?.focus({preventScroll:true});
    $("gridCoverage").textContent=`70 毫秒容差、一对一匹配；分别报告两种拍层级，不取较高值。开头 ${number(data.initial_ungridded_seconds)} 秒不在自动结构网格内，该区域须独立复核。`+
      (data.final_interval_seconds===null?"没有检测到小节重拍。":`末区间 ${number(data.final_interval_seconds)} 秒由文件终点截断，完整性未确认。`);
    $("keyWindows").replaceChildren();
    for(const window of data.key_windows){const mixed=window.scope.status==="multiple",button=node("button",undefined,"key-window "+(mixed?"mixed":window.exact_match===false?"mismatch":""));
      button.append(node("span",`${number(window.start)}–${number(window.end)} s`),node("strong",keyName(window.prediction)),
        node("span",mixed?`跨调窗口：${scopeText(window.scope)}；不设单一答案。`:`规格 ${scopeText(window.scope)} · ${window.exact_match?"匹配":"不一致"}`));
      button.title=`CNN 窗口 ${number(window.start)}–${number(window.end)} s · ${keyName(window.prediction)}`;
      button.onclick=()=>{state.windowStart=window.start;choose(window.start,window.end,`CNN 窗口 · ${keyName(window.prediction)}`,"key_window");};$("keyWindows").append(button);}
    if(windowFocus)[...$("keyWindows").querySelectorAll("button")].find(button=>button.title===windowFocus)?.focus({preventScroll:true});
  }
  async function loadRhythm(item,epoch,signal) {
    state.rhythm=null;$("rhythmContent").hidden=true;
    if(item.truth_kind==="none"){$("rhythmStatus").textContent="此公开 MIDI 没有确认的调性或拍点真值，不显示规格评分。";return;}
    if(!state.rhythmIndex)return;
    const entry=state.rhythmIndex.cases.find(row=>row.id===item.id);
    if(!entry){$("rhythmStatus").textContent="当前样例没有已绑定的调性/节拍诊断记录。";return;}
    try {const bytes=await fetchBytes(entry.file,signal,2*1024*1024);
      if(await sha256(bytes)!==entry.sha256)throw Error("调性/节拍文件 SHA 不匹配");
      if(epoch!==state.epoch)return;
      state.rhythm=core.validateRhythm(JSON.parse(new TextDecoder().decode(bytes)),item,entry);
      $("rhythmStatus").textContent="记录已绑定当前音频与作曲规格；冻结模型未作调整。点击拍点可试听前后各 0.25 秒。";renderRhythm();
    }catch(error){if(epoch===state.epoch&&error.name!=="AbortError"){$("rhythmStatus").textContent=error.message;$("rhythmContent").hidden=true;}}
  }
  function renderLocalBeats() {
    const data=state.localBeats;if(!data||!state.item)return;
    $("localBeatContent").hidden=false;const view=windowRange();
    $("localBeatView").textContent=`当前 ${number(view.start)}–${number(view.end)} s，与和声时间窗同步。实线为重拍，虚线为未确认接缝。`;
    const focus=document.activeElement?.classList.contains("local-beat-button")?document.activeElement.title:null;
    $("localBeatTracks").replaceChildren();$("localBeatWindows").replaceChildren();
    const tracks=[{id:"original",title:"原全段默认",beats:data.baseline_beats,metrics:data.baseline_metrics,seams:[]},
      ...data.variants.map(variant=>({...variant,title:`窗口起点 ${variant.origin_seconds} 秒`}))];
    for(const track of tracks){
      const row=node("div",undefined,"track local-beat-track"),heading=node("div",track.title,"track-title"),lane=node("div",undefined,"track-lane");
      row.dataset.localBeatTrack=track.id;heading.append(node("strong",`重拍 F1 ${track.metrics.downbeats.f1.toFixed(3)}`));
      for(const [values,seam] of [[track.beats.filter(b=>b.beat===1),false],[track.seams,true]])for(const value of values){
        const time=value.time;if(time<view.start||time>=view.end)continue;
        const button=node("button",undefined,"local-beat-button "+(seam?"beat-seam":"rhythm-tick"));
        button.style.left=`calc(${(time-view.start)/view.length*100}% - ${time===view.start?0:3}px)`;
        button.title=`${track.title} · ${seam?"未确认接缝":"重拍"} · ${number(time)} s`;button.setAttribute("aria-label",button.title);
        button.onclick=()=>choose(Math.max(0,time-(seam?.5:.25)),Math.min(state.item.duration,time+(seam?.5:.25)),button.title,"local_beat");lane.append(button);
      }
      row.append(heading,lane);$("localBeatTracks").append(row);
    }
    for(const variant of data.variants){
      const group=node("section",undefined,"local-window-group");group.append(node("h4",`起点 ${variant.origin_seconds} 秒 · 每窗模型拍数（非确认拍号）`));
      const windows=node("div",undefined,"key-windows");
      for(const window of variant.windows){const button=node("button",undefined,"local-beat-button key-window local-beat-window");
        button.append(node("span",`${number(window.start)}–${number(window.end)} s`),node("strong",`模型每循环 ${window.model_count} 拍`));
        button.title=`起点 ${variant.origin_seconds} 秒的局部解码窗 · ${number(window.start)}–${number(window.end)} s`;
        button.onclick=()=>{state.windowStart=window.start;choose(window.start,window.end,button.title,"local_beat_window");};windows.append(button);}
      group.append(windows);
      const seams=node("div",undefined,"local-seam-list");
      for(const seam of variant.seams){const button=node("button",`接缝 ${number(seam.time)} s · 未确认`,"local-beat-button local-seam-choice");
        button.title=`试听起点 ${variant.origin_seconds} 秒方案的接缝 · ${number(seam.time)} s`;
        button.onclick=()=>choose(Math.max(0,seam.time-.5),Math.min(state.item.duration,seam.time+.5),button.title,"local_beat_seam");seams.append(button);}
      group.append(seams);$("localBeatWindows").append(group);
    }
    if(focus)[...$("localBeatContent").querySelectorAll("button")].find(button=>button.title===focus)?.focus({preventScroll:true});
  }
  async function loadLocalBeats(item,epoch,signal) {
    state.localBeats=null;$("localBeatContent").hidden=true;$("localBeatTracks").replaceChildren();$("localBeatWindows").replaceChildren();
    if(item.truth_kind==="none"){$("localBeatStatus").textContent="此公开 MIDI 没有确认的节拍规格，不显示局部试验评分。";return;}
    if(!state.localBeatIndex)return;
    const entry=state.localBeatIndex.cases.find(row=>row.id===item.id);
    if(!entry){$("localBeatStatus").textContent="当前样例没有已绑定的局部节拍试验。";return;}
    try{const bytes=await fetchBytes(entry.file,signal,2*1024*1024);
      if(await sha256(bytes)!==entry.sha256)throw Error("局部节拍文件 SHA 不匹配");if(epoch!==state.epoch)return;
      state.localBeats=core.validateLocalBeats(JSON.parse(new TextDecoder().decode(bytes)),item,entry);
      $("localBeatStatus").textContent="记录已绑定当前音频与作曲规格；两种起点分别展示，未自动选用。";renderLocalBeats();
    }catch(error){if(epoch===state.epoch&&error.name!=="AbortError"){$("localBeatStatus").textContent=error.message;$("localBeatContent").hidden=true;}}
  }
  const pitchName = pitch => ["C","C♯","D","E♭","E","F","F♯","G","A♭","A","B♭","B"][pitch%12]+(Math.floor(pitch/12)-1);
  function renderNotes() {
    const item=state.item; if (!item) return;
    const start=Number($("start").value),end=Number($("end").value);
    const notes=item.notes.filter(note=>note.end_sample/item.sample_rate>start && note.start_sample/item.sample_rate<end);
    $("noteList").replaceChildren();
    if (!notes.length) { $("noteList").append(node("p",item.truth_kind==="none"?"公开 MIDI 不提供已确认的和弦、局部调性或音符角色真值。":"当前范围没有作曲音符。")); return; }
    $("noteList").append(node("p",`所选范围与 ${notes.length} 个 MIDI 音符相交；表内显示原始触发起止。`));
    const table=node("table",undefined,"note-table"), head=node("tr");
    for (const name of ["音高","声部 / 角色","起止（秒）"]) head.append(node("th",name)); table.append(head);
    for (const note of notes.slice(0,80)) {
      const row=node("tr",undefined,note.intent==="nonchord_tone"||note.intent?.includes("passing")?"passing":"");
      const roles={bass:"低音",accompaniment:"伴奏",melody:"旋律",drums:"打击乐"};
      const intents={chord_tone:"和弦音",nonchord_passing_ninth:"经过九音",rhythmic_marker:"节奏标记"};
      row.append(node("td",pitchName(note.pitch)),node("td",`${roles[note.role]||note.role} · ${intents[note.intent]||note.intent}`),node("td",`${number(note.start_sample/item.sample_rate)}–${number(note.end_sample/item.sample_rate)}`)); table.append(row);
    }
    $("noteList").append(table); if (notes.length>80) $("noteList").append(node("p","仅展示前 80 个，请缩小范围查看。"));
  }
  function drawWaveform() {
    if (!state.item) return;
    const canvas=$("waveform"), rect=canvas.getBoundingClientRect(), ratio=devicePixelRatio||1;
    canvas.width=Math.max(1,Math.round(rect.width*ratio)); canvas.height=84*ratio;
    const ctx=canvas.getContext("2d"); ctx.scale(ratio,ratio); const width=rect.width,height=84,peaks=state.item.peaks;
    ctx.clearRect(0,0,width,height); ctx.fillStyle="#5f8b83";
    peaks.forEach((peak,index)=>ctx.fillRect(index/peaks.length*width,(height-peak*height*.9)/2,Math.max(1,width/peaks.length*.65),Math.max(1,peak*height*.9)));
    const duration=state.item.duration, start=Number($("start").value),end=Number($("end").value);
    if (isNumber(start+end)) { ctx.fillStyle="#8ce4ca25"; ctx.fillRect(start/duration*width,0,(end-start)/duration*width,height); }
    ctx.fillStyle="#e8c191";ctx.fillRect(audio.currentTime/duration*width,0,1,height);
  }
  function provenance(item) {
    $("identity").replaceChildren();
    for (const text of [`音频 SHA-256: ${item.audio_sha256}`,`样例: ${item.id} · ${item.sample_rate} Hz`,
      "程序时间轴绑定实际 PCM 采样位置；录音钢琴的攻击形状不等于 MIDI 瞬时触发。浏览器区间播放为交互试听，不是采样级导出。",
      "辅助模型固定根音及基础大小三和弦，250 毫秒单元、1 秒上下文。尚未用真实外部验证确认收益。"])
      $("identity").append(node("p",text,"identity-line"));
    function sourceLink(title,url) { const link=node("a",title); link.href=url;link.target="_blank";link.rel="noopener noreferrer";const line=node("p");line.append(link);$("identity").append(line); }
    if (item.attribution) { $("identity").append(node("p",`${item.attribution} · ${item.license}`));sourceLink("MIDI 原始来源与署名",item.source_url); }
    if (item.timbre==="kw_piano") sourceLink("FreePats Upright Piano KW small SFZ 2019-07-03 · CC0",state.index.instrument.source_url);
  }
  function renderCase(item) {
    $("content").hidden=false; $("caseTitle").textContent=item.title; $("caseDomain").textContent=domainNames[item.domain]; $("duration").textContent=number(item.duration)+" 秒";
    $("badges").replaceChildren(node("span",splitNames[item.split],"badge "+item.split),node("span",timbreNames[item.timbre]||item.timbre,"badge"));
    if (item.transpose!==null) $("badges").append(node("span",`移调 +${item.transpose} 半音`,"badge"));
    $("scopeNote").textContent=item.truth_kind==="none"?"此公开 MIDI 没有确认的和弦真值；各轨仅为模型预测，不计算准确率。踏板与音量已保留，所有有音高通道使用同一钢琴音源。":
      (item.split==="train"?"此模板已用于辅助模型训练，只能展示机制，不能作为泛化证据。":"此模板未参与辅助模型训练；少量相关移调与音色变体不等同于独立歌曲。")+" 完整和声规格匹配不代表听辨真值或真实歌曲准确率。";
    $("start").max=$("end").max=item.duration; $("start").value=0; $("end").value=Math.min(2,item.duration);
    $("selected").textContent="选择一段和弦，查看精确标签与时间。"; $("clock").textContent="0.000 s";
    $("findings").replaceChildren();
    for (const finding of item.findings) { const button=node("button",undefined,"finding");button.append(node("strong",finding.title),node("span",finding.detail));button.onclick=()=>{state.windowStart=Math.max(0,finding.start-1);choose(finding.start,finding.end,finding.title,"finding");};$("findings").append(button); }
    $("scoreNote").textContent="局部调性和段落是手写作曲意图，不强迫全段只有一个主调。小节内拍点按书面拍号单位记录；6/8 不自动等同于六个感知重拍。";
    $("keySections").replaceChildren();
    const pcs=["C","C♯","D","E♭","E","F","F♯","G","A♭","A","B♭","B"];
    for (const key of item.local_keys) $("keySections").append(node("p",`${number(key.start)}–${number(key.end)} s：${pcs[key.tonic_pc]} ${key.mode==="major"?"大调":"小调"}（作曲意图）`));
    if (item.sections.length) $("keySections").append(node("p",item.sections.map(s=>`${s.id} ${number(s.start)}–${number(s.end)} s`).join(" · ")));
    const alternatives=core.pairedEntries(item,state.index);$("timbreComparison").hidden=!alternatives.length;
    $("pairedCase").replaceChildren();$("pairStatus").textContent="";$("compareTimbre").disabled=true;
    for(const entry of alternatives){const option=node("option",timbreNames[entry.timbre]||entry.timbre);option.value=entry.id;$("pairedCase").append(option);}
    rangeLabel();renderTracks();renderNotes();provenance(item);requestAnimationFrame(drawWaveform);
  }
  async function loadCase(id) {
    const entry=state.index.cases.find(item=>item.id===id); if (!entry) return;
    const requested=state.pendingSelection?.id===id?state.pendingSelection:null;state.pendingSelection=null;
    const epoch=++state.epoch;state.pairEpoch++;state.controller?.abort();state.controller=new AbortController(); const signal=state.controller.signal;
    stop();state.ready=false;state.item=null;state.rhythm=null;state.localBeats=null;state.windowStart=0;state.selected=null;$("play").disabled=true;$("pause").disabled=true;$("content").hidden=true;
    $("rhythmContent").hidden=true;$("rhythmStatus").textContent="正在查找绑定此音频的记录…";
    $("localBeatContent").hidden=true;$("localBeatStatus").textContent="正在查找绑定此音频的局部诊断…";
    $("localBeatTracks").replaceChildren();$("localBeatWindows").replaceChildren();
    $("rangeLink").hidden=true;$("rangeLink").removeAttribute("href");
    audio.removeAttribute("src");audio.load(); if (state.blob) {URL.revokeObjectURL(state.blob);state.blob=null;}
    $("badges").replaceChildren();status("正在核验样例与音频身份…");
    try {
      const bytes=await fetchBytes(entry.file,signal,4*1024*1024);
      if (await sha256(bytes)!==entry.sha256) throw Error("样例文件 SHA 不匹配，已停止加载");
      if (epoch!==state.epoch) return;
      const item=core.validateCase(JSON.parse(new TextDecoder().decode(bytes)),entry);state.item=item;renderCase(item);
      loadRhythm(item,epoch,signal);loadLocalBeats(item,epoch,signal);
      const sound=await fetchBytes(item.audio,signal,100*1024*1024);
      if (await sha256(sound)!==item.audio_sha256) throw Error("音频 SHA 不匹配，已禁止试听");
      if (epoch!==state.epoch) return;
      const blob=URL.createObjectURL(new Blob([sound],{type:"audio/wav"}));state.blob=blob;audio.src=blob;
      await new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>finish(Error("音频元数据读取超时")),10000);
        function finish(error) { clearTimeout(timer);audio.removeEventListener("loadedmetadata",loaded);audio.removeEventListener("error",failed);signal.removeEventListener("abort",aborted);error?reject(error):resolve(); }
        function loaded(){finish();}function failed(){finish(Error("浏览器无法读取此 WAV"));}function aborted(){finish(new DOMException("Aborted","AbortError"));}
        audio.addEventListener("loadedmetadata",loaded,{once:true});audio.addEventListener("error",failed,{once:true});signal.addEventListener("abort",aborted,{once:true});audio.load();
      });
      if (epoch!==state.epoch) return;
      if (Math.abs(audio.duration-item.duration)>1/item.sample_rate+1e-6) throw Error("解码时长与样例不一致");
      state.ready=true;$("play").disabled=false;$("pause").disabled=false;$("compareTimbre").disabled=false;status("音频与样例身份已核验 · 点击和弦选择范围，然后试听。");
      if(requested){$("context").value=String(requested.context);state.windowStart=requested.start;
        choose(requested.start,requested.end,requested.reason==="timbre"?"同作曲音色对照":"链接指定范围",requested.reason==="timbre"?"timbre":"link");
        $("navigationStatus").textContent=requested.reason==="timbre"?
          `已核对同一作曲规格，切换音色并保持 ${number(requested.start)}–${number(requested.end)} s；点击试听后才会播放。`:
          `已恢复链接范围 ${number(requested.start)}–${number(requested.end)} s；点击试听后才会播放。`;$("navigationStatus").hidden=false;
        if(requested.reason==="timbre")$("compareTimbre").focus({preventScroll:true});}
      rangeLabel();
    } catch (error) {
      if (epoch!==state.epoch || error.name==="AbortError") return;
      stop();state.ready=false;$("play").disabled=true;$("pause").disabled=true;status(error.message,true);
    }
  }
  function populate() {
    const filter=$("filter").value;
    const entries=state.index.cases.filter(item=>filter==="all"||item.split===filter||item.domain===filter);
    const previous=state.pendingSelection?.id||$("caseSelect").value;$("caseSelect").replaceChildren();
    for (const item of entries) { const option=node("option",`${item.title} · ${timbreNames[item.timbre]||item.timbre}${item.transpose===null?"":` · +${item.transpose}`} · ${splitNames[item.split]}`);option.value=item.id;$("caseSelect").append(option); }
    $("caseSelect").disabled=!entries.length; if(entries.some(item=>item.id===previous)) $("caseSelect").value=previous;
    if(entries.length) loadCase($("caseSelect").value);
  }
  $("filter").onchange=()=>{$("navigationStatus").hidden=true;populate();};$("caseSelect").onchange=()=>{$("navigationStatus").hidden=true;loadCase($("caseSelect").value);};
  $("pairedCase").onchange=()=>{state.pairEpoch++;$("compareTimbre").disabled=!state.ready;$("pairStatus").textContent="";};
  $("compareTimbre").onclick=async()=>{
    if(!state.ready||!state.item)return;
    const source=state.item,epoch=state.epoch,pairEpoch=++state.pairEpoch;
    const entry=core.pairedEntries(source,state.index).find(item=>item.id===$("pairedCase").value);
    if(!entry)return;
    const start=Number($("start").value),end=Number($("end").value),context=Number($("context").value);
    stop();$("compareTimbre").disabled=true;$("pairStatus").textContent="正在核对作曲规格…";
    try{
      core.range(start,end,context,source.duration);
      const bytes=await fetchBytes(entry.file,state.controller.signal,4*1024*1024);
      if(await sha256(bytes)!==entry.sha256)throw Error("对照样例版本不一致，未切换音色");
      if(epoch!==state.epoch||pairEpoch!==state.pairEpoch)return;
      const target=core.validateCase(JSON.parse(new TextDecoder().decode(bytes)),entry);
      if(!core.sameComposition(source,target))throw Error("两个样例的作曲规格不一致，未切换音色");
      if(start!==Number($("start").value)||end!==Number($("end").value)||context!==Number($("context").value))throw Error("所选范围已变化，请重新切换音色");
      state.pendingSelection={id:entry.id,case_sha256:entry.sha256,start,end,context,reason:"timbre"};
      $("filter").value="all";populate();
    }catch(error){if(epoch===state.epoch&&pairEpoch===state.pairEpoch&&error.name!=="AbortError")$("pairStatus").textContent=error.message;}
    finally{if(epoch===state.epoch&&pairEpoch===state.pairEpoch)$("compareTimbre").disabled=!state.ready;}
  };
  $("play").onclick=async()=>{if(!state.ready)return;const epoch=state.epoch;try{const range=selectedRange();audio.currentTime=range.start;audio.playbackRate=Number($("speed").value);state.playingRange=range;await audio.play();if(epoch!==state.epoch)return;status(`试听 ${number(range.start)}–${number(range.end)} s`);}catch(error){if(epoch===state.epoch){stop();status(error.message,true);}}};
  $("pause").onclick=stop;$("speed").onchange=()=>{audio.playbackRate=Number($("speed").value);};
  for(const id of ["start","end","context"])$(id).onchange=()=>{stop();rangeLabel();renderNotes();drawWaveform();};
  $("previous").onclick=()=>{state.windowStart-=windowRange().length;renderTracks();};$("next").onclick=()=>{state.windowStart+=windowRange().length;renderTracks();};
  $("windowSize").onchange=renderTracks;window.addEventListener("resize",drawWaveform);
  function playbackBoundary(){const range=state.playingRange;if(!range)return;if(audio.currentTime>=range.end-.004||audio.ended){if($("loop").checked){audio.currentTime=range.start;if(audio.paused)audio.play().catch(()=>stop());}else{stop();audio.currentTime=range.end;}}}
  audio.addEventListener("timeupdate",playbackBoundary);audio.addEventListener("ended",playbackBoundary);
  function frame(){if(!audio.paused){playbackBoundary();$("clock").textContent=number(audio.currentTime)+" s";drawWaveform();}requestAnimationFrame(frame);}requestAnimationFrame(frame);
  window.addEventListener("beforeunload",()=>{state.controller?.abort();if(state.blob)URL.revokeObjectURL(state.blob);});
  (async()=>{try{const bytes=await fetchBytes("index.json",undefined,2*1024*1024);const index=JSON.parse(new TextDecoder().decode(bytes));
    if(index.schema!=="harmony_listening_index_v1"||index.readonly!==true||index.private_audio_included!==false||!Array.isArray(index.cases)||!index.cases.length||index.cases.length>200)throw Error("诊断索引无效");
    const ids=new Set();for(const entry of index.cases){if(ids.has(entry.id)||!digest(entry.sha256))throw Error("样例索引身份重复或缺失");ids.add(entry.id);}
    state.index=index;
    try{state.pendingSelection=core.selectionFromURL(location.search,index);}catch(error){$("navigationStatus").textContent=error.message;$("navigationStatus").hidden=false;}
    populate();
    try {const rhythmBytes=await fetchBytes("rhythm_index.json",undefined,2*1024*1024),rhythmIndex=JSON.parse(new TextDecoder().decode(rhythmBytes));
      if(rhythmIndex.schema!=="harmony_rhythm_review_index_v1"||rhythmIndex.listening_index_sha256!==await sha256(bytes)||!Array.isArray(rhythmIndex.cases))throw Error("调性/节拍索引与诊断集不匹配");
      for(const entry of rhythmIndex.cases){const original=index.cases.find(row=>row.id===entry.id);if(!original||original.sha256!==entry.case_sha256||!digest(entry.sha256))throw Error("调性/节拍样例身份不一致");}
      state.rhythmIndex=rhythmIndex;if(state.item)loadRhythm(state.item,state.epoch,state.controller.signal);
    }catch(error){$("rhythmStatus").textContent=error.message;}
    try {const localBytes=await fetchBytes("local_beat_index.json",undefined,2*1024*1024),localIndex=JSON.parse(new TextDecoder().decode(localBytes));
      if(localIndex.schema!=="harmony_local_beat_index_v1"||localIndex.listening_index_sha256!==await sha256(bytes)||!Array.isArray(localIndex.cases))throw Error("局部节拍索引与诊断集不匹配");
      const seen=new Set();for(const entry of localIndex.cases){const original=index.cases.find(row=>row.id===entry.id);
        if(seen.has(entry.id)||!original||original.sha256!==entry.case_sha256||!digest(entry.sha256))throw Error("局部节拍样例身份不一致");seen.add(entry.id);}
      state.localBeatIndex=localIndex;if(state.item)loadLocalBeats(state.item,state.epoch,state.controller.signal);
    }catch(error){$("localBeatStatus").textContent=error.message;}
    }catch(error){status(error.message,true);}})();
})();
