/* All exposed DEV, read only. Case payloads and audio are checked before use. */
(function () {
  "use strict";
  const number=v=>typeof v==="number"&&Number.isFinite(v), digest=v=>typeof v==="string"&&/^[a-f0-9]{64}$/.test(v);
  const names={harmony:"和声规格",local_key:"局部调性",downbeat:"重拍"};
  const ids=["sections","bars","harmony_reference","harmony_baseline","harmony_candidate","key_reference","key_baseline","key_candidate","pulse_reference","pulse_predicted","bar_reference","bar_baseline","bar_candidate"];
  const fields=["id","group_id","dataset","cohort","native_split","training_eligible","exposed_DEV","renderer","duration","audio_sha256","labels_sha256"];
  const core={
    rate(m){return "f1" in m?m.f1:m.seconds?m.correct_seconds/m.seconds:null;},
    range(start,end,context,duration){
      if(![start,end,context,duration].every(number)||start<0||end<=start||end>duration+1e-8||![0,.5,1,2].includes(context))throw Error("请输入录音内有效的起止范围");
      return {start:Math.max(0,start-context),end:Math.min(duration,end+context)};
    },
    validateIndex(index){
      if(index?.schema!=="model_review_index_v1"||index.cases?.length!==240||index.defaults_changed!==false||index.real_recording_accuracy!==false)throw Error("完整 DEV 清单无效");
      const seen=new Set(),counts={};
      for(const e of index.cases){
        if(!/^[A-Za-z0-9_]+$/.test(e.id)||seen.has(e.id)||e.file!==e.id+".json"||!digest(e.sha256)||!digest(e.audio_sha256)||!digest(e.labels_sha256)||!number(e.duration)||e.duration<=0||e.duration>1800||e.training_eligible!==false||e.exposed_DEV!==true||!["source","grammar"].includes(e.dataset)||!Array.isArray(e.regressions))throw Error("DEV 样例身份或范围不一致");
        seen.add(e.id);const k=e.dataset+":"+e.cohort;counts[k]=(counts[k]||0)+1;
      }
      const required={"source:dev_source_only":96,"source:dev_family_seen":48,"source:dev_family_unseen":48,"grammar:dev_family_seen":24,"grammar:dev_family_unseen":24};
      if(Object.keys(counts).length!==5||Object.keys(required).some(k=>counts[k]!==required[k]||index.counts[k]!==required[k]))throw Error("DEV 样例分组不完整");
      return index;
    },
    validateCase(item,entry){
      if(item?.schema!=="model_review_case_v1"||fields.some(k=>item[k]!==entry[k])||item.audio!==item.audio_sha256+".wav"||!Number.isInteger(item.sample_rate)||item.sample_rate<8000||item.sample_rate>192000||!Array.isArray(item.tracks)||item.tracks.length!==13)throw Error("样例身份或音频信息不一致");
      for(let i=0;i<ids.length;i++){
        const t=item.tracks[i];if(t.id!==ids[i]||typeof t.title!=="string")throw Error("时间轴来源不一致");
        if(i>=8){
          if(t.kind!=="events"||!Array.isArray(t.events)||t.events.some((v,j)=>!number(v)||v<0||v>=item.duration||(j&&v<=t.events[j-1])))throw Error("拍点时间轴无效");
        }else{
          const kind=i===3||i===6?"baseline":i===4||i===7?"candidate":"reference";
          if(t.kind!==kind||!Array.isArray(t.rows)||!t.rows.length)throw Error("区间时间轴无效");
          let end=0;
          for(const r of t.rows){
            if(!number(r.start)||!number(r.end)||r.start<end-1e-8||r.end<=r.start||r.end>item.duration+1e-8||typeof r.label!=="string"||(kind==="reference"&&typeof r.status!=="string"))throw Error("区间时间轴无效");
            if(i>=2&&Math.abs(r.start-end)>1e-7)throw Error("和声或调性时间轴存在缺口");
            end=r.end;
          }
          if(i>=2&&Math.abs(end-item.duration)>1e-7)throw Error("和声或调性时间轴缺少尾部");
        }
      }
      for(const arm of ["baseline","candidate"]){
        if(!item.metrics?.[arm]||Object.keys(item.metrics[arm]).sort().join()!==Object.keys(names).sort().join())throw Error("评分来源不完整");
        for(const metric of ["harmony","local_key"]){const m=item.metrics[arm][metric];
          if(!number(m.seconds)||!number(m.correct_seconds)||m.seconds<0||m.seconds>item.duration+1e-7||m.correct_seconds<0||m.correct_seconds>m.seconds+1e-7)throw Error("评分时间分母无效");
          if(metric==="harmony"&&Math.abs(m.seconds-item.duration)>1e-7)throw Error("和声必须保留完整分母");
        }
        const m=item.metrics[arm].downbeat;
        if(["true_positive","predicted","reference"].some(k=>!Number.isInteger(m[k])||m[k]<0)||m.true_positive>Math.min(m.predicted,m.reference)||m.tolerance_seconds!==.07)throw Error("重拍计数或容差无效");
        const f=m.predicted+m.reference?2*m.true_positive/(m.predicted+m.reference):null;
        if(f===null?m.f1!==null:!number(m.f1)||Math.abs(f-m.f1)>1e-10)throw Error("重拍 F1 与计数不一致");
      }
      const actual=Object.keys(names).filter(k=>core.rate(item.metrics.baseline[k])!==null&&core.rate(item.metrics.candidate[k])<core.rate(item.metrics.baseline[k])-1e-10);
      if(actual.join()!==item.regressions?.join()||actual.join()!==entry.regressions.join())throw Error("退步标记与评分不一致");
      return item;
    }
  };
  if(typeof module!=="undefined"&&module.exports)module.exports=core;
  if(typeof document==="undefined")return;
  window.ModelReviewCore=core;
  const $=id=>document.getElementById(id), data="/assets/diagnostic-data/model-review-v1/";
  let index,item,epoch=0,controller,url,windowStart=0,playRange=null,frame=0;
  const audio=$("audio"), pct=v=>v===null?"无分母":(100*v).toFixed(1)+"%";
  const el=(tag,text,className)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;};
  async function hash(bytes){return [...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))].map(v=>v.toString(16).padStart(2,"0")).join("");}
  async function fetchBytes(path,signal){const r=await fetch(data+path,{signal,cache:"no-store"});if(!r.ok)throw Error("本地文件不可用："+r.status);return r.arrayBuffer();}
  const stop=()=>{audio.pause();playRange=null;cancelAnimationFrame(frame);frame=0;};
  function viewport(){const value=$("windowSize").value,size=value==="all"?item.duration:Number(value);return {start:windowStart,end:Math.min(item.duration,windowStart+size),size};}
  function clocks(){
    cancelAnimationFrame(frame);frame=0;
    $("clock").textContent=audio.currentTime.toFixed(3)+" s";
    if(item){const v=viewport(),x=100*(audio.currentTime-v.start)/(v.end-v.start);
      document.querySelectorAll(".playhead").forEach(h=>{h.style.left=x+"%";h.hidden=x<0||x>100;});}
    if(playRange&&!audio.paused&&audio.currentTime>=playRange.end-.007){
      if($("loop").checked){audio.currentTime=playRange.start;}
      else{audio.pause();audio.currentTime=playRange.end;playRange=null;}
    }
    if(!audio.paused)frame=requestAnimationFrame(clocks);else frame=0;
  }
  function updateRange(){
    try{const v=core.range(Number($("start").value),Number($("end").value),Number($("context").value),item.duration);$("playRange").textContent="播放 "+v.start.toFixed(3)+"–"+v.end.toFixed(3)+" s";}
    catch(e){$("playRange").textContent=e.message;}
  }
  function select(start,end,title){stop();$("start").value=start;$("end").value=end;$("selected").textContent=title+" · "+start.toFixed(6)+"–"+end.toFixed(6)+" s";if(!$("play").disabled)audio.currentTime=start;updateRange();clocks();}
  function draw(){
    const v=viewport();$("windowLabel").textContent=v.start.toFixed(1)+"–"+v.end.toFixed(1)+" s";$("previous").disabled=v.start<=0;$("next").disabled=v.end>=item.duration;
    $("ruler").replaceChildren();for(let i=0;i<=4;i++){const t=el("span",(v.start+(v.end-v.start)*i/4).toFixed(1));t.style.left=(i*25)+"%";$("ruler").append(t);}
    $("tracks").replaceChildren();
    for(const t of item.tracks){
      const group=["harmony_reference","key_reference","pulse_reference","bar_reference"].includes(t.id),kind=t.id.includes("candidate")?"candidate":t.id.includes("baseline")?"baseline":t.kind;
      const row=el("div",undefined,"track "+kind+(group?" group":""));row.id="track-"+t.id;row.append(el("div",t.title,"track-title"));const lane=el("div",undefined,"timeline");
      if(t.kind==="events"){
        for(const time of t.events){if(time<v.start||time>=v.end)continue;const b=el("button",undefined,"event");b.type="button";b.style.left=(100*(time-v.start)/(v.end-v.start))+"%";b.title=t.title+" · "+time.toFixed(6)+" s";b.setAttribute("aria-label",b.title);b.onclick=()=>select(Math.max(0,time-.25),Math.min(item.duration,time+.25),t.title);lane.append(b);}
      }else{
        for(const s of t.rows){const a=Math.max(v.start,s.start),z=Math.min(v.end,s.end);if(z<=a)continue;
          const b=el("button",s.label,"segment"+(["mixed","ambiguous"].includes(s.status)?" uncertain":""));b.type="button";b.style.left=(100*(a-v.start)/(v.end-v.start))+"%";b.style.width=(100*(z-a)/(v.end-v.start))+"%";b.title=t.title+" · "+s.label+" · "+s.start.toFixed(6)+"–"+s.end.toFixed(6)+" s";b.setAttribute("aria-label",b.title);b.onclick=()=>select(s.start,s.end,t.title+" · "+s.label);lane.append(b);
        }
      }
      const playhead=el("span",undefined,"playhead");playhead.hidden=true;lane.append(playhead);row.append(lane);$("tracks").append(row);
    }
    clocks();
  }
  function render(){
    $("caseTitle").textContent=item.title;$("caseDomain").textContent=item.dataset.toUpperCase()+" / "+item.cohort;
    $("caseSubtitle").textContent=item.id+" · "+item.renderer;$("duration").textContent=item.duration.toFixed(3)+" s";
    $("globalKeys").textContent="全局调性规格："+item.global_key.reference+"　原 FFT："+item.global_key.baseline+"　候选："+item.global_key.candidate+"。局部变化见下方，未强制跟随全局调。";
    $("regressions").replaceChildren(el("span","旧 DEV · 已用于诊断","badge"),el("span","实际训练资格：否","badge"));
    for(const m of item.regressions)$("regressions").append(el("span",names[m]+"退步","badge regressed"));
    if(item.native_split==="train")$("regressions").append(el("span","历史原生 split=train · 不改写","badge"));
    $("metrics").replaceChildren();
    for(const [k,label] of Object.entries(names)){
      const a=item.metrics.baseline[k],b=item.metrics.candidate[k],x=core.rate(a),y=core.rate(b),card=el("div",undefined,"metric");card.append(el("div",label+(k==="downbeat"?" F1":"匹配"),"metric-label"));
      const values=el("div",undefined,"metric-values");values.append(el("span",pct(x)),el("span","→","arrow"),el("span",pct(y),"candidate"));if(x!==null&&y!==null)values.append(el("span",(y>=x?"+":"")+((y-x)*100).toFixed(1)+" pp","delta"+(y<x-1e-10?" negative":"")));card.append(values);
      const denom=k==="downbeat"?"TP / 预测 / 规格　"+a.true_positive+" / "+a.predicted+" / "+a.reference+" → "+b.true_positive+" / "+b.predicted+" / "+b.reference:"匹配 / 分母（秒）　"+a.correct_seconds.toFixed(3)+" / "+a.seconds.toFixed(3)+" → "+b.correct_seconds.toFixed(3)+" / "+b.seconds.toFixed(3);card.append(el("small",denom));$("metrics").append(card);
    }
    $("identity").replaceChildren();for(const [label,value] of [["样例",item.id],["作品组",item.group_id],["WAV SHA256",item.audio_sha256],["作曲标签 SHA256",item.labels_sha256],["来源许可",item.provenance.license||"见冻结作曲来源"],["标签性质",item.label_policy]])$("identity").append(el("p",label+"："+value));
    for(const [k,v] of Object.entries(item.inputs))$("identity").append(el("p",k+" SHA256："+v.sha256));
    $("start").max=item.duration;$("end").max=item.duration;$("start").value=0;$("end").value=Math.min(2,item.duration);$("selected").textContent="选择和弦、调性区间或拍点进行复核。";windowStart=0;updateRange();draw();$("content").hidden=false;
  }
  async function load(id){
    const ticket=++epoch;controller?.abort();controller=new AbortController();const signal=controller.signal;stop();item=null;
    $("play").disabled=$("pause").disabled=true;$("content").hidden=true;audio.onloadedmetadata=null;audio.removeAttribute("src");audio.load();if(url){URL.revokeObjectURL(url);url=null;}
    $("status").textContent="正在核验样例与音频身份…";
    try{
      const entry=index.cases.find(e=>e.id===id);if(!entry)throw Error("未知 DEV 样例");
      const bytes=await fetchBytes(entry.file,signal);if(await hash(bytes)!==entry.sha256)throw Error("样例 JSON SHA 不匹配");if(ticket!==epoch)return;
      item=core.validateCase(JSON.parse(new TextDecoder().decode(bytes)),entry);render();
      const wav=await fetchBytes(item.audio,signal);if(await hash(wav)!==item.audio_sha256)throw Error("音频 SHA 不匹配");if(ticket!==epoch)return;
      url=URL.createObjectURL(new Blob([wav],{type:"audio/wav"}));
      audio.onloadedmetadata=()=>{if(ticket!==epoch)return;if(!number(audio.duration)||Math.abs(audio.duration-item.duration)>1/item.sample_rate+1e-6){$("status").textContent="音频时长与样例身份不一致";return;}
        $("play").disabled=$("pause").disabled=false;$("status").textContent="身份已核验 · 只读试听 · "+item.id;};
      audio.src=url;audio.load();
    }catch(e){if(ticket!==epoch||e.name==="AbortError")return;stop();$("play").disabled=$("pause").disabled=true;$("status").textContent=e.message;}
  }
  function filter(){
    const value=$("filter").value,rows=index.cases.filter(e=>value==="all"||value==="regressed"&&e.regressions.length||value==="source_only"&&e.cohort==="dev_source_only"||value==="seen"&&e.cohort==="dev_family_seen"||value==="unseen"&&e.cohort==="dev_family_unseen"||value==="grammar"&&e.dataset==="grammar");
    $("caseSelect").replaceChildren(...rows.map(e=>{const o=el("option",(e.regressions.length?"↓ ":"")+e.title+" · "+e.id);o.value=e.id;return o;}));$("caseSelect").disabled=!rows.length;$("population").textContent=rows.length+" / 240 段";if(rows.length)load(rows[0].id);
  }
  $("filter").onchange=filter;$("caseSelect").onchange=()=>load($("caseSelect").value);
  $("play").onclick=async()=>{const ticket=epoch;try{playRange=core.range(Number($("start").value),Number($("end").value),Number($("context").value),item.duration);audio.currentTime=playRange.start;audio.playbackRate=Number($("speed").value);await audio.play();if(ticket===epoch){cancelAnimationFrame(frame);clocks();}}catch(e){if(ticket===epoch)$("playRange").textContent=e.message;}};
  $("pause").onclick=stop;$("speed").onchange=()=>{audio.playbackRate=Number($("speed").value);};
  for(const id of ["start","end","context"])$(id).onchange=()=>{stop();if(item)updateRange();};
  audio.onended=()=>{if($("loop").checked&&playRange){audio.currentTime=playRange.start;audio.play().then(()=>{cancelAnimationFrame(frame);clocks();}).catch(()=>stop());}};
  $("windowSize").onchange=()=>{if(!item)return;windowStart=0;draw();};
  $("previous").onclick=()=>{windowStart=Math.max(0,windowStart-viewport().size);draw();};$("next").onclick=()=>{windowStart=Math.min(item.duration,windowStart+viewport().size);draw();};
  fetch(data+"index.json",{cache:"no-store"}).then(r=>{if(!r.ok)throw Error("本地 DEV 清单不可用");return r.json();}).then(v=>{index=core.validateIndex(v);filter();}).catch(e=>{$("status").textContent=e.message;});
})();
