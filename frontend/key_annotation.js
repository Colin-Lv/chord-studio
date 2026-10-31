"use strict";
const $=id=>document.getElementById(id),pid=new URLSearchParams(location.search).get("project"),player=$("listen");
let current,entryContext=null,playRange=null,frame=null,saving=false,loadedFromRevision=null,formDirty=false,formVersion=0;

function markDirty(){formDirty=true;formVersion++;}
$('annotationForm').addEventListener('input',markDirty);
$('annotationForm').addEventListener('change',markDirty);
window.addEventListener('beforeunload',event=>{if(formDirty){event.preventDefault();event.returnValue='';}});
function node(tag,className,text){const value=document.createElement(tag);if(className)value.className=className;if(text!==undefined)value.textContent=text;return value;}
function renderFragments(){
  const view=KeyListeningHistory.fragmentEntries(current.history,current.duration);
  $('savedFragments').hidden=!view.locks;$('fragmentList').replaceChildren();
  $('fragmentCount').textContent=`${view.entries.length} 个范围 · ${view.locks} 次锁定`+(view.entries.length>50?' · 显示最近 50 个':'');
  for(const event of view.entries.slice(0,50)){
    const label=event.label,card=node('article','annotation-fragment'),top=node('div','annotation-fragment-heading');
    top.append(node('strong','',KeyListeningHistory.summary(label)),node('span','muted',`修订 ${event.revision}`));
    const timeline=node('div','annotation-fragment-track'),span=node('i');
    span.style.left=(100*label.excerpt_start/current.duration)+'%';span.style.width=(100*(label.excerpt_end-label.excerpt_start)/current.duration)+'%';timeline.append(span);
    card.append(top,node('p','annotation-fragment-range',`${label.excerpt_start}–${label.excerpt_end} 秒 · ${label.label_scope==='full_recording'?'整曲':'片段'}`),timeline);
    const exposure=event.prediction_visible_before_lock?'已见模型建议':event.prior_exposure_self_report==='not_seen'?'自述未见建议，尚未独立核实':'此前暴露未知';
    card.append(node('p','muted',`${label.reviewer} · ${exposure}`),node('p','annotation-fragment-basis',label.basis));
    const load=node('button','secondary','载入为新草稿');load.type='button';load.disabled=saving;load.dataset.revision=String(event.revision);load.onclick=()=>loadFragment(event);card.append(load);$('fragmentList').append(card);
  }
  if(view.skipped)$('fragmentList').append(node('p','muted',`${view.skipped} 条历史记录格式无法安全载入，完整原文仍保留在历史和导出中。`));
}
function loadFragment(event){
  if(saving)return;if(formDirty&&!confirm('载入会替换尚未锁定的表单内容。继续？'))return;
  stopRange(true);const label=event.label;
  for(const [id,key] of [['start','excerpt_start'],['end','excerpt_end'],['scope','label_scope'],['status','annotation_status'],['tonic','tonic'],['mode','mode'],['originalMode','original_mode_label'],['basis','basis'],['recording','recording_id'],['reviewer','reviewer']])$(id).value=label[key]??'';
  loadedFromRevision=event.revision;markDirty();$('rangeMessage').textContent=`已载入修订 ${event.revision} 的精确范围；点击试听回听，尚未保存新记录。`;
  $('message').textContent=`草稿依据修订 ${event.revision}；修改后再次锁定会新增记录。`;$('start').focus();
}
$('freshLabel').onclick=()=>{
  if(!current||saving)return;if(formDirty&&!confirm('清空尚未锁定的判断，保留当前时间范围并另记片段。继续？'))return;
  stopRange(true);for(const id of ['tonic','mode','originalMode','basis'])$(id).value='';$('status').value='unknown';$('recording').value=current.audio_sha256;
  loadedFromRevision=null;markDirty();$('message').textContent='已清空判断，当前时间范围保留；填写后明确锁定才会保存。';
};

async function request(suffix,body){
  const response=await fetch(`/api/projects/${pid}/key-annotation${suffix}`,body?{
    method:"POST",headers:{"X-Harmony-Client":"studio","Content-Type":"application/json"},body:JSON.stringify(body)
  }:{});
  const value=await response.json();
  if(!response.ok)throw Error(value.detail);
  return value;
}
function render(){
  $("title").textContent=current.title;$("note").textContent=current.blindness_note;
  const known=!!entryContext||current.prior_prediction_exposure_known;
  if(known)$("note").textContent="已记录模型建议暴露。听辨记录仍需独立审核，不自动作为金标准。";
  if(known){$("priorExposure").value="seen";$("priorExposure").disabled=true;$("entryNote").hidden=false;$("entryNote").textContent=entryContext?`从工作区带入 ${entryContext.excerpt_start}–${entryContext.excerpt_end} 秒。你已在工作区看到模型结果，此次记录将保留预测暴露状态；没有带入任何主调答案。`:"此前已记录看过模型建议；后续修订继续保留这一事实。";}
  $("reveal").disabled=saving||!current.locked_label;$("lock").disabled=saving;
  $('freshLabel').disabled=saving;renderFragments();
  $("history").textContent=JSON.stringify(current.history,null,2);
  $("export").href=`/api/projects/${pid}/key-annotation`;
  for(const id of ["listenRange","loopRange","rangeStartHere","rangeEndHere"])$(id).disabled=false;
}
function predictionSummary(value){
  const keyName=key=>typeof key==="string"&&key?key.replace(/m$/,"")+(key.endsWith("m")?" 小调":" 大调"):"未给出";
  const review=value.review,proposal=review?.suggestion;
  const pieces=["原自动主调："+keyName(review?.automatic?.key??value.cnn?.key)];
  pieces.push(proposal?.available?"辅助整曲建议："+keyName(proposal.prediction?.key):"辅助整曲建议尚未生成或不可用");
  if(review?.manual)pieces.push("已有整曲人工选择："+keyName(review.effective_key)+"（辅助记录，未经独立验证）");
  $("predictionSummary").textContent=pieces.join("；");
}
function rangeValues(){
  const a=$("start").value,b=$("end").value,start=Number(a),end=Number(b);
  if(!current||!a.trim()||!b.trim()||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>current.duration)
    throw Error("请填写录音范围内的有效起止时间，结束须晚于开始。");
  return {start,end};
}
function stopRange(pause=false){
  playRange=null;if(frame)cancelAnimationFrame(frame);frame=null;
  $("loopRange").setAttribute("aria-pressed","false");
  if(pause)player.pause();
}
function tickRange(){
  frame=null;const range=playRange;
  if(!range||player.paused)return;
  if(player.currentTime>=range.end){
    if(range.repeat)player.currentTime=range.start;
    else{stopRange(true);player.currentTime=range.end;$("rangeMessage").textContent=`试听结束 · ${range.start}–${range.end} 秒`;return;}
  }
  frame=requestAnimationFrame(tickRange);
}
async function listenRange(repeat){
  let active;
  try{
    const range=rangeValues();stopRange();playRange={...range,repeat};player.currentTime=range.start;
    $("loopRange").setAttribute("aria-pressed",String(repeat));
    $("rangeMessage").textContent=`${repeat?"循环":"试听"} ${range.start}–${range.end} 秒 · 只播放，不保存标注`;
    active=playRange;await player.play();
    if(playRange===active&&!frame)frame=requestAnimationFrame(tickRange);
  }catch(error){if(!active||playRange===active){stopRange();$("rangeMessage").textContent=error.message;}}
}
function rangeChanged(){
  stopRange(true);
  $("scope").value=current&&Number($("start").value)===0&&Number($("end").value)===current.duration?"full_recording":"excerpt";
  $("rangeMessage").textContent="范围已调整；点击试听或循环进行核对。";
}
for(const id of ["start","end"])$(id).addEventListener("input",rangeChanged);
$("listenRange").onclick=()=>listenRange(false);
$("loopRange").onclick=()=>{if(playRange?.repeat){stopRange(true);$("rangeMessage").textContent="循环已停止。";}else listenRange(true);};
$("stopRange").onclick=()=>{stopRange(true);$("rangeMessage").textContent="试听已停止。";};
for(const [id,field] of [["rangeStartHere","start"],["rangeEndHere","end"]])$(id).onclick=()=>{
  if(!current||!Number.isFinite(player.currentTime))return;
  $(field).value=String(Math.min(current.duration,Math.max(0,player.currentTime)));rangeChanged();
};
$("scope").onchange=()=>{stopRange(true);if(current&&$("scope").value==="full_recording"){$("start").value="0";$("end").value=String(current.duration);}$("rangeMessage").textContent="听辨范围已更新。";};
player.addEventListener("play",()=>{if(playRange&&!frame)frame=requestAnimationFrame(tickRange);});
player.addEventListener("pause",()=>{if(frame)cancelAnimationFrame(frame);frame=null;});
player.addEventListener("ended",()=>{
  const active=playRange;
  if(active?.repeat){player.currentTime=active.start;player.play().catch(error=>{
    if(playRange===active){stopRange();$("rangeMessage").textContent=error.message;}
  });}else stopRange();
});
player.addEventListener("seeking",()=>{
  if(playRange&&(player.currentTime<playRange.start-.001||player.currentTime>playRange.end+.001)){
    stopRange();$("rangeMessage").textContent="已跳出所选范围，区间播放已停止。";
  }
});

async function update(action,label={}){
  if(!current||saving)return;const version=formVersion;saving=true;render();
  try{
    current=await request("",{run_id:current.run_id,audio_sha256:current.audio_sha256,revision:current.revision,action,label,entry_context:entryContext,prior_exposure:$("priorExposure").value,based_on_revision:loadedFromRevision});
    if(action==='lock'){loadedFromRevision=current.locked_label.revision;if(formVersion===version)formDirty=false;$('rangeMessage').textContent=`修订 ${current.revision} 已保存`+(formDirty?'；当前表单仍有未锁定修改。':'；可继续试听或载入其他片段。');}
    $("message").textContent="已保存修订 "+current.revision+(formDirty?' · 表单仍有尚未锁定的修改':'');
    if(action==="reveal"){
      const value=await request("/predictions");$("prediction").hidden=false;predictionSummary(value);$("evidence").textContent=JSON.stringify(value,null,2);
    }
  }catch(error){$("message").textContent=error.message;}
  finally{saving=false;render();}
}
$("annotationForm").addEventListener("submit",event=>{
  event.preventDefault();
  try{
    const range=rangeValues();
    update("lock",{recording_id:$("recording").value,reviewer:$("reviewer").value,basis:$("basis").value,
      excerpt_start:range.start,excerpt_end:range.end,label_scope:$("scope").value,annotation_status:$("status").value,
      tonic:$("tonic").value||null,mode:$("mode").value||null,original_mode_label:$("originalMode").value||null});
  }catch(error){$("message").textContent=error.message;}
});
$("reveal").onclick=()=>update("reveal");
for(const note of ["C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"])$("tonic").append(new Option(note,note));
request("").then(value=>{
  const handoff=KeyListeningHandoff.parse(new URLSearchParams(location.search),value);
  current=value;entryContext=handoff;player.src=value.audio_url;$("recording").value=value.audio_sha256;
  $("start").value=String(handoff?.excerpt_start??0);$("end").value=String(handoff?.excerpt_end??value.duration);
  $("scope").value=Number($("start").value)===0&&Number($("end").value)===value.duration?"full_recording":"excerpt";render();
}).catch(error=>$("message").textContent=error.message);
