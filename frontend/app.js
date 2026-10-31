"use strict";
const $ = id => document.getElementById(id);
const notation = window.HarmonyNotation, audio = $("audio");
const state = {tab:"file",file:null,project:null,result:null,source:null,selected:0,active:-1,
  review:null,key:"",keyOrigin:"unknown",mode:"roman",drafts:new Map(),cards:new Map(),rows:[],
  busy:false,job:null,poll:null,navigation:0,objectUrl:null,loop:false,auditionEnd:null,frame:null};
const statusNames = {ready:"待分析",queued:"排队中",running:"分析中",completed:"已完成",failed:"失败，可重试",interrupted:"已中断"};
const fmt = (time,precise=false) => {const t=Math.max(0,Number(time)||0), min=Math.floor(t/60),sec=t%60;return String(min).padStart(2,"0")+":"+(precise?sec.toFixed(2).padStart(5,"0"):String(Math.floor(sec)).padStart(2,"0"));};
const element=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
function toast(message){$("toast").textContent=message;$("toast").hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$("toast").hidden=true,6500);}
async function api(url,options={}){const response=await fetch(url,{...options,headers:{"X-Harmony-Client":"studio",...options.headers}});if(!response.ok){let message;try{const body=await response.json();message=typeof body.detail==="string"?body.detail:JSON.stringify(body.detail);}catch(_){message=response.statusText;}throw new Error(message||"请求失败");}return response.json();}
const post=(url,body)=>api(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
function duration(){return state.result?.duration||(Number.isFinite(audio.duration)?audio.duration:0);}
function seek(time){if(!audio.src)return;audio.currentTime=Math.max(0,Math.min(duration(),time));tick();}
function at(rows,time){let lo=0,hi=rows.length;while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid].start<=time)lo=mid+1;else hi=mid;}const i=lo-1;return i>=0&&time<rows[i].end?i:-1;}
function currentDraft(){return state.drafts.get(state.source.id);}
function chord(source,index){return state.drafts.get(source.id)?.edits.get(index)??source.rows[index].label;}
function keyName(value){return value?value.replace(/m$/,"")+(value.endsWith("m")?" 小调":" 大调"):"未确定";}
function renderChordKeySuggestion(automatic){
  const button=$("adoptChordKey"), hint=$("keyHint");
  const evidence=state.result?.key_analysis?.chord_evidence;
  if(!button||!evidence||!evidence.key){if(button)button.hidden=true;return;}
  const differs=evidence.key!==automatic&&state.key!==evidence.key;
  state.chordKeySuggestion=evidence.key;
  button.hidden=!differs;
  if(differs)button.textContent=`采用和弦证据：${keyName(evidence.key)}（margin ${evidence.margin>0?"+":""}${Number(evidence.margin).toFixed(2)}）`;
  if(hint)hint.textContent+=` 和弦证据（${evidence.chords} 个和弦）：${keyName(evidence.key)}，margin ${evidence.margin>0?"+":""}${Number(evidence.margin).toFixed(2)}${state.key===evidence.key?"（已采用）":""}。`;
}
function modelDisplayText(result){const identity=(result.model_identity||{}).retained10;if(result.profile==='retained10'&&identity&&identity.version){const frozen=identity.frozen_utc?` · 冻结 ${identity.frozen_utc.slice(0,10)}（北京）`:'';return `当前显示结果：${identity.version}`+(identity.entry?` · 音频链 ${identity.entry}`:'')+frozen+(identity.weights_sha256?` · 权重 ${identity.weights_sha256.slice(0,8)}`:'');}return `当前显示结果：${result.profile_name||result.profile}`+(result.profile==='retained10'?' · 冻结模型':' · 历史结果；重新分析将使用上方所选模型');}
function shown(label){const parsed=notation.parse(label), name=parsed?.chordDisplay||label,roman=parsed?.romanDisplay;
  if(!state.key)return {main:name,sub:state.mode==="roman"?"主调未定":""};
  if(!roman)return {main:name,sub:"记法未解析"};
  return state.mode==="roman"?{main:roman,sub:roman===name?"":name}:{main:name,sub:roman===name?"":roman};}
function setLabel(primary,secondary,label){const value=shown(label);primary.textContent=value.main;primary.title=label;secondary.textContent=value.sub;}
function draftKey(source){return "harmony-web:"+state.result.project_id+":"+state.result.run_id+":"+source.id+":"+source.sha256;}
function remember(){
  if(!state.result||!sourceDraftGuard(false))return;
  const draft=currentDraft();draft.dirty=true;draft.version=(draft.version||0)+1;
  const raw=HarmonySourceDraft.encode(sourceDraftIdentity(),draft,state.mode);let retained=false;
  try{localStorage.setItem(draftKey(state.source),raw);draft.browserRaw=raw;retained=true;}catch(_){toast("浏览器无法暂存，请保存到项目或导出当前草稿。");}
  $("saveStatus").textContent=retained?"有未保存修改 · 已暂存于此浏览器":"有未保存修改 · 浏览器暂存失败，请保存或导出";
}
function sourceDraftIdentity(source=state.source){return HarmonySourceDraft.identity(state.result,source);}
function readSourceDraft(raw,source=state.source,result=state.result){return HarmonySourceDraft.read(raw,HarmonySourceDraft.identity(result,source),source.rows.length,label=>label==="X"||label==="N"||!!notation.parse(label));}
function sourceDraftGuard(notify=true){
  const draft=currentDraft();if(!draft)return false;
  if(!draft.cached){try{const actual=localStorage.getItem(draftKey(state.source));if(actual!==draft.browserRaw)draft.cached=readSourceDraft(actual)||{raw:"",problem:"浏览器草稿已在另一窗口改变，请重新读取。"};}catch(_){}}
  if(draft.cached){renderSourceDraftControls();if(notify)toast("请先处理保留的浏览器草稿；当前修改可单独导出。");return false;}return true;
}
function downloadDraftText(raw,name){const url=URL.createObjectURL(new Blob([raw],{type:"application/json"})),link=element("a");link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function backupSourceText(raw,key){if(raw===null||raw===undefined)return;const backup=key+":backup:"+new Date().toISOString()+":"+crypto.randomUUID();localStorage.setItem(backup,raw);if(localStorage.getItem(backup)!==raw)throw Error("浏览器备份无法验证，原草稿仍保留。");}
function renderSourceDraftControls(){
  if(!state.source)return;const draft=currentDraft(),cached=draft.cached,panel=$("sourceDraftPanel"),blocked=!!cached;
  panel.hidden=!blocked;
  for(const id of ["chordInput","applyEdit","saveCorrections"])$(id).disabled=blocked||!!draft.saving;
  $("restoreChord").disabled=blocked||!!draft.saving||!draft.edits.has(state.selected);
  if(blocked){
    $("sourceDraftMessage").textContent=cached.problem||(cached.value.revision!==draft.revision?`草稿修订 ${cached.value.revision} 与项目修订 ${draft.revision} 不同，请导出核对。`:"发现同一来源的浏览器草稿，尚未恢复；当前显示项目保存的内容。");
    $("sourceDraftRestore").disabled=!!cached.problem||cached.value?.revision!==draft.revision||!!draft.saving;
    $("sourceDraftReload").disabled=!!draft.saving;
    $("sourceDraftRaw").textContent=cached.raw.length>12000?cached.raw.slice(0,12000)+"\n（预览已截短，导出保留全文）":cached.raw;
  }
  const list=$("sourceDraftBackups"),prefix=draftKey(state.source)+":backup:";list.replaceChildren();
  try{const keys=[];for(let i=0;i<localStorage.length;i++){const key=localStorage.key(i);if(key?.startsWith(prefix))keys.push(key);}keys.sort().reverse();
    if(keys.length){const details=element("details");details.append(element("summary","",`此来源的浏览器备份（${keys.length}）`));for(const key of keys){const button=element("button","text-button",key.slice(prefix.length,prefix.length+24)+" · 导出备份");button.onclick=()=>downloadDraftText(localStorage.getItem(key)||"","source_draft_backup.json");details.append(button);}list.append(details);}
  }catch(_){}
}
function restoreSourceDraft(){
  const draft=currentDraft(),cached=draft.cached;
  if(!cached||cached.problem||cached.value.revision!==draft.revision)return;
  try{if(localStorage.getItem(draftKey(state.source))!==cached.raw)throw Error("另一窗口已改变草稿，请重新读取。");
    if(draft.dirty)backupSourceText(HarmonySourceDraft.encode(sourceDraftIdentity(),draft,state.mode),draftKey(state.source));
  }catch(error){toast(error.message);return;}
  draft.edits=new Map(cached.value.edits);draft.dirty=true;draft.version=(draft.version||0)+1;draft.browserRaw=cached.raw;draft.cached=null;
  state.mode=cached.value.display;notation.setKey(state.key);renderKey();renderRows();selectRow(state.selected);tick();
  $("saveStatus").textContent="已恢复浏览器和弦草稿，尚未保存到项目；主调以当前审阅记录为准。";
}
async function reloadSourceDraft(){
  const source=state.source,result=state.result,draft=currentDraft(),key=draftKey(source);if(!draft.cached||draft.saving)return;
  draft.saving=true;renderSourceDraftControls();
  try{
    const raw=localStorage.getItem(key);backupSourceText(raw,key);
    if(draft.dirty)backupSourceText(HarmonySourceDraft.encode(sourceDraftIdentity(source),draft,state.mode),key);
    const saved=await api(`/api/projects/${result.project_id}/corrections/${source.id}`);
    if(localStorage.getItem(key)!==raw)throw Error("另一窗口又改变了草稿，已保留备份，请重新核对。");
    localStorage.removeItem(key);
    if(state.result!==result)return;
    state.drafts.set(source.id,{revision:saved.revision,edits:new Map(saved.edits.map(e=>[e.index,e.label])),dirty:false,browserRaw:null,cached:null});
    if(state.source.id===source.id){renderRows();selectRow(state.selected);tick();$("saveStatus").textContent="已保留浏览器备份，并重新读取项目保存内容。";}
  }catch(error){toast(error.message);$("saveStatus").textContent=error.message;}
  finally{draft.saving=false;if(state.result===result&&state.source.id===source.id)renderSourceDraftControls();}
}
function stopPolling(){clearTimeout(state.poll);state.poll=null;state.job=null;}
function setBusy(on){state.busy=on;$("analyze").disabled=on||(state.tab==="file"?!state.file&&!state.project?.audio:!$("bilibiliUrl").value.trim());$("audioFile").disabled=on;$("newProject").disabled=on;}
function showTab(tab){state.tab=tab;$("filePanel").hidden=tab!=="file";$("linkPanel").hidden=tab!=="link";$("tabFile").setAttribute("aria-selected",String(tab==="file"));$("tabLink").setAttribute("aria-selected",String(tab==="link"));$("analyze").replaceChildren(document.createTextNode(tab==="link"?"获取音频并分析":"开始分析"),element("span","","→"));setBusy(state.busy);}
function clearResult(){window.BarEditor?.close();state.result=null;state.review=null;state.source=null;state.drafts.clear();state.cards.clear();state.rows=[];state.active=-1;state.loop=false;state.auditionEnd=null;$("resultPanel").hidden=true;$("loop").disabled=true;$("loop").setAttribute("aria-pressed","false");$("nowChord").textContent="准备试听";$("nowSecondary").textContent="";}
function setAudio(url,title){audio.pause();if(state.objectUrl&&url!==state.objectUrl){URL.revokeObjectURL(state.objectUrl);state.objectUrl=null;}audio.src=url;$("songTitle").textContent=title;$("playerPanel").hidden=false;$("playStatus").textContent="正在加载音频…";$("emptyState").hidden=true;}
function selectFile(file){if(!file||state.busy)return;state.navigation++;stopPolling();state.project=null;state.file=file;clearResult();$("jobPanel").hidden=true;showTab("file");$("fileTitle").textContent=file.name;$("fileMeta").textContent=(file.size/1024/1024).toFixed(1)+" MB · 点击可重新选择";$("inputStatus").textContent="已选择音乐，可以试听或开始分析。";const url=URL.createObjectURL(file);setAudio(url,file.name);state.objectUrl=url;setBusy(false);}
async function refreshProjects(){const projects=await api("/api/projects");const fragment=document.createDocumentFragment();for(const project of projects){const button=element("button","project-item");button.classList.toggle("active",project.id===state.project?.id);button.append(element("strong","",project.title),element("small","",statusNames[project.status]||project.status));button.title=project.title;button.addEventListener("click",()=>openProject(project.id).catch(error=>toast(error.message)));fragment.append(button);}$("projectList").replaceChildren(fragment);}
async function openProject(id){if(state.busy)return toast("请等待当前提交完成。");const navigation=++state.navigation;stopPolling();audio.pause();const project=await api("/api/projects/"+id);if(navigation!==state.navigation)return;state.project=project;state.file=null;clearResult();$("jobPanel").hidden=true;$("fileTitle").textContent=project.original_filename||project.title;$("fileMeta").textContent="已保存在本机 · 可切换模型重新分析";$("inputStatus").textContent="当前项目："+project.title;showTab("file");if(project.audio)setAudio(`/api/projects/${id}/audio`,project.title);else $("playerPanel").hidden=true;
  if(project.current_run)await loadResult(id,navigation);
  if(project.active_job)watchJob(project.active_job,navigation);else if(["failed","interrupted"].includes(project.status))$("inputStatus").textContent="上次任务未完成，可点击开始分析重试。";
  setBusy(false);await refreshProjects();}
async function beginAnalysis(){if(state.busy)return;const navigation=state.navigation;setBusy(true);try{let job;
  if(state.tab==="link"){job=await post("/api/bilibili",{url:$("bilibiliUrl").value.trim(),profile:$("model").value,key_mode:$("keyMode").value});state.project=await api("/api/projects/"+job.project_id);clearResult();audio.pause();$("playerPanel").hidden=true;}
  else{if(state.file){$("inputStatus").textContent="正在上传音频…";const form=new FormData();form.append("file",state.file);state.project=await api("/api/upload",{method:"POST",body:form});state.file=null;}if(!state.project)throw new Error("请先选择音乐");job=await post(`/api/projects/${state.project.id}/analyze`,{profile:$("model").value,key_mode:$("keyMode").value});}
  if(navigation!==state.navigation)return;$("inputStatus").textContent="任务已提交，可以在下方查看进度。";watchJob(job.id,navigation);await refreshProjects();
  }catch(error){$("inputStatus").textContent=error.message;toast(error.message);}finally{setBusy(false);}}
async function watchJob(id,navigation){clearTimeout(state.poll);state.job=id;$("jobPanel").hidden=false;try{const job=await api("/api/jobs/"+id);if(navigation!==state.navigation||state.job!==id)return;$("jobTitle").textContent=job.stage_label;$("jobState").textContent=statusNames[job.status]||job.status;$("jobLog").href=`/api/jobs/${id}/log`;$("jobActivity").className="activity "+(["running","queued"].includes(job.status)?"running":"done");$("stageList").replaceChildren(...job.stages.map(stage=>element("li",stage.status==="failed"?"failed":"",stage.label+" · "+stage.seconds.toFixed(1)+"s")));$("jobError").hidden=!job.error;$("jobError").textContent=job.error||"";$("retry").hidden=!["failed","interrupted"].includes(job.status);
  if(job.status==="completed"){state.project=await api("/api/projects/"+job.project_id);if(navigation!==state.navigation)return;await loadResult(job.project_id,navigation);await refreshProjects();return;}
  if(["failed","interrupted"].includes(job.status)){await refreshProjects();return;}
  state.poll=setTimeout(()=>watchJob(id,navigation),1500);
  }catch(error){if(navigation!==state.navigation)return;$("jobError").hidden=false;$("jobError").textContent="暂时无法读取任务状态："+error.message+"。将自动重试。";state.poll=setTimeout(()=>watchJob(id,navigation),3000);}}
async function loadResult(projectId,navigation){const result=await api(`/api/projects/${projectId}/result`);const review=await api(`/api/projects/${projectId}/key-review`);const corrections=await Promise.all(result.sources.map(source=>api(`/api/projects/${projectId}/corrections/${source.id}`)));if(navigation!==state.navigation)return;
  state.result=result;state.review=review;state.source=result.sources.find(source=>source.id===result.default_source)||result.sources[0];state.selected=0;state.active=-1;state.drafts.clear();let displayPreferences=null;
  result.sources.forEach((source,index)=>{const correction=corrections[index],draft={revision:correction.revision,edits:new Map(correction.edits.map(edit=>[edit.index,edit.label])),dirty:false,browserRaw:null,cached:null};
    try{draft.browserRaw=localStorage.getItem(draftKey(source));draft.cached=readSourceDraft(draft.browserRaw,source);}catch(_){toast("浏览器暂存无法读取，请及时保存或导出当前修改。");}
    state.drafts.set(source.id,draft);const preferences=correction;
    if(preferences.key!==undefined&&(!displayPreferences||(preferences.updated_at||preferences.saved_at||"")>(displayPreferences.updated_at||displayPreferences.saved_at||"")))displayPreferences=preferences;
  });
  $("model").value="retained10";$("resultModel").textContent=modelDisplayText(result);$("keyMode").value=result.key_mode||"madmom";state.key=displayPreferences?.key??result.key_fusion?.key??result.key_analysis?.key??"";state.keyOrigin=displayPreferences?.key_origin||(state.key?"automatic":"unknown");if(review.revision>0){state.key=review.effective_key||"";state.keyOrigin=review.manual?"manual":(state.key?"automatic":"unknown");}state.mode=displayPreferences?.display||"roman";notation.setKey(state.key);$("analysisDownload").href=`/api/projects/${projectId}/result`;$("resultPanel").hidden=false;$("emptyState").hidden=true;$("loop").disabled=false;
  $("warnings").textContent=(result.profile==="v2"?"实验 v2 与冻结 v1 的训练歌曲数不同；当前结果不代表准确率更高。\n":"")+result.warnings.join("\n");
  if(!audio.src.endsWith(result.audio.url))setAudio(result.audio.url,result.title);$("seek").max=result.duration;renderKey();renderSources();switchSource(state.source);renderStructure();tick();}
function renderKeyEvidence(){
  const panel=$("keyEvidence"),fusion=state.result.key_fusion;
  panel.replaceChildren();panel.hidden=!fusion;if(!fusion)return;
  panel.append(element("h3","","原分歧裁决 · 实验"),element("p","",`该策略原结果：${keyName(fusion.key)} · ${fusion.needs_review?"待审核":"模型一致，未经验证"}`));
  const names={madmom_cnn:"CNN 原判",raw_chord_functional:"原全曲和声",functional_joint:"辅助联合分析"};
  for(const [id,name] of Object.entries(names)){const evidence=fusion.evidence?.[id];panel.append(element("p","",`${name}：${keyName(evidence?.key)}；${(evidence?.candidates||[]).slice(0,3).map(r=>`${r.key}（分数 ${Number(r.score).toFixed(4)}）`).join("、")}`));}
  const timeline=fusion.evidence?.window_timeline,local=element("details");local.append(element("summary","","局部调性候选 · 未验证"),element("p","muted","原局部结果只供诊断；候选变化不等于已确认转调，不自动改变全局主调或小节级数。"));
  local.append(element("pre","review-json",JSON.stringify({bar_shares:timeline?.bar_shares,valid_bars:timeline?.valid_bars,decisive_bars:timeline?.decisive_bars,unresolved_bars:timeline?.unresolved_bars,candidate_metrics:timeline?.candidate_metrics},null,2)));
  for(const segment of timeline?.segments||[]){const button=element("button","secondary",`${fmt(segment.start)}–${fmt(segment.end)} ${segment.key} · 未验证`);button.addEventListener("click",()=>seek(segment.start));local.append(button);}panel.append(local);
  panel.append(element("p","muted","原裁决器与门槛保持原样，来源内分数不是已校准正确概率。和声主调 v2 在独立审阅开关中展示。"));
  const thresholds=fusion.policy?.disagreement_thresholds;if(thresholds&&thresholds.cnn>1&&thresholds.harmonic>1)panel.append(element("p","muted","当前校准配置尚未启用自动分歧裁决。"));
}
function renderKeyExperimental(){
  const panel=$("keyExperimental"),block=state.result?.key_analysis?.experimental;
  if(!panel)return;panel.replaceChildren();panel.hidden=!block;if(!block)return;
  const active=state.result?.key_mode==="continuous"&&state.result?.key_analysis?.source==="continuous_key_v1";
  const keyIdentity=value=>{const m=/^([A-G])([#b]?)(.*)$/.exec(String(value||"").trim());if(!m)return null;const pc={C:0,D:2,E:4,F:5,G:7,A:9,B:11}[m[1]]+({"#":1,"b":-1,"":0}[m[2]]||0);const rest=m[3]||"";const minor=/^m(?!aj)/.test(rest)||/^:m(in(or)?)?$/.test(rest)||/minor$/i.test(rest);return((pc%12)+12)%12+(minor?"m":"");};
  const same=keyIdentity(block.key)===keyIdentity(state.key)&&state.key!=="";
  panel.append(element("h3","",active?"连续评分 · 实验（当前主调来源）":"连续评分 · 实验（第二档）"));
  let summary=`实验主调 ${keyName(block.key)}`;
  summary+=` · ${same?"与当前主调一致":"与当前主调不同"}；分差 ${Number(block.gap)>=0?"+":""}${Number(block.gap).toFixed(3)}；`;
  summary+=`声学 CNN 原判 ${keyName(block.acoustic_key)}`;
  if(block.published_key&&block.published_key!==block.acoustic_key)summary+=`；覆盖裁决 ${keyName(block.published_key)}`;
  summary+="。";
  panel.append(element("p","",summary));
  const adopt=element("button","secondary",`采用实验主调：${keyName(block.key)}`);adopt.hidden=same;
  if(!state.review){adopt.disabled=true;adopt.title="审阅记录不可用，无法保存改调";}
  adopt.addEventListener("click",()=>{if(state.review)updateKeyReview("custom",block.key);});
  panel.append(adopt);
  const table=element("table","key-experiment-table"),head=element("tr","");
  for(const title of ["候选","总分","声学项","和声项","调内占比","主和弦占比","正格到达","属回次数"])head.append(element("th","",title));
  table.append(head);
  for(const row of block.ranking||[]){const line=element("tr","");
    line.append(element("td","",keyName(row.key)),element("td","",Number(row.score).toFixed(3)),
      element("td","",Number(row.acoustic_term).toFixed(3)),element("td","",Number(row.harmonic_term).toFixed(3)),
      element("td","",`${(100*row.native_share).toFixed(1)}%`),element("td","",`${(100*row.tonic_share).toFixed(1)}%`),
      element("td","",String(row.authentic_arrivals)),element("td","",String(row.predominant_returns)));
    if(row.key===block.key)line.className="chosen";table.append(line);}
  const details=element("details");details.append(element("summary","",`候选与分项（24 个中前 ${(block.ranking||[]).length}）`),table);
  panel.append(details);
  const parse=block.parse||{};
  const facts=element("details");facts.append(element("summary","","解析统计与已知退步"),
    element("p","",`来源 ${block.source_id||"—"}；标签 ${parse.tokens??"—"} 个，可解析 ${block.chords}，N ${parse.N??0}，X ${parse.X??0}，缺失 ${parse.missing??0}。`));
  for(const item of block.known_regressions||[])facts.append(element("p","",`已知退步 ${item.case}：参考 ${item.reference}，实验给 ${item.experimental}（${item.note}）`));
  panel.append(facts);
  for(const warning of block.warnings||[])panel.append(element("p","muted",warning));
  panel.append(element("p","muted",active?"当前主调即该实验结果；改调仍走手动保存路径，恢复自动估计会回到同一结果。":"该分数不参与默认主调，也不改变已保存的和弦来源；采用后与手动改调走同一条保存路径。"));
}
function renderKey(){const badge=$("appModelBadge"),identity=(state.result?.model_identity||{}).retained10;if(badge)badge.textContent=identity&&identity.version?`${identity.version}${identity.frozen_utc?' · '+String(identity.frozen_utc).slice(0,10):''}`:'本地工作台';renderGlobalReview();renderKeyEvidence();renderKeyExperimental();$("globalKey").textContent=keyName(state.key);$("keyOrigin").textContent=state.keyOrigin==="manual"?"手动设置":state.keyOrigin==="automatic"?(state.result.key_analysis?.source==="continuous_key_v1"?"连续评分 · 实验（未校准）":(state.result.key_fusion?(state.result.key_fusion.needs_review?"实验暂定 · 需审核":"自动融合 · 可调整"):"自动估计 · 可调整")):"请选择主调";$("keySelect").value=state.key;$("displayRoman").setAttribute("aria-pressed",String(state.mode==="roman"));$("displayName").setAttribute("aria-pressed",String(state.mode==="name"));const automatic=state.result.key_fusion?.key||state.result.key_analysis?.key,alternative=state.result.key_analysis?.candidates?.find(item=>item.key!==automatic);$("resetKey").disabled=!automatic;$("keyHint").textContent=(automatic?"自动估计："+keyName(automatic)+(alternative?"；第二候选："+keyName(alternative.key):"")+"。":"没有可用的自动调性结果。")+"级数以当前主调为基准，m 表示小和弦，斜杠后为低音级数；改调仅改变显示，不移调。";renderChordKeySuggestion(automatic);}
function renderGlobalReview(){
  const review=state.review,panel=$("keyReviewContent");panel.replaceChildren();$("keyReviewEnabled").checked=!!review?.enabled;$("adoptKeySuggestion").hidden=true;try{$("keyAnnotationLink").href=KeyListeningHandoff.url({project_id:state.result.project_id,run_id:state.result.run_id,audio_sha256:state.result.audio?.sha256,duration:state.result.duration},0,state.result.duration);$("keyAnnotationLink").removeAttribute("aria-disabled");}catch(_){$("keyAnnotationLink").removeAttribute("href");$("keyAnnotationLink").setAttribute("aria-disabled","true");$("keyAnnotationLink").title="当前分析缺少可验证的录音身份；请重新分析后进入听辨记录。";}$("exportKeyReview").href=`/api/projects/${state.result.project_id}/key-review`;
  if(!review)return;
  if(review.revision>0)$("keyReviewStatus").textContent=review.manual?`人工选择已保存 · 修订 ${review.revision} · 辅助标注，未经独立验证`:`未人工确认 · 修订 ${review.revision} · 自动结果保持原样`;
  if(review.derived_key_analysis_stale)panel.append(element("p","result-note","与原主调相关的段落/功能分析已过期；级数按当前主调重新显示，原结构仅供定位。"));
  if(!review.enabled)return;
  const evidence=review.suggestion;
  if(!evidence?.available){panel.append(element("p","muted","未生成/不可用："+(evidence?.reason||"缺少原始输入")));return;}
  const p=evidence.prediction,cnn=state.result.key_analysis?.key;
  panel.append(element("h3","",cnn===p.key?"模型一致 · 尚未验证":review.manual?"主调存在分歧 · 已记录人工选择":"主调存在分歧，待确认"),element("p","",`原自动结果：${keyName(review.automatic.key)}；和声主调 v2 全曲建议：${keyName(p.key)}`));
  panel.append(element("p","muted",`引擎状态 ${p.status} 仅表示工程分差条件；分数未经校准，不是正确概率。未启用自动覆盖。`));
  const mode=p.mode_candidates_by_tonic?.[p.tonic]?.find(row=>row.mode===p.mode),f=mode?.components;
  if(f)panel.append(element("p","",`建议调的主音和弦时长 ${(100*f.tonic_root_duration).toFixed(1)}%；结尾支持 ${f.ending}；四小节末端弱代理 ${(100*f.phrase_ending).toFixed(1)}%；终止式计数 ${f.cadence_count}。`));
  const globalDetails=element("details");globalDetails.append(element("summary","","完整全曲候选、分项与加权贡献"));const pre=element("pre","review-json",JSON.stringify({tonic_candidates:p.tonic_candidates,mode_candidates_by_tonic:p.mode_candidates_by_tonic,final_key_candidates:p.final_key_candidates,weighted_contributions:evidence.weighted_contributions,policy:p.policy,contract:evidence.inputs.contract},null,2));globalDetails.append(pre);panel.append(globalDetails);
  const local=element("details");local.append(element("summary","","局部调性候选 · 未验证"),element("p","muted","同一和弦来源的诊断，不能独立投票、确认转调或自动改变小节级数。"));
  for(const row of p.window_evidence?.segments||[]){const button=element("button","secondary",`${fmt(row.start)}–${fmt(row.end)} ${row.key} · 未验证`);button.addEventListener("click",()=>seek(row.start));local.append(button);}panel.append(local);
  $("adoptKeySuggestion").hidden=!p.key;$("adoptKeySuggestion").textContent=`采用此建议：${keyName(p.key)}`;
}
async function updateKeyReview(action,key=""){
  if(!state.result||!state.review)return;
  const projectId=state.result.project_id,run=state.result.run_id,review=state.review;
  for(const id of ["keyReviewEnabled","adoptKeySuggestion","keySelect","resetKey"])$(id).disabled=true;
  try{const saved=await post(`/api/projects/${projectId}/key-review`,{run_id:run,audio_sha256:review.audio_sha256,revision:review.revision,action,key});if(state.result?.run_id!==run)return;
    state.review=saved;if(["adopt","custom","restore"].includes(action)||saved.manual||review.revision>0){state.key=saved.effective_key||"";state.keyOrigin=saved.manual?"manual":(state.key?"automatic":"unknown");}
    notation.setKey(state.key);renderKey();renderRows();renderCandidates();tick();window.BarEditor?.refresh?.();
  }catch(error){$("keyReviewStatus").textContent=error.message;toast(error.message);$("keySelect").value=state.key;$("keyReviewEnabled").checked=!!state.review?.enabled;}
  finally{for(const id of ["keyReviewEnabled","adoptKeySuggestion","keySelect","resetKey"])$(id).disabled=false;}
}
function changeNotation(){notation.setKey(state.key);renderKey();renderRows();renderCandidates();tick();window.BarEditor?.refresh?.();remember();}
function renderSources(){state.cards.clear();$("sources").replaceChildren();for(const source of state.result.sources){const button=element("button","source-card"),name=element("span","source-name",source.name),main=element("span","source-chord","—"),sub=element("span","source-secondary","");button.append(name,main,sub);button.setAttribute("aria-pressed","false");button.addEventListener("click",()=>switchSource(source));state.cards.set(source.id,{button,main,sub});$("sources").append(button);}}
function switchSource(source){window.BarEditor?.open({project:state.project,result:state.result,source,audio,onRangePlayback:()=>{state.loop=false;state.auditionEnd=null;$("loop").setAttribute("aria-pressed","false");}});state.source=source;state.selected=Math.max(0,at(source.rows,audio.currentTime));state.loop=false;state.auditionEnd=null;$("loop").setAttribute("aria-pressed","false");for(const [id,card]of state.cards)card.button.setAttribute("aria-pressed",String(id===source.id));$("sourceTitle").textContent=source.name;const prefix=`/api/projects/${state.result.project_id}/export/${source.id}/`;$("exportLab").href=prefix+"corrected.lab";$("exportCorrections").href=prefix+"corrections.json";$("exportOriginal").href=prefix+"original.lab";renderRows();selectRow(state.selected);$("saveStatus").textContent=currentDraft().dirty?"有未保存草稿":"已读取项目保存的修改";tick();}
function renderRows(){const scroll=$("chordRows").scrollTop;const fragment=document.createDocumentFragment();state.rows=state.source.rows.map((row,index)=>{const button=element("button","chord-row"),label=element("span",""),main=element("span","row-main"),sub=element("span","row-sub");setLabel(main,sub,chord(state.source,index));label.append(main,sub);button.append(element("span","row-index",String(index+1).padStart(3,"0")),element("span","row-time",fmt(row.start,true)+"–"+fmt(row.end,true)),label,element("span","row-mark",currentDraft().edits.has(index)?"已修改":"预测"));button.classList.toggle("edited",currentDraft().edits.has(index));button.classList.toggle("selected",index===state.selected);button.title=`${row.start_text} – ${row.end_text} 秒`;button.addEventListener("click",()=>{selectRow(index);state.auditionEnd=null;seek(row.start);});fragment.append(button);return button;});$("chordRows").replaceChildren(fragment);$("chordRows").scrollTop=scroll;state.active=-1;$("rowCount").textContent=state.rows.length+" 个原始区间";$("editCount").textContent=currentDraft().edits.size+" 处修改";}
function selectRow(index){state.selected=index;state.rows.forEach((node,i)=>node.classList.toggle("selected",i===index));const row=state.source.rows[index];$("editTime").textContent=fmt(row.start,true)+" – "+fmt(row.end,true);$("editOriginal").textContent="原始标签："+row.label;$("chordInput").value=chord(state.source,index);$("restoreChord").disabled=!currentDraft().edits.has(index);$("editStatus").textContent="这里仅修改旧区间记录；上方小节编辑轨单独自动保存，播放器显示编辑轨。";renderCandidates();drawWaveform();renderSourceDraftControls();}
function renderCandidates(){const row=state.source.rows[state.selected];const nodes=(row.candidates||[]).map(candidate=>{const button=element("button","candidate"),view=shown(candidate.label);button.append(element("strong","",view.main+(view.sub?" · "+view.sub:"")));const sources=Object.keys(candidate.sources).map(id=>state.result.sources.find(source=>source.id===id)?.name||id);button.append(element("small","",sources.join(" / ")));button.title=candidate.label;button.addEventListener("click",()=>{$("chordInput").value=candidate.label;$("chordInput").focus();});return button;});$("candidates").replaceChildren(...nodes);if(!nodes.length)$("candidates").textContent="此区间没有来源提案。";}
function applyEdit(){if(!sourceDraftGuard())return;const text=$("chordInput").value.trim();if(!text||/[\s]/.test(text))return $("editStatus").textContent="请输入单个和弦标签或级数记号。";const direct=notation.parse(text),parsed=direct||(state.key?notation.parseEditable(text):null);if(!parsed)return $("editStatus").textContent=state.key?"未识别此记法，请检查和弦或级数。":"请先设置主调，或直接输入和弦名称。";const label=direct?text:parsed.chordDisplay;if(label===state.source.rows[state.selected].label)currentDraft().edits.delete(state.selected);else currentDraft().edits.set(state.selected,label);remember();renderRows();selectRow(state.selected);tick();$("editStatus").textContent="已应用："+label+"。原始预测保持不变。";}
function correctionPayload(){return {run_id:state.result.run_id,source_id:state.source.id,source_sha256:state.source.sha256,revision:currentDraft().revision,key:state.key,key_origin:state.keyOrigin,display:state.mode,key_review_revision:state.review?.revision,edits:[...currentDraft().edits].map(([index,label])=>({index,label}))};}
async function saveCorrections(){
  if(!state.result||!sourceDraftGuard())return;
  const result=state.result,source=state.source,draft=currentDraft(),payload=correctionPayload(),version=draft.version||0,storageKey=draftKey(source);
  draft.saving=true;renderSourceDraftControls();
  try{
    const saved=await post(`/api/projects/${result.project_id}/corrections`,payload);draft.revision=saved.revision;draft.dirty=(draft.version||0)!==version;
    try{const actual=localStorage.getItem(storageKey);if(actual===draft.browserRaw){if(!draft.dirty){localStorage.removeItem(storageKey);draft.browserRaw=null;}else{const cached=JSON.parse(actual);cached.revision=saved.revision;const raw=JSON.stringify(cached);localStorage.setItem(storageKey,raw);draft.browserRaw=raw;}}
      else draft.cached=readSourceDraft(actual,source,result)||{raw:"",problem:"保存期间浏览器草稿已在另一窗口改变，请重新读取。"};
    }catch(_){toast("项目已保存，但浏览器暂存状态无法更新，请保留当前草稿导出。");}
    if(state.result===result&&state.source.id===source.id)$("saveStatus").textContent=draft.cached?"项目已保存；另一窗口的浏览器草稿仍保留，需核对。":draft.dirty?"上次修改已保存，仍有新修改尚未保存":"已保存到项目 · 修正 LAB 与修改记录已更新";
    toast("修改已保存");
  }catch(error){if(state.result===result&&state.source.id===source.id)$("saveStatus").textContent=error.message;toast(error.message);}
  finally{draft.saving=false;if(state.result===result&&state.source.id===source.id)renderSourceDraftControls();}
}
function renderStructure(){const result=state.result,sections=result.structure?.sections||[];$("structureSummary").textContent=result.beats.length+" 拍 · "+result.bars.length+" 小节 · "+sections.length+" 段";$("sections").replaceChildren(...sections.map(section=>{const button=element("button","section-chip",section.name+"　"+fmt(section.start));button.title=fmt(section.start)+" – "+fmt(section.end);button.dataset.start=section.start;button.dataset.end=section.end;button.addEventListener("click",()=>seek(section.start));return button;}));window.StructureReview.render(result);}
function drawWaveform(){if(!state.result)return;const canvas=$("waveform"),box=canvas.getBoundingClientRect();if(!box.width)return;const ratio=devicePixelRatio||1,w=box.width,h=130;if(canvas.width!==Math.round(w*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const d=state.result.duration,peaks=state.result.waveform?.peaks||[];ctx.fillStyle="#81ab90";peaks.forEach((peak,index)=>{const height=Math.max(1,peak*61);ctx.fillRect(index/peaks.length*w,40-height/2,Math.max(1,w/peaks.length*.72),height);});for(const beat of state.result.beats){ctx.fillStyle=beat.beat===1?"#acc7ae99":"#acc7ae35";ctx.fillRect(beat.time/d*w,78,beat.beat===1?1.5:1,beat.beat===1?12:6);}state.source.rows.forEach((row,index)=>{ctx.fillStyle=index%2?"#375646":"#4e7560";if(currentDraft().edits.has(index))ctx.fillStyle="#a68b52";ctx.fillRect(row.start/d*w,96,Math.max(.5,(row.end-row.start)/d*w-.4),20);});const selected=state.source.rows[state.selected];ctx.strokeStyle="#ebc894";ctx.lineWidth=1.5;ctx.strokeRect(selected.start/d*w,94,Math.max(1,(selected.end-selected.start)/d*w),24);ctx.fillStyle="#f1f5e6";ctx.fillRect(Math.min(w-1,audio.currentTime/d*w),3,1.5,118);}
function tick(){const time=audio.currentTime||0;$("clock").textContent=fmt(time)+" / "+fmt(duration());$("seek").value=time;if(!state.result)return;const index=at(state.source.rows,time),editorPlayback=window.BarEditor?.playbackAt(time);if(editorPlayback){if(editorPlayback.label)setLabel($("nowChord"),$("nowSecondary"),editorPlayback.label);else {$("nowChord").textContent="未覆盖";$("nowSecondary").textContent="";}$("nowSecondary").textContent+= " · 编辑轨 / "+editorPlayback.origin;}else if(index<0){$("nowChord").textContent="未覆盖";$("nowSecondary").textContent="";}else setLabel($("nowChord"),$("nowSecondary"),chord(state.source,index));if(index!==state.active){if(state.active>=0)state.rows[state.active]?.classList.remove("playing");if(index>=0){const node=state.rows[index];node.classList.add("playing");const container=$("chordRows");if($("follow").checked&&(node.offsetTop<container.scrollTop||node.offsetTop+node.offsetHeight>container.scrollTop+container.clientHeight))container.scrollTop=Math.max(0,node.offsetTop-container.clientHeight/3);}state.active=index;}for(const source of state.result.sources){const i=at(source.rows,time),card=state.cards.get(source.id);if(i<0){card.main.textContent="未覆盖";card.sub.textContent="";}else setLabel(card.main,card.sub,chord(source,i));}for(const button of $("sections").children)button.classList.toggle("active",time>=Number(button.dataset.start)&&time<Number(button.dataset.end));drawWaveform();}
function playback(){audio.play().catch(error=>$("playStatus").textContent="播放失败："+error.message);}
function frame(){if(audio.paused){state.frame=null;return;}if(state.result){const row=state.source.rows[state.selected];if(state.loop&&audio.currentTime>=row.end-.006)audio.currentTime=row.start;else if(state.auditionEnd!==null&&audio.currentTime>=state.auditionEnd){audio.pause();state.auditionEnd=null;}}tick();state.frame=requestAnimationFrame(frame);}

$("tabFile").addEventListener("click",()=>showTab("file"));$("tabLink").addEventListener("click",()=>showTab("link"));$("bilibiliUrl").addEventListener("input",()=>setBusy(state.busy));$("audioFile").addEventListener("change",event=>selectFile(event.target.files[0]));
for(const name of ["dragenter","dragover"]){$("dropZone").addEventListener(name,event=>{event.preventDefault();$("dropZone").classList.add("dragover");});}for(const name of ["dragleave","drop"]){$("dropZone").addEventListener(name,event=>{event.preventDefault();$("dropZone").classList.remove("dragover");});}$("dropZone").addEventListener("drop",event=>selectFile(event.dataTransfer.files[0]));
$("analyze").addEventListener("click",beginAnalysis);$("retry").addEventListener("click",()=>{showTab("file");beginAnalysis();});$("refreshProjects").addEventListener("click",()=>refreshProjects().catch(error=>toast(error.message)));
$("newProject").addEventListener("click",()=>{state.navigation++;stopPolling();audio.pause();audio.removeAttribute("src");audio.load();if(state.objectUrl)URL.revokeObjectURL(state.objectUrl);state.objectUrl=null;state.project=null;state.file=null;clearResult();$("audioFile").value="";$("fileTitle").textContent="将音乐拖到这里，或点击选择";$("fileMeta").textContent="MP3 / WAV / FLAC / M4A 等 · 最大 512 MB";$("inputStatus").textContent="选择音频后可以立即试听。";$("playerPanel").hidden=true;$("jobPanel").hidden=true;$("emptyState").hidden=false;showTab("file");refreshProjects().catch(error=>toast(error.message));});
$("play").addEventListener("click",()=>audio.paused?playback():audio.pause());audio.addEventListener("play",()=>{$("play").textContent="Ⅱ";$("play").setAttribute("aria-label","暂停");if(!state.frame)state.frame=requestAnimationFrame(frame);});audio.addEventListener("pause",()=>{$("play").textContent="▶";$("play").setAttribute("aria-label","播放");tick();});audio.addEventListener("timeupdate",tick);audio.addEventListener("loadedmetadata",()=>{$("seek").max=duration();$("playStatus").textContent="音频已连接 · 点击时间轴或和弦区间定位";tick();});audio.addEventListener("error",()=>{if(audio.getAttribute("src"))$("playStatus").textContent="音频暂时无法播放，请检查格式或重新打开项目。";});audio.addEventListener("ended",()=>{if(state.loop&&state.result){seek(state.source.rows[state.selected].start);playback();}});
$("seek").addEventListener("input",()=>{state.auditionEnd=null;seek(Number($("seek").value));});$("speed").addEventListener("change",()=>audio.playbackRate=Number($("speed").value));$("loop").addEventListener("click",()=>{if(window.BarEditor?.getContext()){window.BarEditor.toggleLoop();return;}state.loop=!state.loop;state.auditionEnd=null;$("loop").setAttribute("aria-pressed",String(state.loop));if(state.loop){seek(state.source.rows[state.selected].start);playback();}});$("audition").addEventListener("click",()=>{window.BarEditor?.stopLoop();state.loop=false;$("loop").setAttribute("aria-pressed","false");const row=state.source.rows[state.selected];seek(row.start);state.auditionEnd=row.end;playback();});
$("keySelect").append(new Option("未确定",""));for(const root of notation.roots)for(const suffix of ["","m"])$("keySelect").append(new Option(keyName(root+suffix),root+suffix));$("keySelect").addEventListener("change",()=>updateKeyReview("custom",$("keySelect").value));$("resetKey").addEventListener("click",()=>updateKeyReview("restore"));$("adoptChordKey").addEventListener("click",()=>{if(state.chordKeySuggestion)updateKeyReview("custom",state.chordKeySuggestion);});$("displayRoman").addEventListener("click",()=>{state.mode="roman";changeNotation();});$("displayName").addEventListener("click",()=>{state.mode="name";changeNotation();});
$("applyEdit").addEventListener("click",applyEdit);$("chordInput").addEventListener("keydown",event=>{if(event.key==="Enter")applyEdit();});$("restoreChord").addEventListener("click",()=>{if(!sourceDraftGuard())return;currentDraft().edits.delete(state.selected);remember();renderRows();selectRow(state.selected);tick();});$("saveCorrections").addEventListener("click",saveCorrections);$("exportDraft").addEventListener("click",()=>{const blob=new Blob([JSON.stringify(correctionPayload(),null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),link=element("a");link.href=url;link.download="unsaved_corrections.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$("keyReviewEnabled").addEventListener("change",()=>updateKeyReview($("keyReviewEnabled").checked?"enable":"disable"));$("adoptKeySuggestion").addEventListener("click",()=>updateKeyReview("adopt"));
for(const id of ["exportLab","exportCorrections"])$(id).addEventListener("click",()=>{if(currentDraft().dirty)toast("此链接导出上次保存的版本。当前草稿请先保存，或点击导出当前草稿。");});
$("waveform").addEventListener("click",event=>{const box=$("waveform").getBoundingClientRect();state.auditionEnd=null;seek((event.clientX-box.left)/box.width*duration());const i=at(state.source.rows,audio.currentTime);if(i>=0)selectRow(i);});window.addEventListener("resize",drawWaveform);document.addEventListener("keydown",event=>{if(event.target.closest("input,textarea,select,button,a"))return;if(event.code==="Space"&&audio.src){event.preventDefault();$("play").click();}else if(event.key==="ArrowLeft"||event.key==="ArrowRight"){event.preventDefault();state.auditionEnd=null;seek(audio.currentTime+(event.key==="ArrowLeft"?-5:5));}});
window.addEventListener("harmony-editor-updated",tick);
refreshProjects().catch(error=>{$("inputStatus").textContent="无法连接本地服务："+error.message;});

$("sourceDraftRestore").addEventListener("click",restoreSourceDraft);
$("sourceDraftReload").addEventListener("click",reloadSourceDraft);
$("sourceDraftExport").addEventListener("click",()=>{const cached=currentDraft().cached;if(cached)downloadDraftText(cached.raw,"preserved_source_draft.json");});
window.addEventListener("storage",event=>{if(state.source&&event.key===draftKey(state.source)&&event.newValue!==currentDraft().browserRaw){currentDraft().cached=readSourceDraft(event.newValue)||{raw:"",problem:"浏览器草稿已在另一窗口改变，请重新读取。"};renderSourceDraftControls();}});

// Remove the one-time session secret from navigation history after the HttpOnly cookie is set.
if(new URL(location.href).searchParams.has("session"))history.replaceState(null,"",location.pathname);
