"""Add the 音轨分离 entry button and its handler to the workbench."""
from __future__ import annotations

from pathlib import Path

FRONTEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend")

html = FRONTEND / "index.html"
text = html.read_text(encoding="utf-8")

OLD_ACTIONS = '<button id="analyze" class="primary" disabled>开始分析 <span>→</span></button>'
NEW_ACTIONS = ('<button id="separate" class="secondary" title="分离成人声 / 鼓 / 贝斯 / 其他四轨">'
               '音轨分离</button><button id="analyze" class="primary" disabled>开始分析 <span>→</span></button>')
assert text.count(OLD_ACTIONS) == 1
text = text.replace(OLD_ACTIONS, NEW_ACTIONS)

OLD_HEADING = ('<a id="analysisDownload" class="text-link" download="analysis.json">分析 JSON ↓</a>')
NEW_HEADING = ('<a id="stemsLink" class="text-link" hidden>分轨视图 →</a> '
               '<a id="analysisDownload" class="text-link" download="analysis.json">分析 JSON ↓</a>')
assert text.count(OLD_HEADING) == 1
text = text.replace(OLD_HEADING, NEW_HEADING)
html.write_text(text, encoding="utf-8")
print("index.html: separate button + stems link")

js = FRONTEND / "app.js"
text = js.read_text(encoding="utf-8")

OLD_LISTEN = '$("analyze").addEventListener("click",beginAnalysis);'
NEW_LISTEN = ('$("analyze").addEventListener("click",beginAnalysis);'
              '$("separate").addEventListener("click",()=>beginSeparation().catch(error=>{'
              'toast(error.message);$("inputStatus").textContent=error.message;}));')
assert text.count(OLD_LISTEN) == 1
text = text.replace(OLD_LISTEN, NEW_LISTEN)

OLD_FUNC = 'async function watchJob(id,navigation){'
NEW_FUNC = '''async function beginSeparation(){
  if(state.busy)return;setBusy(true);
  try{
    if(!state.project){
      if(!state.file)throw new Error("请先选择音乐，或打开一个已有项目");
      $("inputStatus").textContent="正在上传音频…";
      const form=new FormData();form.append("file",state.file);
      state.project=await api("/api/upload",{method:"POST",body:form});state.file=null;
    }
    const id=state.project.id;
    const job=await post(`/api/projects/${id}/separate`,{force:false});
    $("inputStatus").textContent="分轨任务已提交（人声 / 鼓 / 贝斯 / 其他），完成后会自动打开分轨页…";
    toast("分轨通常需要 1–2 分钟，可以继续浏览其他页面");
    for(let i=0;i<300;i++){
      await new Promise(resolve=>setTimeout(resolve,2000));
      let row;try{row=await api(`/api/jobs/${job.id}`);}catch(_){continue;}
      if(row.stage_label)$("inputStatus").textContent="分轨：" + row.stage_label;
      if(row.status==="completed"){location.href=`/stems?project=${id}`;return;}
      if(row.status==="failed"||row.status==="interrupted")throw new Error(row.error||"分轨失败，请重试");
    }
    throw new Error("分轨耗时过长，请稍后在分轨页查看");
  }finally{setBusy(false);}
}
async function watchJob(id,navigation){'''
assert text.count(OLD_FUNC) == 1
text = text.replace(OLD_FUNC, NEW_FUNC)

OLD_LOAD = '$("model").value="retained10";$("resultModel").textContent=modelDisplayText(result);'
NEW_LOAD = ('$("model").value="retained10";$("resultModel").textContent=modelDisplayText(result);'
            'const stemsLink=$("stemsLink");if(stemsLink){stemsLink.hidden=!state.project?.stems;'
            'stemsLink.href=`/stems?project=${result.project_id}`;}')
assert text.count(OLD_LOAD) == 1
text = text.replace(OLD_LOAD, NEW_LOAD)
js.write_text(text, encoding="utf-8")
print("app.js: separation flow added")
