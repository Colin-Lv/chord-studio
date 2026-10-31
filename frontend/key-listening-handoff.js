/* Exact listening ranges with recording identity; never contains a key answer. */
(function(root){
  "use strict";
  function parse(params,current){
    if(!["from","run","audio","start","end"].some(k=>params.has(k)))return null;
    if(params.get("from")!=="workbench"||params.get("project")!==current.project_id||params.get("run")!==current.run_id||params.get("audio")!==current.audio_sha256)throw Error("片段链接与当前项目、分析或音频不一致；请从工作区重新选择范围。");
    const a=params.get("start"),b=params.get("end"),start=Number(a),end=Number(b);
    if(a===null||b===null||!a.trim()||!b.trim()||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>current.duration)throw Error("片段链接的时间范围无效；请从工作区重新选择范围。");
    return {origin:"workbench",project_id:current.project_id,run_id:current.run_id,audio_sha256:current.audio_sha256,excerpt_start:start,excerpt_end:end};
  }
  function url(doc,start,end){
    const params=new URLSearchParams({project:doc.project_id,from:"workbench",run:doc.run_id,audio:doc.audio_sha256,start:String(start),end:String(end)});
    parse(params,doc);return "/assets/key_annotation.html?"+params.toString();
  }
  const api={parse,url};if(typeof module!=="undefined"&&module.exports)module.exports=api;else root.KeyListeningHandoff=api;
})(typeof window!=="undefined"?window:globalThis);
