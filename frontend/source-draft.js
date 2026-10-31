(function(root,factory){const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;else root.HarmonySourceDraft=api;})(typeof globalThis!=="undefined"?globalThis:this,function(){
  "use strict";
  const schema="harmony_source_draft_v1";
  function identity(result,source){return {schema,project_id:result.project_id,run_id:result.run_id,source_id:source.id,source_sha256:source.sha256,audio_sha256:result.audio?.sha256,duration:result.duration};}
  function read(raw,expected,rowCount,parseLabel){
    if(raw===null)return null;
    const response={raw,problem:"",value:null};
    try{
      const value=JSON.parse(raw);response.value=value;
      if(!value||Array.isArray(value)||typeof value!=="object")throw Error("草稿格式无法读取");
      for(const key of ["schema","project_id","run_id","source_id","source_sha256","audio_sha256","duration"]){
        if(expected[key]===undefined||value[key]!==expected[key])throw Error("草稿的来源、录音或时间身份无法验证（旧格式也可能缺少这些字段）");
      }
      if(!Number.isInteger(value.revision)||value.revision<0)throw Error("草稿修订号无效");
      if(!Array.isArray(value.edits)||value.edits.length>rowCount)throw Error("草稿区间列表无效");
      const seen=new Set();
      for(const edit of value.edits){
        if(!Array.isArray(edit)||edit.length!==2||!Number.isInteger(edit[0])||edit[0]<0||edit[0]>=rowCount||seen.has(edit[0]))throw Error("草稿含重复或不存在的区间");
        if(typeof edit[1]!=="string"||!edit[1]||edit[1].length>160||/\s/.test(edit[1])||(parseLabel&&!parseLabel(edit[1])))throw Error("草稿中有无法解析的和弦");
        seen.add(edit[0]);
      }
      if(!["roman","name"].includes(value.display))throw Error("草稿记法选项无效");
    }catch(error){response.problem=error instanceof SyntaxError?"草稿文本无法解析，原始内容已保留，可导出核对。":error.message||"草稿格式无法读取";}
    return response;
  }
  function encode(binding,draft,display){return JSON.stringify({...binding,revision:draft.revision,edits:[...draft.edits],display,updated_at:new Date().toISOString()});}
  return {identity,read,encode};
});
