/* Presentation projection only; saved history remains unchanged. */
(function(root){
  "use strict";
  function fragmentEntries(history,duration){
    const ranges=new Map();let skipped=0,locks=0;
    for(const event of history||[]){
      if(event?.action!=="lock")continue;locks++;
      const label=event.label,start=label?.excerpt_start,end=label?.excerpt_end;
      if(!Number.isSafeInteger(event.revision)||event.revision<1||!label||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>duration+1e-6||typeof label.recording_id!=="string"||!label.recording_id||typeof label.basis!=="string"||typeof label.reviewer!=="string"||!["full_recording","excerpt"].includes(label.label_scope)||!["single_key","multi_key","ambiguous","unknown"].includes(label.annotation_status)){skipped++;continue;}
      const key=JSON.stringify([label.recording_id,start,end]);const old=ranges.get(key);
      if(!old||old.revision<event.revision)ranges.set(key,event);
    }
    return {entries:[...ranges.values()].sort((a,b)=>b.revision-a.revision),locks,skipped};
  }
  function summary(label){
    if(label.annotation_status==="multi_key")return "多调";
    if(label.annotation_status==="ambiguous")return "多个候选 / 歧义";
    if(label.annotation_status!=="single_key")return "尚不确定";
    return (label.tonic||"主音未知")+" · "+({major:"大调",minor:"小调"}[label.mode]||"调式未定");
  }
  const api={fragmentEntries,summary};if(typeof module!=="undefined"&&module.exports)module.exports=api;else root.KeyListeningHistory=api;
})(typeof window!=="undefined"?window:globalThis);
