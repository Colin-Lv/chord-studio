(function(root){
  "use strict";
  function time(value,digits=2){
    const scale=10**digits, ticks=Math.round(Math.max(0,Number(value)||0)*scale);
    const minutes=Math.floor(ticks/(60*scale)), seconds=(ticks-minutes*60*scale)/scale;
    return String(minutes).padStart(2,"0")+":"+seconds.toFixed(digits).padStart(digits?3+digits:2,"0");
  }
  function range(start,end){
    if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start)return "无效时间区间";
    for(let digits=2;digits<=9;digits++){
      const a=time(start,digits),b=time(end,digits);
      if(a!==b)return a+"–"+b;
    }
    return time(start,9)+"（持续 "+(end-start).toPrecision(3)+" 秒）";
  }
  const api={time,range};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root)root.HarmonyTime=api;
})(typeof window!=="undefined"?window:null);
