(function(root){
  "use strict";
  function layout(bars,rows,duration){
    if(!Number.isFinite(duration)||duration<=0)return [];
    const valid=(bars||[]).filter(b=>Number.isFinite(b.start)&&Number.isFinite(b.end)&&b.end>b.start);
    const edges=[...new Set([0,duration,...valid.flatMap(b=>[b.start,b.end]).filter(t=>t>0&&t<duration)])].sort((a,b)=>a-b);
    let rowIndex=0;
    return edges.slice(0,-1).map((start,i)=>{
      const end=edges[i+1],original=valid.find(b=>b.start===start);
      const number=original?.number??original?.bar??(valid.length?0:null),parts=[];
      let cursor=start;
      while(rowIndex<rows.length&&rows[rowIndex].end<=start)rowIndex++;
      const add=(a,b,index)=>{if(b>a)parts.push({start:a,end:b,sourceIndex:index,left:100*(a-start)/(end-start),width:100*(b-a)/(end-start)});};
      for(let j=rowIndex;j<rows.length&&rows[j].start<end;j++){
        const a=Math.max(start,rows[j].start),b=Math.min(end,rows[j].end);
        if(a>cursor)add(cursor,a,-1);
        add(a,b,j);cursor=b;
      }
      if(cursor<end)add(cursor,end,-1);
      return {start,end,number,parts};
    });
  }
  const api={layout};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root)root.ChordGrid=api;
})(typeof window!=="undefined"?window:null);
