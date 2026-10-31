"use strict";
// Keep the existing secondary tools within the desktop window, preserving drafts.
document.addEventListener("click", event => {
  const link=event.target.closest("a[target='_blank']");
  if(!link)return;
  const url=new URL(link.href,location.href);
  if(url.origin!==location.origin)return;
  event.preventDefault();
  const dialog=document.createElement("dialog"),bar=document.createElement("div"),close=document.createElement("button"),frame=document.createElement("iframe");
  dialog.style.cssText="width:92vw;height:88vh;padding:12px;background:#101719;color:#dce9e5;border:1px solid #55756a;border-radius:12px";
  close.textContent="关闭工具，返回和声工作台";close.className="secondary";close.onclick=()=>dialog.close();
  bar.append(close);frame.src=url.href;frame.title=link.textContent;frame.style.cssText="width:100%;height:calc(100% - 48px);border:0;background:#101719";
  dialog.append(bar,frame);document.body.append(dialog);dialog.addEventListener("close",()=>dialog.remove());dialog.showModal();
});
function desktopModelMode(){
  const model=document.getElementById("model"),key=document.getElementById("keyMode");
  if(model.value==="retained10"){key.value="madmom";key.disabled=true;}else key.disabled=false;
}
document.getElementById("model").addEventListener("change",desktopModelMode);
document.getElementById("newProject").addEventListener("click",()=>{document.getElementById("model").value="retained10";desktopModelMode();});
desktopModelMode();
