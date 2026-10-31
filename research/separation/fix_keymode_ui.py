"""Let the workbench actually choose the key mode for the frozen model.

desktop.js used to force keyMode=madmom (and disable the select) whenever the selected model
was retained10, so the continuous scorer could never be picked from the UI even though the
backend supports it and the HTML defaults to it.
"""
from __future__ import annotations

from pathlib import Path

FRONTEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend")

DESKTOP = FRONTEND / "desktop.js"
text = DESKTOP.read_text(encoding="utf-8")
OLD = """function desktopModelMode(){
  const model=document.getElementById("model"),key=document.getElementById("keyMode");
  if(model.value==="retained10"){key.value="madmom";key.disabled=true;}else key.disabled=false;
}
"""
NEW = """function desktopModelMode(){
  const model=document.getElementById("model"),key=document.getElementById("keyMode");
  const retained=model.value==="retained10";
  key.disabled=false;
  for(const option of key.options){
    const allowed=option.value==="continuous"?retained:option.value==="fused"?!retained:true;
    option.hidden=!allowed;option.disabled=!allowed;
  }
  const current=key.options[key.selectedIndex];
  if(!current||current.disabled)key.value=retained?"continuous":"madmom";
}
"""
assert text.count(OLD) == 1, text.count(OLD)
text = text.replace(OLD, NEW)
DESKTOP.write_text(text, encoding="utf-8")
print("desktop.js: key mode select enabled for retained10")

APP = FRONTEND / "app.js"
text = APP.read_text(encoding="utf-8")
OLD_LOAD = '$("keyMode").value=result.key_mode||"madmom";'
NEW_LOAD = 'if(result.key_mode)$("keyMode").value=result.key_mode;'
assert text.count(OLD_LOAD) == 1, text.count(OLD_LOAD)
APP.write_text(text.replace(OLD_LOAD, NEW_LOAD), encoding="utf-8")
print("app.js: keep the user's key mode for old runs")
