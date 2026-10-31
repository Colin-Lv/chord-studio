"""Fix the summary construction in renderKeyExperimental (array -> string)."""
from pathlib import Path

PATH = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\app.js")

OLD = '''  const parts=[`实验主调 ${keyName(block.key)}`);
  parts[0]+=` · ${same?"与当前主调一致":"与当前主调不同"}；分差 ${Number(block.gap)>=0?"+":""}${Number(block.gap).toFixed(3)}；`;
  parts[0]+=`声学 CNN 原判 ${keyName(block.acoustic_key)}`;
  if(block.published_key&&block.published_key!==block.acoustic_key)parts[0]+=`；覆盖裁决 ${keyName(block.published_key)}`;
  parts[0]+="。";
  panel.append(element("p","",parts[0]));
'''

NEW = '''  let summary=`实验主调 ${keyName(block.key)}`;
  summary+=` · ${same?"与当前主调一致":"与当前主调不同"}；分差 ${Number(block.gap)>=0?"+":""}${Number(block.gap).toFixed(3)}；`;
  summary+=`声学 CNN 原判 ${keyName(block.acoustic_key)}`;
  if(block.published_key&&block.published_key!==block.acoustic_key)summary+=`；覆盖裁决 ${keyName(block.published_key)}`;
  summary+="。";
  panel.append(element("p","",summary));
'''

text = PATH.read_text(encoding="utf-8")
count = text.count(OLD)
print(f"occurrences: {count}")
if count != 1:
    raise SystemExit("expected exactly one")
PATH.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("fixed")
