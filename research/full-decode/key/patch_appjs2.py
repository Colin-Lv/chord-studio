"""Rewrite renderKeyExperimental and the key-origin label for the continuous key mode."""
from pathlib import Path

PATH = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\app.js")

NEW_FUNCTION = r'''function renderKeyExperimental(){
  const panel=$("keyExperimental"),block=state.result?.key_analysis?.experimental;
  if(!panel)return;panel.replaceChildren();panel.hidden=!block;if(!block)return;
  const active=state.result?.key_mode==="continuous"&&state.result?.key_analysis?.source==="continuous_key_v1";
  const same=block.key===state.key;
  panel.append(element("h3","",active?"连续评分 · 实验（当前主调来源）":"连续评分 · 实验（第二档）"));
  const parts=[`实验主调 ${keyName(block.key)}`);
  parts[0]+=` · ${same?"与当前主调一致":"与当前主调不同"}；分差 ${Number(block.gap)>=0?"+":""}${Number(block.gap).toFixed(3)}；`;
  parts[0]+=`声学 CNN 原判 ${keyName(block.acoustic_key)}`;
  if(block.published_key&&block.published_key!==block.acoustic_key)parts[0]+=`；覆盖裁决 ${keyName(block.published_key)}`;
  parts[0]+="。";
  panel.append(element("p","",parts[0]));
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
'''

ORIGIN_OLD = ('(state.result.key_fusion?(state.result.key_fusion.needs_review?"实验暂定 · 需审核":'
              '"自动融合 · 可调整"):"自动估计 · 可调整")')
ORIGIN_NEW = ('(state.result.key_analysis?.source==="continuous_key_v1"?"连续评分 · 实验（未校准）":'
              '(state.result.key_fusion?(state.result.key_fusion.needs_review?"实验暂定 · 需审核":'
              '"自动融合 · 可调整"):"自动估计 · 可调整"))')

text = PATH.read_text(encoding="utf-8")

start = text.index("function renderKeyExperimental(){")
end = text.index("\nfunction renderKey()", start)
text = text[:start] + NEW_FUNCTION.rstrip("\n") + text[end:]
print("renderKeyExperimental replaced")

count = text.count(ORIGIN_OLD)
print(f"origin text occurrences: {count}")
if count != 1:
    raise SystemExit("unexpected origin text count")
text = text.replace(ORIGIN_OLD, ORIGIN_NEW)

PATH.write_text(text, encoding="utf-8")
print("patched", PATH)
