"""Offer both checkpoints again now that the GPU makes htdemucs_ft cheap, with an explicit ETA."""
from __future__ import annotations

from pathlib import Path

JS = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\stems.js")
text = JS.read_text(encoding="utf-8")

OLD = '''function renderModelSelect(manifest) {
  const select = $("modelSelect");
  // Only list checkpoints that already exist for this song: choosing a missing one used to
  // start a multi-minute separation from a dropdown, which is far too easy to trigger.
  const options = (manifest.model_options || []).filter(option => option.ready);
  select.replaceChildren();
  for (const option of options) {
    const node = document.createElement("option");
    node.value = option.id;
    node.textContent = option.label;
    select.appendChild(node);
  }
  select.value = manifest.model_key || options[0]?.id || "";
  select.hidden = options.length < 2;
  select.onchange = () => switchModel(select.value);
}
'''
NEW = '''function renderModelSelect(manifest) {
  const select = $("modelSelect");
  const options = manifest.model_options || [];
  select.replaceChildren();
  for (const option of options) {
    const node = document.createElement("option");
    node.value = option.id;
    node.textContent = `${option.label}${option.ready ? "" : "（未生成）"}`;
    node.title = option.note || "";
    select.appendChild(node);
  }
  select.value = manifest.model_key || options.find(o => o.ready)?.id || "";
  select.hidden = options.length < 2;
  select.onchange = () => switchModel(select.value);
}
'''
assert text.count(OLD) == 1
text = text.replace(OLD, NEW)

OLD_SWITCH = '''async function switchModel(model) {
  if (!model || model === state.manifest?.model_key) return;
  const option = (state.manifest.model_options || []).find(item => item.id === model);
  if (!option?.ready) {
    $("modelSelect").value = state.manifest.model_key;
    return;
  }
  try {
'''
NEW_SWITCH = '''async function switchModel(model) {
  if (!model || model === state.manifest?.model_key) return;
  const option = (state.manifest.model_options || []).find(item => item.id === model);
  if (!option?.ready) {
    const note = option?.note ? `\\n\\n${option.note}` : "";
    if (!window.confirm(`用「${option?.label || model}」再分离一遍？${note}`)) {
      $("modelSelect").value = state.manifest.model_key;
      return;
    }
    await runSeparation(false, model);
    return;
  }
  try {
'''
assert text.count(OLD_SWITCH) == 1
text = text.replace(OLD_SWITCH, NEW_SWITCH)
JS.write_text(text, encoding="utf-8")
print("model select offers both checkpoints with ETA")
