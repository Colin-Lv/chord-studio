"""Drum-parts toggle in the stems page: expand the drums row into six parts."""
from __future__ import annotations

from pathlib import Path

JS = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\stems.js")
text = JS.read_text(encoding="utf-8")

# state fields
OLD_STATE = """  referenceOn: false, loopBar: false, loopIndex: -1,"""
NEW_STATE = """  referenceOn: false, loopBar: false, loopIndex: -1,
  drumParts: null, expandDrums: false,"""
assert text.count(OLD_STATE) == 1
text = text.replace(OLD_STATE, NEW_STATE)

# fetch parts in start()
OLD_START = """  state.chords = chordTimeline(result);
"""
NEW_START = """  state.chords = chordTimeline(result);
  state.drumParts = await api(`/api/projects/${PROJECT}/stems/drums-parts?model=${encodeURIComponent(manifest.model_key || "htdemucs")}`).catch(() => null);
  state.tracks = composedTracks();
  renderDrumButton();
"""
assert text.count(OLD_START) == 1
text = text.replace(OLD_START, NEW_START)

# composed tracks + button + toggle
OLD_HEADS = """function buildHeads() {"""
NEW_HEADS = """function composedTracks() {
  const base = state.manifest?.tracks || [];
  if (!state.expandDrums || !state.drumParts?.tracks?.length) return base;
  const out = [];
  for (const track of base) {
    if (track.id === "drums") {
      for (const part of state.drumParts.tracks) out.push({ ...part, group: "drums" });
    } else {
      out.push(track);
    }
  }
  return out;
}

function renderDrumButton() {
  const button = $("drumPartsBtn");
  if (!button) return;
  const hasDrums = (state.manifest?.tracks || []).some(track => track.id === "drums");
  button.hidden = !hasDrums;
  if (!hasDrums) return;
  button.textContent = state.drumParts
    ? (state.expandDrums ? "收起鼓组" : "展开鼓组")
    : "细分鼓组";
  button.onclick = () => (state.drumParts ? toggleDrums() : runDrumParts());
}

function rebuildTracks() {
  const time = currentTime();
  stop();
  state.tracks = composedTracks();
  state.audios = {};
  $("trackHeads").replaceChildren();
  buildAudio();
  buildHeads();
  layout();
  buildRuler();
  drawGrid();
  updateInfo();
  seek(time);
  renderDrumButton();
}

function toggleDrums() {
  state.expandDrums = !state.expandDrums;
  rebuildTracks();
}

async function runDrumParts() {
  const button = $("drumPartsBtn");
  button.disabled = true;
  button.textContent = "正在细分…";
  try {
    const job = await api(`/api/projects/${PROJECT}/drums-parts`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: false, model: state.manifest.model_key || "htdemucs" }),
    });
    if (job.profile !== "drums_parts") throw new Error("当前有任务在跑，等它结束再细分鼓组");
    for (;;) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const row = await api(`/api/jobs/${job.id}`);
      button.textContent = row.stage_label || "正在细分…";
      if (row.status === "completed") break;
      if (row.status === "failed" || row.status === "interrupted") {
        throw new Error(row.error || "鼓组细分失败");
      }
    }
    state.drumParts = await api(`/api/projects/${PROJECT}/stems/drums-parts?model=${encodeURIComponent(state.manifest.model_key || "htdemucs")}`);
    state.expandDrums = true;
    rebuildTracks();
  } catch (error) {
    toast(error.message);
    renderDrumButton();
  } finally {
    button.disabled = false;
  }
}

function buildHeads() {"""
assert text.count(OLD_HEADS) == 1
text = text.replace(OLD_HEADS, NEW_HEADS)

# audio URL lookup must cover drum parts
OLD_AUDIO = """    audio.src = state.manifest.urls?.[track.id] || `/api/projects/${PROJECT}/stems/${track.file}`;"""
NEW_AUDIO = """    audio.src = state.manifest.urls?.[track.id] || state.drumParts?.urls?.[track.id]
      || `/api/projects/${PROJECT}/stems/${track.file}`;"""
assert text.count(OLD_AUDIO) == 1
text = text.replace(OLD_AUDIO, NEW_AUDIO)

JS.write_text(text, encoding="utf-8")
print("stems.js: drum parts toggle")
