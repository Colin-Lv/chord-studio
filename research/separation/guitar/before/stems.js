"use strict";
/* Stems view: FL-style step grid + synchronised 4-track playback. */
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const PROJECT = params.get("project");
const MODEL = params.get("model");
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const state = {
  projectId: PROJECT, project: null, manifest: null,
  audios: {}, mix: null, tracks: [], bars: [], stepsPerCell: 16,
  playing: false, solo: null, muted: {}, master: 1,
  referenceOn: false, loopBar: false, loopIndex: -1,
  drumParts: null, expandDrums: false, vocalParts: null, expandVocals: false,
  vocalCatalog: {}, vocalReads: {}, vocalJob: null, vocalPolling: false, vocalNotice: "",
  viewRequest: 0, keysAttached: false,
  cellW: 20, rowH: 74, hoverBar: -1, raf: null,
};

const fmt = seconds => {
  const t = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const preferenceKey = `stems:${PROJECT}`;
function readPreference() {
  try { return JSON.parse(localStorage.getItem(preferenceKey)) || {}; } catch (_) { return {}; }
}
function savePreference(patch) {
  try { localStorage.setItem(preferenceKey, JSON.stringify({ ...readPreference(), ...patch })); } catch (_) { /* private storage */ }
}
function saveExpansion(model, expanded) {
  savePreference({ expanded: { ...readPreference().expanded, [model]: expanded } });
}
function modelLabel(model) {
  return state.manifest?.model_options?.find(item => item.id === model)?.label || model;
}
async function optionalResult(url) {
  try { return await api(url); } catch (error) { if (error.status === 404) return null; throw error; }
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 6000);
}
function showError(message) {
  const node = $("error");
  node.hidden = !message;
  node.textContent = message || "";
}

async function api(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { "X-Harmony-Client": "studio", ...(options.headers || {}) } });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); message = body.detail || body.error || message; } catch (_) { /* ignore */ }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  if (!PROJECT) { showError("缺少 project 参数：请从工作台点「音轨分离」进入。"); return; }
  try {
    state.project = await api(`/api/projects/${PROJECT}`);
    $("songTitle").textContent = state.project.title || "分轨";
    $("downloadMix").href = `/api/projects/${PROJECT}/audio`;
    $("downloadMix").hidden = false;
    $("backToWorkbench").href = "/";
  } catch (error) { showError(error.message); return; }
  try {
    const selected = MODEL || readPreference().model;
    const query = selected ? `?model=${encodeURIComponent(selected)}` : "";
    const manifest = await api(`/api/projects/${PROJECT}/stems${query}`);
    await start(manifest);
  } catch (error) {
    if (error.status === 404) showPending();
    else showError(error.message);
  }
  await resumeVocalJob();
}

function showPending() {
  $("pending").hidden = false;
  $("workspace").hidden = true;
  $("modelSelect").hidden = true;
  $("songMeta").textContent = "尚未生成分轨";
  $("startSeparate").onclick = () => runSeparation(false);
}

async function runSeparation(force, model) {
  $("pending").hidden = false;
  $("startSeparate").disabled = true;
  $("startSeparate").textContent = "正在分离…";
  $("pendingTitle").textContent = force ? "重新分离" : "正在分离";
  $("pendingNote").textContent = "人声 / 鼓 / 贝斯 / 其他四轨；标准模型约 2–3 分钟，精细模型约 2 倍时间。";
  try {
    const job = await api(`/api/projects/${PROJECT}/separate`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: !!force, model: model || "htdemucs" }),
    });
    if (job.profile && job.profile !== "stems") {
      throw new Error("当前有分析任务在跑，等它结束再点「开始分离」");
    }
    await followJob(job.id, model);
  } catch (error) {
    $("separateStatus").textContent = error.message;
    $("startSeparate").disabled = false;
    $("startSeparate").textContent = "重试分离 →";
    if (state.manifest) $("modelSelect").value = state.manifest.model_key || "htdemucs";
  }
}

async function followJob(jobId, model) {
  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    let job;
    try { job = await api(`/api/jobs/${jobId}`); } catch (error) { $("separateStatus").textContent = error.message; continue; }
    $("separateStatus").textContent = job.stage_label || "处理中…";
    if (job.status === "completed") {
      $("separateStatus").textContent = "分轨完成，正在载入…";
      const query = model ? `?model=${encodeURIComponent(model)}` : "";
      await start(await api(`/api/projects/${PROJECT}/stems${query}`));
      return;
    }
    if (job.status === "failed" || job.status === "interrupted") {
      $("separateStatus").textContent = job.error || "分轨失败，请重试。";
      $("startSeparate").disabled = false;
      $("startSeparate").textContent = "重试分离 →";
      return;
    }
  }
}

/* ------------------------------------------------------------------ main */
function resetPlayback() {
  stop();
  state.audios = {};
  state.mix = null;
  state.solo = null;
  state.muted = {};
  state.referenceOn = false;
  state.loopBar = false;
  state.loopIndex = -1;
  if ($("reference")) $("reference").checked = false;
  if ($("loopBar")) $("loopBar").checked = false;
}

function renderModelSelect(manifest) {
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

async function switchModel(model) {
  if (!model) return;
  const request = ++state.viewRequest;
  if (model === state.manifest?.model_key) return;
  const option = (state.manifest.model_options || []).find(item => item.id === model);
  if (!option?.ready) {
    const note = option?.note ? `\n\n${option.note}` : "";
    if (!window.confirm(`用「${option?.label || model}」再分离一遍？${note}`)) {
      $("modelSelect").value = state.manifest.model_key;
      return;
    }
    await runSeparation(false, model);
    return;
  }
  try {
    const manifest = await api(`/api/projects/${PROJECT}/stems?model=${encodeURIComponent(model)}`);
    if (request !== state.viewRequest) return;
    await start(manifest, request);
  } catch (error) {
    if (request !== state.viewRequest) return;
    $("modelSelect").value = state.manifest.model_key;
    toast(error.message);
  }
}

async function start(manifest, request = ++state.viewRequest) {
  const model = manifest.model_key || "htdemucs";
  const [result, drums] = await Promise.all([
    optionalResult(`/api/projects/${PROJECT}/result`),
    optionalResult(`/api/projects/${PROJECT}/stems/drums-parts?model=${encodeURIComponent(model)}`),
    refreshVocalCatalog(manifest),
  ]);
  if (request !== state.viewRequest) return;
  resetPlayback();
  state.manifest = manifest;
  state.drumParts = drums;
  state.vocalParts = state.vocalCatalog[model]?.result || null;
  state.expandVocals = readPreference().expanded?.[model] ?? !!state.vocalParts;
  savePreference({ model });
  const url = new URL(location.href);
  url.searchParams.set("model", model);
  history.replaceState(null, "", url);
  state.tracks = manifest.tracks || [];
  state.bars = manifest.bars || [];
  $("pending").hidden = true;
  $("workspace").hidden = false;
  showError("");
  const sourceText = manifest.bpm_source === "beats.json" ? "小节来自拍点分析"
    : manifest.bpm_source === "librosa estimate" ? "BPM 与小节为估计值" : "BPM 与小节为默认值";
  $("songMeta").textContent = `${fmt(manifest.duration)} · ${state.tracks.length} 轨 · ${sourceText}`;
  state.metaSuffix = sourceText;
  $("bpm").textContent = Number(manifest.bpm).toFixed(1);
  $("gridSource").textContent = manifest.bpm_source === "beats.json" ? "小节来自拍点分析" : "小节为估计值";
  $("barLabel").textContent = state.bars.length;
  const residual = manifest.reconstruction?.relative_db;
  $("modelInfo").textContent = `${manifest.model?.label || manifest.model?.name || "htdemucs"} · ${manifest.model?.seconds ?? "—"} s`
    + (residual == null ? "" : ` · 重构残差 ${Number(residual).toFixed(1)} dB`);
  $("clockTotal").textContent = "/ " + fmt(manifest.duration);
  $("reSeparate").hidden = false;
  $("reSeparate").onclick = () => runSeparation(true, manifest.model_key || "htdemucs");
  renderModelSelect(manifest);

  const key = result?.key_analysis?.key;
  $("keyLabel").textContent = key ? key.replace(/m$/, "") + (key.endsWith("m") ? " 小调" : " 大调") : "—";
  state.chords = chordTimeline(result);
  state.tracks = composedTracks();
  $("songMeta").textContent = `${fmt(manifest.duration)} · ${state.tracks.length} 轨 · ${sourceText}`;
  renderDrumButton();
  renderVocalButton();

  buildAudio();
  buildHeads();
  layout();
  buildRuler();
  drawGrid();
  updateInfo();
  attachTransport();
  attachGridEvents();
  attachKeys();
  renderClock();
}

function master() { return state.audios[state.tracks[0]?.id]; }

function buildAudio() {
  document.querySelectorAll("audio.stem-audio").forEach(node => { node.pause(); node.remove(); });
  state.audios = {};
  for (const track of state.tracks) {
    const audio = document.createElement("audio");
    audio.className = "stem-audio";
    audio.preload = "auto";
    audio.src = state.manifest.urls?.[track.id] || state.drumParts?.urls?.[track.id] || state.vocalParts?.urls?.[track.id]
      || `/api/projects/${PROJECT}/stems/${track.file}`;
    audio.dataset.track = track.id;
    audio.addEventListener("ended", () => { if (audio === master()) stop(); });
    document.body.appendChild(audio);
    state.audios[track.id] = audio;
  }
  const mix = document.createElement("audio");
  mix.className = "stem-audio";
  mix.preload = "auto";
  mix.src = state.manifest.audio_url || `/api/projects/${PROJECT}/audio`;
  mix.dataset.track = "mix";
  document.body.appendChild(mix);
  state.mix = mix;
}

function composedTracks() {
  const base = state.manifest?.tracks || [];
  
  const out = [];
  for (const track of base) {
    if (track.id === "vocals" && state.expandVocals && state.vocalParts?.tracks?.length) {
      for (const part of state.vocalParts.tracks) out.push({ ...part, group: "vocals" });
    } else if (track.id === "drums" && state.expandDrums && state.drumParts?.tracks?.length) {
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
  state.solo = null;
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
  $("songMeta").textContent = `${fmt(state.manifest.duration)} · ${state.tracks.length} 轨 · ${state.metaSuffix || ""}`;
  renderDrumButton();
  renderVocalButton();
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

async function readVocalResult(model) {
  const serial = state.vocalReads[model] = (state.vocalReads[model] || 0) + 1;
  let entry;
  try {
    entry = { result: await optionalResult(`/api/projects/${PROJECT}/stems/vocal-parts?model=${encodeURIComponent(model)}`) };
  } catch (error) { entry = { result: null, error: error.message }; }
  // A slow pre-completion read must not replace a newer completed result.
  if (serial === state.vocalReads[model]) state.vocalCatalog[model] = entry;
  return state.vocalCatalog[model];
}

async function refreshVocalCatalog(manifest) {
  const models = new Set([manifest.model_key || "htdemucs",
    ...(manifest.model_options || []).filter(item => item.ready).map(item => item.id)]);
  await Promise.all([...models].map(readVocalResult));
}

async function showVocalResult(model) {
  saveExpansion(model, true);
  if (state.manifest?.model_key !== model) await switchModel(model);
  if (state.manifest?.model_key !== model || !state.vocalParts) return;
  state.expandVocals = true;
  rebuildTracks();
  document.querySelector('.track-head[data-track="lead_vocals"]')?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderVocalPanel() {
  if (!state.manifest) return;
  const model = state.manifest.model_key || "htdemucs";
  $("vocalSource").textContent = `当前来源：${modelLabel(model)} · 人声轨`;
  const entry = state.vocalCatalog[model];
  $("vocalStatus").textContent = entry?.error ? `结果读取失败：${entry.error}。请刷新重试，已有文件不会因此删除。`
    : state.vocalParts ? (state.expandVocals ? "已保存 · 主唱与和声已展开在下方音轨列表。" : "已保存 · 当前已收起，点击下方「查看结果」展开。")
    : "本版暂无可用的人声细分结果（尚未生成或源人声已更新）。其他版本的结果仍可从下方打开。";
  const job = state.vocalJob;
  $("vocalJobStatus").hidden = !job && !state.vocalNotice;
  $("vocalJobStatus").textContent = job
    ? `${modelLabel(job.model)} · ${job.stage_label || "正在细分人声…"}。切换版本或刷新页面不影响此任务。`
    : state.vocalNotice;
  const container = $("vocalSavedResults");
  container.replaceChildren();
  for (const [source, saved] of Object.entries(state.vocalCatalog)) {
    if (!saved.result) continue;
    const button = document.createElement("button");
    button.className = "secondary";
    button.dataset.model = source;
    button.textContent = `查看结果 · ${modelLabel(source)}`;
    button.title = "切换到对应版本并展开主唱与和声";
    button.onclick = () => showVocalResult(source);
    container.appendChild(button);
  }
  const errors = Object.entries(state.vocalCatalog).filter(([source, saved]) => source !== model && saved.error);
  if (errors.length) {
    const note = document.createElement("span");
    note.className = "muted";
    note.textContent = `${errors.map(([source]) => modelLabel(source)).join("、")}的结果暂时无法读取，请刷新重试。`;
    container.appendChild(note);
  }
}

function renderVocalButton() {
  const button = $("vocalPartsBtn");
  button.hidden = !(state.manifest?.tracks || []).some(t => t.id === "vocals");
  button.disabled = !state.vocalParts && !!state.vocalJob;
  button.textContent = state.vocalParts ? (state.expandVocals ? "收起人声细分" : "展开主唱／和声")
    : state.vocalJob ? "人声细分进行中" : "细分主唱／和声";
  button.title = "主唱为模型估计；和声轨还包含混响、串音与分离余量。";
  button.onclick = () => {
    if (state.vocalParts) {
      state.expandVocals = !state.expandVocals;
      saveExpansion(state.manifest.model_key || "htdemucs", state.expandVocals);
      rebuildTracks();
    }
    else runVocalParts();
  };
  renderVocalPanel();
}

async function runVocalParts() {
  if (state.vocalJob) return;
  const model = state.manifest.model_key || "htdemucs";
  state.vocalJob = { model, stage_label: "正在提交任务" };
  state.vocalNotice = "";
  renderVocalButton();
  try {
    const job = await api(`/api/projects/${PROJECT}/vocal-parts`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    if (job.profile !== "vocal_parts") throw new Error("当前有其他任务在运行，请待完成后再细分人声");
    state.vocalJob = job;
    savePreference({ vocalJobId: job.id });
    await watchVocalJob();
  } catch (error) {
    state.vocalJob = null;
    state.vocalNotice = error.message;
    toast(error.message);
  } finally { renderVocalButton(); }
}

async function resumeVocalJob() {
  const ids = [...new Set([state.project?.active_job, readPreference().vocalJobId].filter(Boolean))];
  for (const id of ids) {
    try {
      const job = await api(`/api/jobs/${id}`);
      if (job.profile !== "vocal_parts") continue;
      state.vocalJob = job;
      savePreference({ vocalJobId: job.id });
      renderVocalButton();
      void watchVocalJob();
      return;
    } catch (error) {
      if (error.status === 404) { savePreference({ vocalJobId: null }); continue; }
      state.vocalNotice = `无法恢复人声任务状态：${error.message}。请刷新重试。`;
      renderVocalPanel();
      return;
    }
  }
}

async function watchVocalJob() {
  if (state.vocalPolling) return;
  state.vocalPolling = true;
  try {
    while (state.vocalJob?.id) {
      const job = state.vocalJob;
      let row;
      try { row = await api(`/api/jobs/${job.id}`); }
      catch (error) {
        if (error.status === 404) throw error;
        state.vocalJob.stage_label = "暂时无法读取进度，正在重连";
        renderVocalButton();
        await new Promise(resolve => setTimeout(resolve, 2000));
        continue;
      }
      state.vocalJob = row;
      renderVocalButton();
      if (["failed", "interrupted"].includes(row.status)) throw new Error(row.error || "人声细分失败，请重试");
      if (row.status === "completed") {
        const model = row.model || "htdemucs";
        const entry = await readVocalResult(model);
        if (entry?.error) {
          state.vocalJob.stage_label = "细分已完成，正在重试读取结果";
          renderVocalButton();
          await new Promise(resolve => setTimeout(resolve, 2000));
          continue;
        }
        if (!entry?.result) throw new Error("任务已完成，但结果不可用；源人声可能已更新，请重新细分。");
        saveExpansion(model, true);
        state.vocalJob = null;
        savePreference({ vocalJobId: null });
        state.vocalNotice = `${modelLabel(model)}的人声细分已保存。点击「查看结果」即可试听和下载。`;
        if (state.manifest?.model_key === model) {
          state.vocalParts = entry.result;
          state.expandVocals = true;
          rebuildTracks();
        }
        toast(state.vocalNotice);
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  } catch (error) {
    state.vocalNotice = error.message;
    state.vocalJob = null;
    savePreference({ vocalJobId: null });
    toast(error.message);
  } finally {
    state.vocalPolling = false;
    renderVocalButton();
  }
}

function buildHeads() {
  const container = $("trackHeads");
  container.replaceChildren();
  const spacer = document.createElement("div");
  spacer.className = "ruler-spacer";
  container.appendChild(spacer);
  for (const track of state.tracks) {
    const row = document.createElement("div");
    row.className = "track-head";
    row.dataset.track = track.id;
    row.style.height = state.rowH + "px";

    const chip = document.createElement("span");
    chip.className = "track-chip";
    chip.style.background = track.color;

    const name = document.createElement("div");
    name.className = "track-name";
    const strong = document.createElement("strong");
    strong.textContent = track.name;
    strong.title = track.name;
    const small = document.createElement("small");
    small.textContent = `${track.short} · ${track.instrument} · ${track.rms_db} dB`;
    small.title = small.textContent;
    name.append(strong, small);

    const buttons = document.createElement("span");
    buttons.className = "track-buttons";
    const download = document.createElement("a");
    download.className = "track-download";
    download.textContent = "↓";
    download.title = `下载 ${track.name} WAV`;
    download.href = state.manifest.wav_urls?.[track.id] || state.drumParts?.wav_urls?.[track.id] || state.vocalParts?.wav_urls?.[track.id]
      || `/api/projects/${PROJECT}/stems/${track.wav}`;
    download.download = `${track.id}.wav`;
    const mute = document.createElement("button");
    mute.textContent = "M";
    mute.title = "静音";
    mute.setAttribute("aria-pressed", "false");
    mute.onclick = event => { event.stopPropagation(); toggleMute(track.id); };
    const solo = document.createElement("button");
    solo.className = "solo";
    solo.textContent = "S";
    solo.title = "独奏";
    solo.setAttribute("aria-pressed", "false");
    solo.onclick = event => { event.stopPropagation(); toggleSolo(track.id); };
    buttons.append(download, mute, solo);

    row.append(chip, name, buttons);
    container.appendChild(row);
  }
}

function applyAudioState() {
  const reference = !!state.referenceOn;
  for (const track of state.tracks) {
    const audio = state.audios[track.id];
    if (!audio) continue;
    const audible = state.solo ? state.solo === track.id : !state.muted[track.id];
    audio.muted = reference || !audible;
    audio.volume = state.master;
  }
  if (state.mix) {
    state.mix.muted = !reference;
    state.mix.volume = state.master;
  }
  for (const head of document.querySelectorAll(".track-head")) {
    const id = head.dataset.track;
    const audible = state.solo ? state.solo === id : !state.muted[id];
    head.classList.toggle("muted", reference || !audible);
    const mute = head.querySelector("button:not(.solo)");
    const solo = head.querySelector("button.solo");
    if (mute) mute.setAttribute("aria-pressed", String(!!state.muted[id]));
    if (solo) solo.setAttribute("aria-pressed", String(state.solo === id));
  }
}

function toggleMute(id) {
  state.muted[id] = !state.muted[id];
  if (state.muted[id] && state.solo === id) state.solo = null;
  applyAudioState();
  drawGrid();
}

function toggleSolo(id) {
  state.solo = state.solo === id ? null : id;
  applyAudioState();
  drawGrid();
}

/* ------------------------------------------------------------------ layout */
function layout() {
  const scroll = $("gridScroll");
  const width = scroll.clientWidth || 900;
  const bars = Math.max(1, state.bars.length);
  state.cellW = clamp(Math.floor(width / bars), 14, 28);
  state.stepsPerCell = state.cellW >= 18 ? 16 : 4;
  state.rowH = 74;
  document.querySelectorAll(".track-head").forEach(node => { node.style.height = state.rowH + "px"; });
}

function timeForX(x) {
  const bars = state.bars;
  if (!bars.length) return 0;
  const index = clamp(Math.floor(x / state.cellW), 0, bars.length - 1);
  const fraction = clamp(x / state.cellW - index, 0, 1);
  const bar = bars[index];
  return bar.start + fraction * (bar.end - bar.start);
}

function xForTime(t) {
  const bars = state.bars;
  if (!bars.length) return 0;
  for (let i = 0; i < bars.length; i += 1) {
    const bar = bars[i];
    if (t < bar.end || i === bars.length - 1) {
      const fraction = clamp((t - bar.start) / Math.max(1e-3, bar.end - bar.start), 0, 1);
      return (i + fraction) * state.cellW;
    }
  }
  return bars.length * state.cellW;
}

/* ------------------------------------------------------------------ canvas */
function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function cellValues(track, barIndex) {
  const per = 16;
  const values = [];
  for (let k = 0; k < state.stepsPerCell; k += 1) {
    let value = 0;
    const group = per / state.stepsPerCell;
    for (let g = 0; g < group; g += 1) {
      value = Math.max(value, track.steps?.[barIndex * per + k * group + g] ?? 0);
    }
    values.push(value);
  }
  return values;
}

function drawGrid() {
  if (!state.tracks.length || !state.bars.length) return;
  const canvas = $("gridCanvas");
  const width = state.bars.length * state.cellW;
  const height = state.tracks.length * state.rowH;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const overlay = $("overlayCanvas");
  overlay.style.width = width + "px";
  overlay.style.height = height + "px";
  overlay.width = Math.round(width * dpr);
  overlay.height = Math.round(height * dpr);

  ctx.clearRect(0, 0, width, height);
  state.tracks.forEach((track, row) => {
    const y = row * state.rowH;
    const audible = state.solo ? state.solo === track.id : !state.muted[track.id];
    ctx.fillStyle = row % 2 ? "#141d20" : "#151f22";
    ctx.fillRect(0, y, width, state.rowH);
    ctx.globalAlpha = audible ? 1 : 0.32;
    const loudest = Math.max(...state.tracks.map(item => item.rms_db ?? -100));
    const levelScale = Math.max(0.18, Math.min(1, Math.pow(10, ((track.rms_db ?? -100) - loudest) / 30)));
    const inner = 7;
    state.bars.forEach((bar, b) => {
      const x = b * state.cellW;
      ctx.fillStyle = hexToRgba(track.color, 0.06);
      roundRect(ctx, x + 1, y + inner, state.cellW - 2, state.rowH - inner * 2, 5);
      ctx.fill();
      const values = cellValues(track, b);
      const stepW = Math.max(1, (state.cellW - 2) / values.length);
      const maxH = state.rowH - inner * 2 - 8;
      values.forEach((value, k) => {
        if (value <= 0.02) return;
        const h = Math.max(2, value * maxH);
        ctx.fillStyle = hexToRgba(track.color, (0.22 + 0.72 * value) * levelScale);
        ctx.fillRect(x + 1 + k * stepW, y + inner + (maxH - h) + 4, Math.max(1, stepW - 0.6), h);
      });
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#ffffff10";
    ctx.beginPath();
    ctx.moveTo(0, y + state.rowH + 0.5);
    ctx.lineTo(width, y + state.rowH + 0.5);
    ctx.stroke();
  });

  for (let b = 0; b <= state.bars.length; b += 1) {
    const x = b * state.cellW + 0.5;
    ctx.strokeStyle = b % 4 === 0 ? "#ffffff22" : "#ffffff0c";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  drawOverlay();
}

function drawOverlay() {
  if (!state.tracks.length) return;
  const canvas = $("overlayCanvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = state.bars.length * state.cellW;
  const height = state.tracks.length * state.rowH;
  ctx.clearRect(0, 0, width, height);

  if (state.hoverBar >= 0 && state.hoverBar < state.bars.length) {
    ctx.fillStyle = "#ffffff08";
    ctx.fillRect(state.hoverBar * state.cellW, 0, state.cellW, height);
  }
  const current = currentBarIndex();
  if (current >= 0) {
    ctx.strokeStyle = "#b6e7ce44";
    ctx.strokeRect(current * state.cellW + 0.5, 0.5, state.cellW - 1, height - 1);
  }
  if (state.loopBar && state.loopIndex >= 0) {
    ctx.strokeStyle = "#ebc894aa";
    ctx.lineWidth = 2;
    ctx.strokeRect(state.loopIndex * state.cellW + 1, 1, state.cellW - 2, height - 2);
    ctx.lineWidth = 1;
  }
  const x = xForTime(currentTime());
  ctx.strokeStyle = "#ebc894";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, height);
  ctx.stroke();
  ctx.lineWidth = 1;
}

function currentBarIndex() {
  const t = currentTime();
  for (let i = 0; i < state.bars.length; i += 1) {
    if (t < state.bars[i].end) return i;
  }
  return state.bars.length - 1;
}

function hexToRgba(hex, alpha) {
  const value = (hex || "#b6e7ce").replace("#", "");
  const r = parseInt(value.slice(0, 2), 16), g = parseInt(value.slice(2, 4), 16), b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const prettifyChord = label => {
  if (!label || label === "N") return "N";
  const [root, quality = ""] = String(label).split(":");
  if (!quality || quality === "maj") return root;
  if (quality.startsWith("min")) return `${root}m${quality.slice(3)}`;
  if (quality.startsWith("maj")) return `${root}${quality}`;
  if (quality.startsWith("dim")) return `${root}°${quality.slice(3)}`;
  if (quality.startsWith("aug")) return `${root}+${quality.slice(3)}`;
  if (quality.startsWith("sus")) return `${root}${quality}`;
  return `${root}${quality}`;
};

function chordTimeline(result) {
  // Bar-level chords: the analysis already gives each bar its chord spans, so label the grid
  // bar by bar (longest span wins) instead of replaying the 0.25 s cell timeline.
  const rows = result?.bars || [];
  const bars = state.bars || [];
  const out = [];
  if (!rows.length || !bars.length) return out;
  let previous = null;
  for (let index = 0; index < bars.length; index += 1) {
    const start = bars[index].start;
    const end = bars[index].end;
    let best = null;
    let bestOverlap = 0;
    for (const row of rows) {
      if (row.end <= start) continue;
      if (row.start >= end) break;
      for (const span of row.spans || []) {
        const overlap = Math.min(span.end, end) - Math.max(span.start, start);
        if (overlap > bestOverlap) { bestOverlap = overlap; best = span.chord; }
      }
    }
    if (best && best !== previous) {
      out.push({ bar: index, label: prettifyChord(best) });
      previous = best;
    }
  }
  return out;
}

function buildRuler() {
  const ruler = $("ruler");
  ruler.replaceChildren();
  ruler.style.width = state.bars.length * state.cellW + "px";
  const step = state.bars.length > 96 ? 8 : state.bars.length > 48 ? 4 : state.bars.length > 24 ? 2 : 1;
  state.bars.forEach((bar, index) => {
    if (index % step !== 0) return;
    const span = document.createElement("span");
    span.textContent = bar.index ?? index + 1;
    span.style.left = index * state.cellW + "px";
    if (index % 4 === 0) span.className = "bar-start";
    ruler.appendChild(span);
  });
  // Chord names above the bar numbers, only where a new chord starts and where they fit.
  let lastX = -Infinity;
  for (const chord of state.chords || []) {
    const x = chord.bar * state.cellW;
    if (x - lastX < 34) continue;
    const node = document.createElement("span");
    node.className = "chord";
    node.textContent = chord.label;
    node.style.left = x + "px";
    ruler.appendChild(node);
    lastX = x;
  }
}

/* ------------------------------------------------------------------ transport */
function currentTime() {
  const audio = master();
  return audio && Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
}

function seek(time) {
  const t = clamp(time, 0, Math.max(0, (state.manifest?.duration || 0) - 0.05));
  for (const track of state.tracks) {
    const audio = state.audios[track.id];
    if (audio && audio.readyState >= 1) audio.currentTime = t;
  }
  if (state.mix && state.mix.readyState >= 1) state.mix.currentTime = t;
  renderClock(t);
  drawOverlay();
}

async function play() {
  const audio = master();
  if (!audio) return;
  const t = clamp(currentTime(), 0, Math.max(0, state.manifest.duration - 0.1));
  for (const track of state.tracks) {
    const node = state.audios[track.id];
    if (!node) continue;
    try { node.currentTime = t; } catch (_) { /* not ready yet */ }
  }
  if (state.mix) { try { state.mix.currentTime = t; } catch (_) { /* ignore */ } }
  applyAudioState();
  await Promise.allSettled([...state.tracks.map(track => state.audios[track.id]?.play()),
                            state.mix ? state.mix.play() : Promise.resolve()]);
  state.playing = true;
  $("play").textContent = "❚❚";
  $("play").setAttribute("aria-label", "暂停");
  loop();
}

function stop() {
  for (const track of state.tracks) state.audios[track.id]?.pause();
  state.mix?.pause();
  state.playing = false;
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "播放");
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = null;
  renderClock();
  drawOverlay();
}

function loop() {
  if (!state.playing) return;
  const audio = master();
  const t = currentTime();
  for (const track of state.tracks) {
    const node = state.audios[track.id];
    if (!node || node === audio) continue;
    if (Math.abs(node.currentTime - t) > 0.08 && node.readyState >= 1) {
      try { node.currentTime = t; } catch (_) { /* ignore */ }
    }
  }
  if (state.mix && Math.abs(state.mix.currentTime - t) > 0.08 && state.mix.readyState >= 1) {
    try { state.mix.currentTime = t; } catch (_) { /* ignore */ }
  }
  if (state.loopBar && state.loopIndex >= 0 && state.bars[state.loopIndex]) {
    const bar = state.bars[state.loopIndex];
    if (t >= bar.end - 0.02) seek(bar.start);
  }
  renderClock(t);
  drawOverlay();
  if ($("follow").checked) {
    const scroll = $("gridScroll");
    const x = xForTime(t);
    const left = scroll.scrollLeft;
    const width = scroll.clientWidth;
    if (x < left + 40) scroll.scrollLeft = Math.max(0, x - 40);
    else if (x > left + width - 80) scroll.scrollLeft = Math.max(0, x - width + 80);
  }
  state.raf = requestAnimationFrame(loop);
}

function renderClock(t = currentTime()) {
  $("clock").textContent = fmt(t);
}

function attachTransport() {
  $("play").onclick = () => (state.playing ? stop() : play());
  $("master").oninput = event => { state.master = Number(event.target.value); applyAudioState(); };
  $("reference").onchange = event => {
    state.referenceOn = event.target.checked;
    if (state.referenceOn && state.mix) {
      try { state.mix.currentTime = currentTime(); } catch (_) { /* ignore */ }
      if (state.playing) state.mix.play().catch(() => {});
    }
    applyAudioState();
    drawGrid();
  };
  $("loopBar").onchange = event => {
    state.loopBar = event.target.checked;
    state.loopIndex = state.loopBar ? currentBarIndex() : -1;
    drawOverlay();
  };
  applyAudioState();
}

function attachGridEvents() {
  const wrap = $("canvasWrap");
  wrap.onmousemove = event => {
    const rect = wrap.getBoundingClientRect();
    const bar = clamp(Math.floor((event.clientX - rect.left) / state.cellW), 0, state.bars.length - 1);
    if (bar !== state.hoverBar) { state.hoverBar = bar; drawOverlay(); }
  };
  wrap.onmouseleave = () => { state.hoverBar = -1; drawOverlay(); };
  wrap.onclick = event => {
    const rect = wrap.getBoundingClientRect();
    const x = event.clientX - rect.left;
    seek(timeForX(x));
    if (state.loopBar) state.loopIndex = clamp(Math.floor(x / state.cellW), 0, state.bars.length - 1);
    if (!state.playing) play();
    drawOverlay();
  };
}

function attachKeys() {
  if (state.keysAttached) return;
  state.keysAttached = true;
  document.addEventListener("keydown", event => {
    if (event.target.closest("input,textarea,select,button,a")) return;
    if (event.code === "Space") { event.preventDefault(); state.playing ? stop() : play(); }
    else if (event.code === "ArrowLeft") { event.preventDefault(); seek(currentTime() - 5); }
    else if (event.code === "ArrowRight") { event.preventDefault(); seek(currentTime() + 5); }
  });
  window.addEventListener("resize", () => {
    const wasScroll = $("gridScroll").scrollLeft;
    layout(); buildRuler(); drawGrid();
    $("gridScroll").scrollLeft = wasScroll;
  });
}

/* ------------------------------------------------------------------ info */
function updateInfo() {
  const container = $("trackInfo");
  container.replaceChildren();
  for (const track of state.tracks) {
    const row = document.createElement("div");
    row.className = "track-info-row";
    const chip = document.createElement("span");
    chip.className = "track-chip";
    chip.style.background = track.color;
    chip.style.height = "14px";
    const label = document.createElement("span");
    label.textContent = `${track.name} · ${track.short}`;
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = clamp((track.rms_db + 60) / 60, 0, 1) * 100 + "%";
    fill.style.background = track.color;
    bar.appendChild(fill);
    row.append(chip, label, bar);
    container.appendChild(row);

    const detail = document.createElement("div");
    detail.className = "track-info-detail";
    detail.style.paddingLeft = "18px";
    const active = `${Math.round((track.active_ratio || 0) * 100)}% 时间有声音`;
    const range = track.first_active == null ? "全程静音" : `${fmt(track.first_active)}–${fmt(track.last_active)}`;
    const bits = [active, range, `${track.register || "—"}（${Math.round(track.centroid_hz || 0)} Hz）`,
      `峰值 ${track.peak_db} dB`];
    if (track.id === "bass" && track.dominant_pitch_class != null) bits.push(`低音主音 ${NOTE_NAMES[track.dominant_pitch_class]}`);
    if (track.silent) bits.push("本曲几乎无声");
    detail.textContent = bits.join(" · ");
    container.appendChild(detail);
  }
}

boot();

