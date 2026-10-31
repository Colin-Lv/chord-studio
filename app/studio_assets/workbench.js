// Shared lexical scope with studio.js. Exported pages remain fully offline.
const byId = (id) => document.getElementById(id);
const SECTION_ROLES = { unknown: "未确认", intro: "前奏", verse: "主歌", pre_chorus: "预副歌", chorus: "副歌", bridge: "桥段", instrumental: "间奏", outro: "尾奏", start: "开头", end: "结尾", inst: "器乐", solo: "独奏" };
let localAudioUrl = "";
const sessionAudioSrc = DATA.audio_src || DATA.audio_file;
let waveformPeaks = null;
let embeddedWaveformReference;
let historyEntries = [];
let historyIndex = -1;
let restoringHistory = false;
let selectedSectionId = "";
let loopRange = { a: null, b: null, enabled: false };
let auditionEnd = null;
let followEnabled = true;
let followSuspendedUntil = 0;
let lastFollowedBar = null;
let toastTimer = null;
let lastPulseBeat = null;
let cachedBarNodes = new Map();
let cachedBeatNodes = new Map();
let cachedSpanNodes = new Map();
let cachedEventNodes = new Map();
let lastActiveSpan = null;
let currentSectionId = null;
let waveformGeneration = 0;
let playbackGuardTimer = null;
let timelineAudit = { conflicted: false, canRebuild: false, overlapCount: 0, mismatchCount: 0, examples: [] };
let editKeyOverride = null;

function canEditTimeline() {
  if (!timelineAudit.conflicted) return true;
  toast("时间轴存在冲突。请先审阅并选择是否按小节谱格重建，再修改和弦。",6000);
  return false;
}

function auditTimeline(data = DATA) {
  const rows = Array.isArray(data.events) ? data.events : [];
  const spans = (data.bars || []).flatMap((bar) => (bar.spans || []).map((span) => ({ ...span, bar: bar.bar })));
  const result = { conflicted:false,canRebuild:spans.length > 0,overlapCount:0,mismatchCount:0,invalidCount:0,examples:[] };
  const valid = (item) => Number.isFinite(Number(item.start)) && Number.isFinite(Number(item.end)) && Number(item.end) > Number(item.start) && Number(item.start) >= 0 && Number(item.end) <= playbackDuration+.1;
  result.invalidCount = rows.filter((item) => !valid(item)).length;
  const sortedRows = rows.filter(valid).slice().sort((a,b) => Number(a.start)-Number(b.start));
  let lastEnd = -Infinity;
  for (const item of sortedRows) { if (Number(item.start) < lastEnd - .001) result.overlapCount += 1; lastEnd = Math.max(lastEnd,Number(item.end)); }
  let spanEnd = -Infinity;
  const sortedSpans = spans.slice().sort((a,b) => Number(a.start)-Number(b.start));
  for (const span of sortedSpans) { if (!valid(span) || Number(span.start) < spanEnd - .001) result.canRebuild = false; spanEnd = Math.max(spanEnd,Number(span.end)); }
  if (spans.length) {
    const points = [...new Set([...sortedRows,...sortedSpans].flatMap((item) => [Number(item.start),Number(item.end)]).filter(Number.isFinite))].sort((a,b) => a-b);
    const chord = (item) => String(item.chord_display || item.chord || "N").trim();
    let rowIndex = 0, spanIndex = 0;
    let activeRows = [], activeSpans = [];
    for (let i = 0; i < points.length-1; i++) {
      if (points[i+1]-points[i] < .002) continue;
      const center = (points[i]+points[i+1])/2;
      while (rowIndex < sortedRows.length && Number(sortedRows[rowIndex].start) <= center) activeRows.push(sortedRows[rowIndex++]);
      while (spanIndex < sortedSpans.length && Number(sortedSpans[spanIndex].start) <= center) activeSpans.push(sortedSpans[spanIndex++]);
      activeRows = activeRows.filter((item) => Number(item.end) > center);
      activeSpans = activeSpans.filter((item) => Number(item.end) > center);
      const rowLabels = [...new Set(activeRows.map(chord))].sort().join(" / ");
      const spanLabels = [...new Set(activeSpans.map(chord))].sort().join(" / ");
      if (rowLabels !== spanLabels) {
        result.mismatchCount += 1;
        if (result.examples.length < 4) result.examples.push({ start:points[i],end:points[i+1],events:rowLabels || "空缺",spans:spanLabels || "空缺" });
      }
    }
  }
  result.conflicted = result.invalidCount > 0 || result.overlapCount > 0 || result.mismatchCount > 0;
  return result;
}

function renderTimelineAudit() {
  timelineAudit = auditTimeline();
  byId("timelineAlert").hidden = !timelineAudit.conflicted;
  if (timelineAudit.conflicted) {
    byId("timelineAlertText").textContent = `检测到 ${timelineAudit.overlapCount} 处事件重叠、${timelineAudit.mismatchCount} 处事件与小节谱格标注不一致${timelineAudit.invalidCount ? `、${timelineAudit.invalidCount} 处非法时间` : ""}。目前可审阅和导出，和弦编辑暂不可用；不会自动决定哪套标注正确。`;
    byId("timelineConflictExamples").innerHTML = timelineAudit.examples.map((item) => `<div>${preciseTime(item.start)}–${preciseTime(item.end)}：事件 ${escapedAttr(item.events)} / 谱格 ${escapedAttr(item.spans)}</div>`).join("");
  }
  byId("rebuildEventTimeline").disabled = !timelineAudit.canRebuild;
  for (const id of ["saveChord","clearChord","resetChord","splitHalf","splitBeats","split8","split16","mergeSame","pasteCell","pasteBar","applyScore","applyBoundary","splitAtPlayhead"]) byId(id).disabled = timelineAudit.conflicted || ((id === "applyBoundary" || id === "splitAtPlayhead") && !canAdjustSelectedBoundary());
}

function rebuildEventTimeline() {
  if (!timelineAudit.canRebuild) { toast("小节谱格本身也有重叠或非法时间，无法安全重建。"); return; }
  DATA.events = DATA.bars.flatMap((bar) => (bar.spans || []).map((span) => ({ ...span,bar:Number(bar.bar) }))).sort((a,b) => Number(a.start)-Number(b.start));
  DATA.timeline_reconciliation = { source:"bar_spans",reason:"explicit_user_choice",at:new Date().toISOString(),previous_overlap_count:timelineAudit.overlapCount,previous_mismatch_count:timelineAudit.mismatchCount };
  selectedEventIndex = 0; selectedBeatTarget = null; dirtyEvents.clear();
  refreshDerived(); renderStable(); updateActive(); autosaveDraft("timeline-rebuild");
  toast("已按小节谱格重建事件，可撤销回到原始冲突状态。");
}

function withKeyAtTime(time, callback) {
  const previous = editKeyOverride;
  editKeyOverride = keyAtTime(Number(time));
  try { return callback(); } finally { editKeyOverride = previous; }
}

function enforcePlaybackRange() {
  if (audio.paused) return;
  const time = analysisTime();
  if (loopRange.enabled && loopRange.a != null && loopRange.b != null && time >= loopRange.b - .005) {
    audio.currentTime = playbackStart + clamp(loopRange.a - syncMs / 1000, 0, playbackDuration);
  } else if (auditionEnd != null && time >= auditionEnd) {
    const end = auditionEnd; auditionEnd = null;
    audio.currentTime = playbackStart + clamp(end - syncMs / 1000,0,playbackDuration);
    audio.pause();
  } else if (playbackDuration > 0 && audio.currentTime >= playbackEnd) {
    audio.currentTime = playbackEnd;
    audio.pause();
  }
}

function toast(message, duration = 3200) {
  const node = byId("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, duration);
}

function preciseTime(value) {
  const milliseconds = Math.round(Math.max(0, Number(value) || 0) * 1000);
  return `${fmt(milliseconds / 1000)}.${String(milliseconds % 1000).padStart(3, "0")}`;
}

function openSeekDialog() {
  byId("seekTimeInput").value = preciseTime(localPlaybackTime());
  byId("seekTimeInput").removeAttribute("aria-invalid");
  byId("seekTimeHint").textContent = `范围 00:00.000–${preciseTime(playbackDuration)}${playbackStart ? `；分析片段在原音频中的偏移为 ${preciseTime(playbackStart)}` : ""}。`;
  byId("seekDialog").showModal();
  byId("seekTimeInput").focus(); byId("seekTimeInput").select();
}

function applyPreciseSeek(play = false) {
  const raw = byId("seekTimeInput").value.trim();
  const parts = raw.split(":");
  const valid = /^\d+(?::\d{1,2}){0,2}(?:\.\d{1,3})?$/.test(raw) && parts.length <= 3 && (parts.length === 1 || Number(parts[parts.length-1]) < 60) && (parts.length < 3 || Number(parts[1]) < 60);
  const time = parts.reduce((value,part) => value*60+Number(part),0);
  if (!valid || !Number.isFinite(time) || time < 0 || time > playbackDuration) { byId("seekTimeInput").setAttribute("aria-invalid","true"); byId("seekTimeHint").textContent = `请输入 0–${playbackDuration.toFixed(3)} 秒，或如 01:23.450 的时间。`; return; }
  auditionEnd = null;
  audio.currentTime = playbackStart+time;
  updateActive();
  const item = activeFor(analysisTime());
  if (item) { selectEventByObject(item); revealBar(beatFor(analysisTime())?.bar ?? item.bar); }
  byId("seekDialog").close();
  if (play) playAudio();
}

function evidenceLabel(item) {
  if (item?.edited) return "人工修改";
  const score = confidenceScore(item);
  if (score == null) return "证据未提供";
  return `${score < 0.52 ? "待复核 · " : ""}证据 ${score.toFixed(2)}`;
}

function confidenceScore(item) {
  if (item?.confidence == null || /^(not_|unknown|unavailable|missing)/i.test(String(item?.confidence_status || ""))) return null;
  const score = Number(item.confidence);
  return Number.isFinite(score) ? score : null;
}

function evidenceExplanation(item) {
  if (item?.edited) return "此片段已由人工修改。";
  if (confidenceScore(item) == null) return "该模型未提供此片段的证据分值。";
  if ((item?.confidence_kind || DATA.confidence_kind) === "weighted_temporal_support") return "表示该时间段中和弦标签的支持程度，不是模型预测正确率。";
  return "模型证据分值未经概率校准，不表示识别正确率。";
}

function barNeedsReview(bar) {
  return (bar.spans?.length ? bar.spans : bar.beats || []).some((item) => !item.edited && (confidenceScore(item) == null || confidenceScore(item) < .52));
}

function barMissingEvidence(bar) {
  const pending = (bar.spans?.length ? bar.spans : bar.beats || []).filter((item) => !item.edited);
  return pending.some((item) => confidenceScore(item) == null) && !pending.some((item) => confidenceScore(item) != null && confidenceScore(item) < .52);
}

function gridCountLabel(bars) {
  const located = bars.filter((bar) => bar.grid_status !== "uncovered").length;
  const uncovered = bars.length-located;
  return [located ? `${located} 小节` : "",uncovered ? `${uncovered} 个未定位区间` : ""].filter(Boolean).join(" + ") || "0 小节";
}

function eventGridLocation(item) {
  const bar = barByNumber(item.bar);
  if (!bar || bar.grid_status === "uncovered") return "拍点未定位";
  if (bar.grid_status === "incomplete") return `第 ${bar.bar} 小节 · 拍点不完整`;
  if (Number(item.end) > Number(bar.end)+.001 || Number(item.start) < Number(bar.start)-.001) return `第 ${bar.bar} 小节起 · 跨小节`;
  const beat = (bar.beats || []).find((row) => Number(item.start) >= Number(row.start)-.001 && Number(item.start) < Number(row.end)-.001);
  return `第 ${bar.bar} 小节${beat ? ` · 第 ${beat.beat} 拍起` : ""}`;
}

function sectionRangeLabel(section) {
  const first = barByNumber(section.start_bar), last = barByNumber(section.end_bar);
  const firstUncovered = !first || first.grid_status === "uncovered";
  const lastUncovered = !last || last.grid_status === "uncovered";
  if (firstUncovered && lastUncovered) return `${fmt(section.start)}–${fmt(section.end)}`;
  if (firstUncovered) return `未定位区间–${section.end_bar} 小节`;
  if (lastUncovered) return `${section.start_bar} 小节–未定位区间`;
  return `${section.start_bar}–${section.end_bar} 小节`;
}

function beatCountForBar(bar) {
  if (bar?.grid_status === "uncovered") return 0;
  const meter = String(bar?.meter || DATA.meter || "").match(/^(\d+)\/(\d+)$/);
  const count = Number(bar?.beats_per_bar) || (meter ? Number(meter[1]) : 0) || Number(DATA.beats_per_bar) || Math.max(0,...(bar?.beats || []).map((beat) => Number(beat.beat) || 0));
  return Math.max(0,Math.min(32,Math.floor(count)));
}

function meterDenominatorForBar(bar) {
  const meter = String(bar?.meter || DATA.meter || "").match(/^(\d+)\/(\d+)$/);
  return meter && Number(meter[2]) > 0 ? Number(meter[2]) : 4;
}

function sections() {
  return (Array.isArray(DATA.structure?.sections) ? DATA.structure.sections : [])
    .filter((section) => Number.isFinite(Number(section.start)) && Number(section.end) > Number(section.start))
    .slice().sort((a, b) => Number(a.start) - Number(b.start));
}

function sectionAt(time) {
  return sections().find((section) => time >= Number(section.start) && time < Number(section.end)) || null;
}

function sectionTitle(section) {
  return String(section.name || (section.role && section.role !== "unknown" ? SECTION_ROLES[section.role] || section.role : `段落 ${section.label || section.group || ""}`));
}

function sectionHue(section) {
  const token = String(section.group || section.repeat_group || section.label || section.id || "A");
  let hash = 0;
  for (const ch of token) hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  return [152, 215, 35, 280, 185, 12, 80, 330][Math.abs(hash) % 8];
}

function sectionHeadingForBar(bar) {
  const section = sections().find((item) => Number(item.start_bar) === Number(bar.bar));
  if (!section) return "";
  const role = section.role && section.role !== "unknown" ? ` · ${SECTION_ROLES[section.role] || section.role}` : "";
  return `<div class="score-section-heading" data-section-heading="${escapedAttr(section.id)}" data-first-bar="${bar.bar}"><span class="section-letter" style="--section-hue:${sectionHue(section)}">${escapedAttr(section.label || section.group || "·")}</span>${escapedAttr(sectionTitle(section))}<small>${escapedAttr(sectionRangeLabel(section))}${escapedAttr(role)}</small><span class="section-rule"></span></div>`;
}

function keyAtTime(time) {
  const timelines = [DATA.key_timeline?.segments || DATA.key_timeline, DATA.modulations, DATA.sections];
  let chosen = null;
  for (const timeline of timelines) {
    for (const item of Array.isArray(timeline) ? timeline : []) {
      const bar = barByNumber(item.start_bar);
      const start = Number(item.start ?? item.start_time ?? item.time ?? bar?.start ?? 0);
      const endBar = barByNumber(item.end_bar);
      const end = Number(item.end ?? item.end_time ?? endBar?.end ?? Infinity);
      if (item.key && time >= start && time < end && (!chosen || start >= chosen.start)) chosen = { start, key: item.key, mode: item.mode };
    }
  }
  const localBar = (DATA.bars || []).find((bar) => time >= Number(bar.start) && time < Number(bar.end));
  if (localBar?.local_key) chosen = { start:Number(localBar.start),key:localBar.local_key,mode:localBar.local_mode };
  let key = String(chosen?.key || DATA.key || "");
  const mode = chosen ? chosen.mode || (key.endsWith("m") ? "minor" : "major") : DATA.mode || "";
  if (key && /minor|min|小调/i.test(mode) && !key.endsWith("m")) key += "m";
  return key;
}

function editingKey() {
  if (editKeyOverride != null) return editKeyOverride;
  const item = selectedBeatTarget ? beatByNumber(selectedBeatTarget.bar, selectedBeatTarget.beat) : events[selectedEventIndex];
  return keyAtTime(Number(item?.start || 0));
}

function openKeyCandidates() {
  const estimate = DATA.key_estimation;
  const candidates = (Array.isArray(estimate?.candidates) ? estimate.candidates : []).filter((candidate) => candidate.key && Number.isFinite(Number(candidate.score))).slice().sort((a,b) => Number(b.score)-Number(a.score));
  if (!candidates.length) return;
  const start = Number(estimate.audio_offset || 0);
  const end = start + Number(estimate.duration || playbackDuration);
  const margin = candidates.length > 1 ? Number(candidates[0].score)-Number(candidates[1].score) : null;
  byId("keyCandidateContext").textContent = `估计音频范围 ${preciseTime(start)}–${preciseTime(end)}${margin != null ? `；前两候选分差 ${margin.toFixed(3)}` : ""}。当前显示调性：${keyAtTime(analysisTime()) || "未确定"}。`;
  byId("keyCandidateList").innerHTML = candidates.slice(0,4).map((candidate,index) => `<li><span class="rank">${String(index+1).padStart(2,"0")}</span><strong>${escapedAttr(candidate.key)}</strong><span class="score">证据 ${Number(candidate.score).toFixed(3)}</span></li>`).join("");
  byId("keyCandidatesDialog").showModal();
}

function preparePrint() {
  byId("printTitle").textContent = byId("songTitle").textContent;
  byId("printMeta").textContent = `${bpmLabel()} BPM · ${DATA.meter || "拍号未知"} · ${DATA.key || "调性未知"} ${DATA.mode || ""} · ${fmt(playbackDuration)} · ${byId("reviewOnly").checked ? "仅待复核小节" : "完整谱面"} · ${displayMode === "roman" ? "级数" : "和弦名"}。带下划线的和弦已人工修改；自动段落功能为待确认建议。`;
}

function updateSelectionDetails(item) {
  const chip = byId("selectionConfidence");
  chip.textContent = evidenceLabel(item);
  chip.classList.toggle("review", !item?.edited && confidenceScore(item) != null && confidenceScore(item) < .52);
  chip.title = evidenceExplanation(item);
  byId("selectionEvidence").textContent = evidenceExplanation(item);
  byId("eventStart").value = Number(item?.start || 0).toFixed(3);
  byId("eventEnd").value = Number(item?.end || 0).toFixed(3);
  byId("applyBoundary").disabled = !canAdjustSelectedBoundary() || timelineAudit.conflicted;
  byId("splitAtPlayhead").disabled = !canAdjustSelectedBoundary() || timelineAudit.conflicted;
  byId("jumpBar").value = item?.bar ?? "";
  const location = barByNumber(item?.bar)?.grid_status === "uncovered" ? "拍点未定位" : `第 ${item?.bar ?? selectedBarNumber()} 小节`;
  byId("mobileSelection").textContent = item ? `${item.chord_display || item.chord || "N"} · ${location}` : "选择一个和弦片段";
  byId("mobileSelectionTime").textContent = item ? `${preciseTime(item.start)}–${preciseTime(item.end)}` : "";
  document.querySelectorAll(".bar[data-bar]").forEach((node) => { node.tabIndex = Number(node.dataset.bar) === Number(item?.bar || selectedBarNumber()) ? 0 : -1; });
  renderSuggestionPanel();
  renderPathProposalPanel();
}

function snapshotState() {
  return JSON.stringify(DATA);
}

function recordHistory(reason = "edit") {
  if (restoringHistory) return;
  const snapshot = snapshotState();
  if (historyEntries[historyIndex]?.snapshot === snapshot) return;
  historyEntries = historyEntries.slice(0, historyIndex + 1);
  historyEntries.push({ snapshot, reason, event: selectedEventIndex, beat: selectedBeatTarget ? { ...selectedBeatTarget } : null });
  if (historyEntries.length > 60) historyEntries.shift();
  let storedCharacters = historyEntries.reduce((sum,entry) => sum + entry.snapshot.length,0);
  while (historyEntries.length > 2 && storedCharacters > 12 * 1024 * 1024) storedCharacters -= historyEntries.shift().snapshot.length;
  historyIndex = historyEntries.length - 1;
  updateHistoryButtons();
}

function updateHistoryButtons() {
  byId("undoEdit").disabled = historyIndex <= 0;
  byId("redoEdit").disabled = historyIndex >= historyEntries.length - 1;
}

function moveHistory(delta) {
  const next = historyIndex + delta;
  if (next < 0 || next >= historyEntries.length) return;
  historyIndex = next;
  const entry = historyEntries[next];
  const selectionTime = Number(selectedCellItem()?.start ?? 0);
  const selectionBeat = selectedBeatTarget ? { ...selectedBeatTarget } : null;
  const state = JSON.parse(entry.snapshot);
  restoringHistory = true;
  for (const key of Object.keys(DATA)) delete DATA[key];
  Object.assign(DATA,state);
  dirtyEvents.clear();
  refreshDerived();
  selectedEventIndex = events.findIndex((event) => selectionTime >= Number(event.start) && selectionTime < Number(event.end));
  if (selectedEventIndex < 0) selectedEventIndex = clamp(entry.event,0,Math.max(0,events.length-1));
  selectedBeatTarget = selectionBeat && beatByNumber(selectionBeat.bar,selectionBeat.beat) ? selectionBeat : null;
  renderStable();
  updateActive();
  autosaveDraft(delta < 0 ? "undo" : "redo");
  restoringHistory = false;
  updateHistoryButtons();
  toast(delta < 0 ? "已撤销修改" : "已重做修改");
}

function seekToAnalysis(time, play = false) {
  auditionEnd = null;
  const target = clamp(Number(time) || 0, 0, playbackDuration);
  audio.currentTime = playbackStart + clamp(target - syncMs / 1000, 0, playbackDuration);
  updateActive();
  if (play) playAudio();
}

function openLinkedTimeRange() {
  const params = new URLSearchParams(location.hash.slice(1));
  if (!params.has("t") || params.get("t").trim() === "") return;
  const start = Number(params.get("t"));
  const end = params.has("end") ? Number(params.get("end")) : NaN;
  if (!Number.isFinite(start) || start < 0 || start > playbackDuration) return;
  const validEnd = Number.isFinite(end) && end > start && end <= playbackDuration;
  const locate = () => {
    audio.pause();
    loopRange = validEnd ? { a:start,b:end,enabled:false } : { a:null,b:null,enabled:false };
    renderLoop();
    const item = activeFor(Math.min(start+.00001,playbackDuration));
    if (item) { selectEventByObject(item); revealBar(item.bar); }
    seekToAnalysis(start);
    toast(validEnd ? "已定位试听区间；按空格播放，按 L 开启 A/B 循环。" : "已定位到链接时间；按空格播放。",6000);
  };
  if (audio.readyState >= 1) locate();
  else audio.addEventListener("loadedmetadata",locate,{ once:true });
}

function revealBar(barNumber, force = true) {
  const node = cachedBarNodes.get(Number(barNumber));
  if (!node || node.hidden) return;
  const selection = node.classList.contains("uncovered-long") ? (cachedSpanNodes.get(selectedEventIndex) || []).find((span) => Number(span.dataset.bar) === Number(barNumber)) : null;
  revealNode(selection || node, force);
}

function revealNode(node, force = true) {
  if (!node) return;
  const rect = node.getBoundingClientRect();
  const headerBottom = document.querySelector(".topbar").getBoundingClientRect().bottom;
  if (force || rect.top < headerBottom + 10 || rect.bottom > window.innerHeight - 50) {
    const target = window.scrollY + rect.top - headerBottom - 32;
    window.scrollTo({ top: Math.max(0, target), behavior: "instant" });
  }
}

function jumpBar(delta = 0) {
  const current = beatFor(analysisTime())?.bar || events[selectedEventIndex]?.bar || DATA.bars[0]?.bar;
  const bar = delta ? DATA.bars[clamp(DATA.bars.findIndex((b) => Number(b.bar) === Number(current)) + delta, 0, DATA.bars.length - 1)] : barByNumber(Number(byId("jumpBar").value));
  if (!bar) { toast("未找到这个小节。"); return; }
  const index = events.findIndex((event) => Number(event.bar) === Number(bar.bar));
  if (index >= 0) selectEvent(index);
  seekToAnalysis(bar.start);
  revealBar(bar.bar);
}

function activateInspector(panel) {
  document.querySelectorAll("[data-panel]").forEach((button) => {
    const active = button.dataset.panel === panel;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    byId(button.dataset.panel).hidden = !active;
  });
}

function editSelectedInput() {
  activateInspector("editPanel");
  if (window.innerWidth <= 700) setMobileInspector(true);
  chordInput.focus({ preventScroll: true });
  chordInput.select();
  if (window.innerWidth <= 700) document.querySelector(".inspector").scrollTop = 0;
}

function setMobileInspector(open) {
  document.body.classList.toggle("inspector-open",open);
  byId("inspectorBackdrop").hidden = !open;
  byId("showInspector").setAttribute("aria-expanded",String(open));
  const inspector = document.querySelector(".inspector");
  if (open) { inspector.setAttribute("role","dialog"); inspector.setAttribute("aria-modal","true"); }
  else { inspector.removeAttribute("role"); inspector.removeAttribute("aria-modal"); }
  document.querySelectorAll(".topbar,.session-strip,.overview-panel,.structure-sidebar,.score-workspace,.app-footer,.mobile-editor-dock").forEach((node) => { node.inert = open; });
  if (open) byId("dismissInspector").focus({ preventScroll:true });
  else byId("showInspector").focus({ preventScroll:true });
}

function renderWorkbench() {
  cachedBarNodes = new Map([...document.querySelectorAll(".bar[data-bar]")].map((node) => [Number(node.dataset.bar), node]));
  cachedBeatNodes = new Map([...document.querySelectorAll(".beat[data-beat]")].map((node) => [`${node.dataset.bar}.${node.dataset.beat}`, node]));
  cachedSpanNodes = new Map();
  document.querySelectorAll(".span-cell[data-event]").forEach((node) => { const index = Number(node.dataset.event); if (!cachedSpanNodes.has(index)) cachedSpanNodes.set(index,[]); cachedSpanNodes.get(index).push(node); });
  cachedEventNodes = new Map([...document.querySelectorAll(".event[data-event]")].map((node) => [Number(node.dataset.event), node]));
  lastActiveSpan = null;
  currentSectionId = null;
  lastPulseBeat = null;
  barsEl.dataset.density = byId("densitySelect").value;
  barsEl.classList.toggle("hide-beats", !byId("showBeats").checked);
  cachedSpanNodes.forEach((nodes) => nodes.forEach((node) => node.classList.toggle("narrow", Number.parseFloat(node.style.width) < 22)));
  renderStructure();
  renderTimelineAudit();
  applyReviewFilter();
  refreshEmbeddedWaveform();
  drawOverview();
  byId("missingEvidenceLegend").hidden = !events.some((event) => !event.edited && confidenceScore(event) == null);
  byId("timeRuler").innerHTML = Array.from({ length: 7 }, (_, index) => `<span>${fmt(playbackDuration * index / 6)}</span>`).join("");
  byId("beatPulse").innerHTML = Array.from({ length: Math.max(0,Math.min(16, Number(DATA.beats_per_bar) || 0)) }, () => "<i></i>").join("");
  byId("jumpBar").max = Math.max(1, ...DATA.bars.map((bar) => Number(bar.bar)));
  byId("jumpBar").min = Math.min(1, ...DATA.bars.map((bar) => Number(bar.bar)));
  renderLoop();
  byId("keyCandidatesButton").hidden = !Array.isArray(DATA.key_estimation?.candidates) || !DATA.key_estimation.candidates.length;
  const incompleteCount = DATA.bars.filter((bar) => bar.grid_status === "incomplete").length;
  const gridReview = Boolean(DATA.structure?.grid_review_required || DATA.grid_review_required || incompleteCount);
  byId("gridReviewNotice").hidden = !gridReview;
  byId("gridReviewNotice").textContent = DATA.structure?.expert_grid?.applied_to_display ? "当前谱格采用专家拍点建议，尚需人工核对。请先试听强拍位置，再确认小节与段落。" : incompleteCount ? `${incompleteCount} 个小节的拍点不完整，已保留实际观测位置。可修改和弦标签，按拍拆分与铺谱需先核对网格。` : "当前拍点与小节网格仍待人工核对。可先按时间试听和弦片段，再确认小节与段落。";
  updateHistoryButtons();
}

function applyReviewFilter(reselect = false) {
  const enabled = byId("reviewOnly").checked;
  let shown = 0;
  for (const bar of DATA.bars) {
    const node = cachedBarNodes.get(Number(bar.bar));
    if (!node) continue;
    node.hidden = enabled && !barNeedsReview(bar);
    if (!node.hidden) shown += 1;
  }
  document.querySelectorAll(".score-section-heading").forEach((node) => {
    const section = sections().find((item) => item.id === node.dataset.sectionHeading);
    node.hidden = enabled && !DATA.bars.some((bar) => Number(bar.bar) >= Number(section?.start_bar) && Number(bar.bar) <= Number(section?.end_bar) && barNeedsReview(bar));
  });
  byId("visibleBarCount").textContent = enabled ? `待复核 ${gridCountLabel(navigableBars())}` : gridCountLabel(DATA.bars);
  byId("emptyScore").hidden = shown > 0;
  byId("emptyScore").textContent = DATA.bars.length ? "当前筛选范围内没有待复核片段。" : "当前分析尚无可用小节。可在文件面板导入完整分析。";
  byId("filterStatus").textContent = enabled ? "依据证据分值筛选，非错误判定" : "";
  if (reselect && enabled && shown > 0 && cachedBarNodes.get(selectedBarNumber())?.hidden) {
    const bar = navigableBars()[0];
    const index = events.findIndex((event) => Number(event.bar) === Number(bar.bar));
    if (index >= 0) selectEvent(index);
    else if (bar.beats?.length) selectBeat(bar.bar, bar.beats[0].beat);
  }
}

function renderStructure() {
  const rows = sections();
  byId("structureSummary").textContent = rows.length ? `${rows.length} 个段落` : "小节与和弦证据";
  byId("sectionTimeline").innerHTML = rows.length ? rows.map((section) => `<button class="section-segment" data-section="${escapedAttr(section.id)}" style="left:${100 * Number(section.start) / Math.max(playbackDuration, 1)}%;width:${100 * (Number(section.end) - Number(section.start)) / Math.max(playbackDuration, 1)}%;--section-hue:${sectionHue(section)}" title="${escapedAttr(sectionTitle(section))} · ${escapedAttr(sectionRangeLabel(section))}">${escapedAttr(sectionTitle(section))}</button>`).join("") : '<span class="section-segment" style="left:0;width:100%">全曲 · 尚无段落分析</span>';
  byId("sectionList").innerHTML = rows.length ? rows.map((section) => `<button class="section-nav" data-section="${escapedAttr(section.id)}" title="${escapedAttr(sectionTitle(section))}"><span class="section-letter" style="--section-hue:${sectionHue(section)}">${escapedAttr(section.label || section.group || "·")}</span><span><strong>${escapedAttr(sectionTitle(section))}</strong><small>${escapedAttr(sectionRangeLabel(section))} · ${fmt(section.start)}</small></span></button>`).join("") : '<p class="hint-empty">尚无段落分析。可在右侧「段落」中新增人工标注。</p>';
  const loops = Array.isArray(DATA.structure?.loops) ? DATA.structure.loops : [];
  byId("loopList").innerHTML = loops.length ? loops.slice(0, 12).map((loop, index) => `<button class="loop-card" data-loop="${index}" title="试听这个循环"><strong>${Number(loop.period_bars || 0)} 小节循环 × ${Number(loop.repetitions || 0)}</strong><span>${escapedAttr(Array.isArray(loop.chords) ? loop.chords.join(" · ") : loop.chords || "重复和声")}</span><small>${escapedAttr(loop.start_bar)}–${escapedAttr(loop.end_bar)} 小节 · 证据 ${Number(loop.confidence || 0).toFixed(2)}</small></button>`).join("") : '<p class="hint-empty">暂无足够的循环证据</p>';
  const previous = selectedSectionId || byId("sectionSelect").value;
  byId("sectionSelect").innerHTML = rows.length ? rows.map((section) => `<option value="${escapedAttr(section.id)}">${escapedAttr(sectionTitle(section))} · ${escapedAttr(sectionRangeLabel(section))}</option>`).join("") : '<option value="">尚无段落</option>';
  selectedSectionId = rows.some((section) => section.id === previous) ? previous : rows[0]?.id || "";
  byId("sectionSelect").value = selectedSectionId;
  fillSectionEditor();
  const quality = DATA.structure?.grid_quality;
  const warnings = DATA.structure?.warnings || [];
  const qualityDetails = [];
  if (quality && typeof quality === "object") {
    if (Number.isFinite(Number(quality.confidence))) qualityDetails.push(`网格证据 ${Number(quality.confidence).toFixed(2)}（非正确概率）`);
    if (Number.isFinite(Number(quality.relative_duration_deviation))) qualityDetails.push(`小节时长偏差 ${(Number(quality.relative_duration_deviation)*100).toFixed(1)}%`);
    if (Number.isFinite(Number(quality.gap_count))) qualityDetails.push(`缺失区间 ${quality.gap_count} 处`);
    if (quality.phase_verified === false) qualityDetails.push("强拍位置尚未经人工核对");
  }
  const comparison = DATA.structure?.expert_grid?.comparison;
  if (comparison?.available) {
    if (Number.isFinite(Number(comparison.median_nearest_beat_difference_ms))) qualityDetails.push(`独立模型的拍点偏差中位数 ${Number(comparison.median_nearest_beat_difference_ms).toFixed(0)}ms`);
    if (Number.isFinite(Number(comparison.downbeat_agreement))) qualityDetails.push(`独立模型的小节起点一致度 ${(Number(comparison.downbeat_agreement)*100).toFixed(1)}%`);
    qualityDetails.push("模型间一致性不等于人工真值");
  }
  byId("gridQuality").textContent = qualityDetails.length ? `${qualityDetails.join("；")}${warnings.length ? `。${warnings.join("；")}` : ""}` : "尚无网格诊断。可通过播放拍点与小节边界进行人工核对。";
  byId("structureNote").textContent = rows.some((section) => section.source === "allin1") ? "段落功能为专家模型建议，尚未经人工确认。S 编号仅用于定位；可在「段落」中修改角色后保存。" : "相同字母表示相似段落。主歌、副歌等功能可在段落标注中人工确认。";
}

function fillSectionEditor() {
  const section = sections().find((row) => row.id === selectedSectionId);
  byId("sectionName").value = section ? sectionTitle(section) : "";
  byId("sectionRole").value = section?.role || "unknown";
  const firstBarNumber = DATA.bars?.length ? Math.min(...DATA.bars.map((bar) => Number(bar.bar))) : 1;
  byId("sectionStartBar").min = byId("sectionEndBar").min = firstBarNumber;
  byId("sectionStartBar").value = section?.start_bar ?? selectedBarNumber() ?? firstBarNumber;
  byId("sectionEndBar").value = section?.end_bar ?? selectedBarNumber() ?? firstBarNumber;
  byId("saveSection").disabled = !section;
  byId("deleteSection").disabled = !section || sections().length < 2;
  byId("deleteSection").textContent = sections().findIndex((row) => row.id === section?.id) > 0 ? "并入上一段" : "并入下一段";
  byId("loopSection").disabled = !section;
  const roleCandidates = (section?.role_candidates || []).filter((candidate) => candidate.score != null && Number.isFinite(Number(candidate.score))).slice(0,2).map((candidate) => `${SECTION_ROLES[candidate.role] || candidate.role} ${Number(candidate.score).toFixed(2)}`).join(" / ");
  const missingBoundary = section?.confidence == null || !Number.isFinite(Number(section.confidence)) || (Object.hasOwn(section?.evidence || {},"boundary_activation") && section.evidence.boundary_activation == null);
  const boundaryEvidence = missingBoundary ? "边界证据未提供" : `边界证据 ${Number(section.confidence).toFixed(2)}`;
  byId("sectionEvidence").textContent = section ? section.source === "manual" ? `人工标注 · ${preciseTime(section.start)}–${preciseTime(section.end)}` : `${section.source === "allin1" ? "专家模型建议" : "自动分析建议"} · 段落组 ${section.group || section.repeat_group || section.label || "—"} · ${boundaryEvidence}。${roleCandidates ? `功能候选分值：${roleCandidates}。` : ""}分值未经正确率校准；请结合试听确认角色后保存。` : "自动分段提供重复与边界证据，歌曲功能可由人工确认。";
}

function selectSection(id, jump = true) {
  const section = sections().find((row) => row.id === id);
  if (!section) return;
  selectedSectionId = id;
  byId("sectionSelect").value = id;
  fillSectionEditor();
  if (jump) {
    const index = events.findIndex((event) => Number(event.bar) === Number(section.start_bar));
    if (index >= 0) selectEvent(index);
    seekToAnalysis(section.start);
    revealBar(section.start_bar);
  }
}

function saveSection(create = false) {
  const start = barByNumber(Number(byId("sectionStartBar").value));
  const end = barByNumber(Number(byId("sectionEndBar").value));
  if (!start || !end || Number(end.end) <= Number(start.start)) { toast("请填写有效的开始与结束小节。"); return; }
  const rows = sections();
  const previous = create ? null : rows.find((section) => section.id === selectedSectionId);
  if (!create && !previous) return;
  const startTime = previous && Number(previous.start_bar) === Number(start.bar) ? Number(previous.start) : Number(start.start);
  const endTime = previous && Number(previous.end_bar) === Number(end.bar) ? Number(previous.end) : Number(end.end);
  if (endTime <= startTime) { toast("段落终点需要晚于起点。"); return; }
  if (rows.some((row,index) => index && Number(row.start) < Number(rows[index-1].end)-.001)) { toast("原段落时间存在重叠，请先导入连续、不重叠的结构标注。"); return; }
  if (!DATA.structure) DATA.structure = { version: 1, method: "manual", sections: [], loops: [], manual_overrides: [] };
  const original = previous || {};
  const nextId = original.id || newSectionId();
  const next = sectionWithTimes({ ...original, id: nextId, label: original.label || `M${rows.filter((row) => String(row.label || "").startsWith("M")).length+1}`, group: original.group || original.repeat_group || nextId, name: byId("sectionName").value.trim() || "人工段落", role: byId("sectionRole").value, source:"manual", edited:true,confirmed:true },startTime,endTime);
  let updated;
  if (create) {
    updated = [];
    for (const row of rows) {
      if (Number(row.end) <= startTime || Number(row.start) >= endTime) { updated.push(row); continue; }
      if (Number(row.start) < startTime-.001) updated.push(sectionWithTimes(row,Number(row.start),startTime));
      if (Number(row.end) > endTime+.001) updated.push(sectionWithTimes({ ...row,id:Number(row.start) < startTime-.001 ? newSectionId() : row.id },endTime,Number(row.end)));
    }
    updated.push(next);
  } else {
    const index = rows.findIndex((row) => row.id === previous.id);
    const left = rows[index-1], right = rows[index+1];
    if ((!left && Math.abs(startTime-Number(previous.start)) > .001) || (!right && Math.abs(endTime-Number(previous.end)) > .001)) { toast("首段起点与尾段终点保持完整覆盖；可新增段落来重新划分范围。"); return; }
    if ((left && startTime <= Number(left.start)+.001) || (right && endTime >= Number(right.end)-.001)) { toast("边界不能越过整个相邻段落；可先并入相邻段，再重新划分。"); return; }
    updated = rows.slice();
    updated[index] = next;
    if (left && Math.abs(Number(left.end)-startTime) > .001) updated[index-1] = sectionWithTimes(left,Number(left.start),startTime);
    if (right && Math.abs(Number(right.start)-endTime) > .001) updated[index+1] = sectionWithTimes(right,endTime,Number(right.end));
  }
  commitManualSections(completeSectionCoverage(updated));
  selectedSectionId = next.id;
  renderStable();
  updateActive();
  autosaveDraft("section");
  toast(create ? "新段落已划分，原段落的左右剩余区间已保留" : "段落已保存，相邻边界保持连续");
}

function newSectionId() {
  return `manual-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
}

function commitManualSections(rows) {
  if (!DATA.structure.manual_overrides?.length && !DATA.structure.automatic_sections) {
    DATA.structure.automatic_sections = cloneData(DATA.structure.sections || []);
    DATA.structure.automatic_boundaries = cloneData(DATA.structure.boundaries || []);
  }
  const previous = DATA.structure.boundaries || [];
  DATA.structure.sections = rows;
  DATA.structure.manual_overrides = cloneData(rows);
  DATA.structure.boundaries = rows.slice(1).map((section) => {
    const original = previous.find((boundary) => Math.abs(Number(boundary.time)-Number(section.start)) < .001);
    return { ...(original || {}),time:Number(section.start),bar:Number(section.start_bar),source:"manual",edited:true,
      confidence:original?.confidence ?? null,evidence:{ ...(original?.evidence || {}),manual:true } };
  });
}

function sectionWithTimes(section, start, end) {
  const covered = DATA.bars.filter((bar) => Number(bar.end) > start+.0001 && Number(bar.start) < end-.0001);
  const changed = Number(section.start) !== start || Number(section.end) !== end;
  return { ...section,start,end,start_bar:covered[0]?.bar ?? section.start_bar,end_bar:covered[covered.length-1]?.bar ?? section.end_bar,
    ...(changed ? { source:"manual",edited:true,boundary_source:"manual",original_boundary:section.original_boundary || { start:section.start,end:section.end,source:section.source } } : {}) };
}

function completeSectionCoverage(rows) {
  const sorted = rows.slice().sort((a,b) => Number(a.start)-Number(b.start));
  const output = [];
  let cursor = Math.min(Number(DATA.bars[0]?.start || 0),Number(sorted[0]?.start ?? Infinity));
  const end = Math.max(Number(DATA.bars[DATA.bars.length-1]?.end || playbackDuration),Number(sorted[sorted.length-1]?.end || 0));
  const unknown = (start,finish) => sectionWithTimes({ id:newSectionId(),label:"?",group:"unassigned",name:"未分段",role:"unknown",source:"manual",confirmed:false,edited:true },start,finish);
  for (const row of sorted) {
    if (Number(row.start) > cursor+.001) output.push(unknown(cursor,Number(row.start)));
    output.push(row); cursor=Number(row.end);
  }
  if (end > cursor+.001) output.push(unknown(cursor,end));
  return output;
}

function deleteSection() {
  if (!selectedSectionId || !DATA.structure) return;
  const rows = sections();
  const index = rows.findIndex((section) => section.id === selectedSectionId);
  if (index < 0 || rows.length < 2) return;
  const selected = rows[index];
  const target = rows[index > 0 ? index-1 : index+1];
  const combined = sectionWithTimes(target,Math.min(Number(selected.start),Number(target.start)),Math.max(Number(selected.end),Number(target.end)));
  commitManualSections(completeSectionCoverage(rows.filter((section) => section.id !== selected.id).map((section) => section.id === target.id ? combined : section)));
  selectedSectionId = target.id;
  renderStable();
  updateActive();
  autosaveDraft("section-delete");
  toast(`已并入「${sectionTitle(target)}」，保留该段名称与功能；可撤销恢复`);
}

function setLoop(a, b, play = false) {
  const start = clamp(Number(a), 0, playbackDuration);
  const end = clamp(Number(b), 0, playbackDuration);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end - start < 0.08) { toast("循环终点需要晚于起点。"); return false; }
  loopRange = { a: start, b: end, enabled: true };
  auditionEnd = null;
  renderLoop();
  if (play) seekToAnalysis(start, true);
  return true;
}

function renderLoop() {
  const valid = loopRange.a != null && loopRange.b != null && loopRange.b > loopRange.a;
  byId("loopAValue").textContent = loopRange.a == null ? "—" : preciseTime(loopRange.a);
  byId("loopBValue").textContent = loopRange.b == null ? "—" : preciseTime(loopRange.b);
  byId("loopToggle").textContent = loopRange.enabled && valid ? "↻ 循环开启" : "↻ 循环关闭";
  byId("loopToggle").setAttribute("aria-pressed", String(loopRange.enabled && valid));
  const overlay = byId("loopOverlay");
  overlay.hidden = !valid;
  if (valid) { overlay.style.left = `${100 * loopRange.a / playbackDuration}%`; overlay.style.width = `${100 * (loopRange.b - loopRange.a) / playbackDuration}%`; overlay.style.opacity = loopRange.enabled ? "1" : "0.4"; }
}

function toggleLoop() {
  if (loopRange.a == null || loopRange.b == null || loopRange.b <= loopRange.a) {
    const item = selectedCellItem();
    if (item) setLoop(item.start, item.end);
  } else { loopRange.enabled = !loopRange.enabled; renderLoop(); }
}

function setLoopPoint(which) {
  loopRange[which] = analysisTime();
  loopRange.enabled = loopRange.a != null && loopRange.b != null && loopRange.b > loopRange.a;
  renderLoop();
}

function drawOverview() {
  const canvas = byId("overviewCanvas");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const css = getComputedStyle(document.documentElement);
  const active = css.getPropertyValue("--active").trim();
  const muted = css.getPropertyValue("--line").trim();
  const accent = css.getPropertyValue("--accent").trim();
  ctx.strokeStyle = muted; ctx.beginPath(); ctx.moveTo(0,height/2); ctx.lineTo(width,height/2); ctx.stroke();
  if (waveformPeaks) {
    ctx.fillStyle = active; ctx.globalAlpha = .5;
    const step = width / waveformPeaks.length;
    waveformPeaks.forEach((peak,index) => { const h = Math.max(1, peak * (height - 14)); ctx.fillRect(index * step, (height - h)/2, Math.max(1,step - .6), h); });
    ctx.globalAlpha = 1;
  } else {
    for (const event of events) {
      const x = Number(event.start) / Math.max(1, playbackDuration) * width;
      const w = Math.max(1, (Number(event.end) - Number(event.start)) / Math.max(1, playbackDuration) * width - 1);
      const score = confidenceScore(event);
      const h = score == null ? 13 : 8 + clamp(score, 0, 1) * (height - 22);
      ctx.fillStyle = event.edited ? "#5881ab" : score == null ? muted : score < .52 ? accent : active;
      ctx.globalAlpha = .36; ctx.fillRect(x,(height-h)/2,w,h);
    }
    ctx.globalAlpha = 1;
  }
  for (const bar of DATA.bars) { const x = Number(bar.start) / Math.max(1,playbackDuration) * width; ctx.fillStyle = muted; ctx.fillRect(x, height - 7, 1, 7); }
  byId("overviewHint").textContent = waveformPeaks ? "音频波形 · 拖动选择循环" : events.every((event) => confidenceScore(event) == null) ? "和弦时间总览 · 模型未提供证据分值" : "和弦证据总览 · 拖动选择循环";
}

function refreshEmbeddedWaveform() {
  if (localAudioUrl || embeddedWaveformReference === DATA.audio_waveform) return;
  const envelope = DATA.audio_waveform;
  embeddedWaveformReference = envelope;
  const valid = envelope?.schema === "peak_envelope_v1" && Array.isArray(envelope.peaks) && envelope.peaks.length > 0 && envelope.peaks.length <= 16384
    && Number.isFinite(Number(envelope.duration)) && Math.abs(Number(envelope.duration)-playbackDuration) <= .05
    && Number.isFinite(Number(envelope.audio_offset)) && Math.abs(Number(envelope.audio_offset)-playbackStart) <= .005
    && envelope.peaks.every((peak) => typeof peak === "number" && Number.isFinite(peak) && peak >= 0 && peak <= 1);
  waveformPeaks = valid ? Float32Array.from(envelope.peaks) : null;
}

async function loadWaveform(file) {
  const generation = ++waveformGeneration;
  waveformPeaks = null; drawOverview();
  if (file.size > 120 * 1024 * 1024) { waveformPeaks = null; drawOverview(); toast("大文件已连接，保留和弦证据总览以减少内存占用。"); return; }
  let context;
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    context = new AudioContextClass();
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    if (generation !== waveformGeneration) return;
    const channels = Array.from({length:buffer.numberOfChannels},(_,index)=>buffer.getChannelData(index));
    const samples = channels[0];
    const count = Math.min(1800, Math.max(300, Math.floor(byId("overviewCanvas").clientWidth * 1.5)));
    const peaks = new Float32Array(count);
    const start = Math.max(0, Math.floor(playbackStart * buffer.sampleRate));
    const end = Math.min(samples.length, Math.ceil(playbackEnd * buffer.sampleRate));
    const span = (end - start) / count;
    for (let i = 0; i < count; i++) {
      let peak = 0; const left = Math.floor(start + i * span); const right = Math.min(end, Math.ceil(left + span));
      const stride = Math.max(1, Math.floor((right-left)/250));
      for (let j = left; j < right; j += stride) for (const channel of channels) peak = Math.max(peak, Math.abs(channel[j]));
      peaks[i] = peak;
    }
    const maximum = peaks.reduce((value,peak)=>Math.max(value,peak),0);
    if (maximum > 0) for (let i=0;i<peaks.length;i++) peaks[i] /= maximum;
    waveformPeaks = peaks; drawOverview();
  } catch (error) { waveformPeaks = null; drawOverview(); }
  finally { if (context) await context.close(); }
}

function updateWorkbenchPlayback(time, item, beat, barNode, eventIndex) {
  const ratio = playbackDuration > 0 ? clamp(time / playbackDuration, 0, 1) : 0;
  byId("overviewPlayhead").style.left = `${ratio * 100}%`;
  byId("overviewTrack").setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
  byId("overviewTrack").setAttribute("aria-valuetext", `${preciseTime(time)} / ${fmt(playbackDuration)}`);
  const section = sectionAt(time);
  if (currentSectionId !== (section?.id || "")) {
    currentSectionId = section?.id || "";
    document.querySelectorAll("[data-section]").forEach((node) => { const active = node.dataset.section === currentSectionId; node.classList.toggle("current", active); if (active) node.setAttribute("aria-current", "true"); else node.removeAttribute("aria-current"); });
  }
  const activeGridBar = DATA.bars.find((bar) => time >= Number(bar.start) && time < Number(bar.end));
  const position = activeGridBar?.grid_status === "uncovered" ? "拍点未定位" : activeGridBar?.grid_status === "incomplete" ? `第 ${activeGridBar.bar} 小节 · 拍点待核对` : beat ? `第 ${beat.bar} 小节 · ${beat.beat} 拍` : item ? `第 ${item.bar} 小节` : "当前和弦";
  byId("currentPosition").textContent = position;
  byId("meter").textContent = activeGridBar?.meter || DATA.meter || "未确定";
  byId("meter").title = DATA.meter_mode === "variable" ? "当前小节的拍号；此曲包含拍号变化" : "当前拍号";
  const localKey = keyAtTime(time);
  byId("key").textContent = localKey ? `${localKey} ${localKey.endsWith("m") ? "minor" : "major"}` : "未确定";
  byId("key").title = "当前播放位置的局部调性";
  const localKeyPresent = Boolean(activeGridBar?.local_key) || (Array.isArray(DATA.modulations) && DATA.modulations.length > 0) || (Array.isArray(DATA.key_timeline?.segments) && DATA.key_timeline.segments.length > 0);
  byId("keyCaption").textContent = localKeyPresent ? "局部调性" : DATA.key_estimation ? "调性估计" : "调性";
  byId("currentSecondary").textContent = item ? displayMode === "name" ? (item.roman_display || item.roman || "") : (item.chord_display || item.chord || "") : "";
  const pulseKey = beat ? `${beat.bar}.${beat.beat}` : "";
  const pulseCount = beatCountForBar(activeGridBar);
  if (byId("beatPulse").children.length !== pulseCount) { byId("beatPulse").innerHTML = Array.from({length:pulseCount},()=>"<i></i>").join(""); lastPulseBeat=null; }
  if (pulseKey !== lastPulseBeat) { lastPulseBeat = pulseKey; [...byId("beatPulse").children].forEach((node,index) => node.classList.toggle("active", !!beat && index + 1 === Number(beat.beat))); }
  const spanNodes = cachedSpanNodes.get(eventIndex) || [];
  const spanNode = spanNodes.find((node) => time >= Number(node.dataset.start) && time < Number(node.dataset.end));
  if (lastActiveSpan !== spanNode) { if (lastActiveSpan) lastActiveSpan.classList.remove("active"); if (spanNode) spanNode.classList.add("active"); lastActiveSpan = spanNode; }
  if (barNode) {
    const bar = barByNumber(Number(barNode.dataset.bar));
    if (bar) barNode.style.setProperty("--playhead", `${100 * barFraction(bar, time)}%`);
    const followTarget = barNode.classList.contains("uncovered-long") && spanNode ? spanNode : barNode;
    const followKey = followTarget === spanNode ? `${barNode.dataset.bar}:${eventIndex}` : Number(barNode.dataset.bar);
    if (!audio.paused && followEnabled && !document.body.classList.contains("inspector-open") && Date.now() > followSuspendedUntil && lastFollowedBar !== followKey && !barNode.hidden) { lastFollowedBar = followKey; revealNode(followTarget, false); }
  }
}

function adjustSelectedBoundary() {
  if (!canEditTimeline()) return;
  if (selectedBeatTarget) { toast("请先选中上方和弦片段，再调整边界。"); return; }
  const target = events[selectedEventIndex]; const bar = barByNumber(target?.bar);
  if (!target || !bar) return;
  const start = Number(byId("eventStart").value), end = Number(byId("eventEnd").value);
  const spans = (bar.spans || []).map((span) => ({ ...span })).sort((a,b) => a.start-b.start);
  const index = spans.findIndex((span) => sameSpan(span,target));
  if (index < 0) return;
  const before = spans[index-1], after = spans[index+1];
  const min = before ? Number(before.start) + .02 : Number(bar.start);
  const max = after ? Number(after.end) - .02 : Number(bar.end);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end-start < .02 || start < min-.001 || end > max+.001 || (!before && Math.abs(start-Number(bar.start)) > .001) || (!after && Math.abs(end-Number(bar.end)) > .001)) { toast("边界需位于当前小节内，至少保留 20ms，且不能跨过相邻片段。小节首尾保持不变。", 6000); return; }
  spans[index].start = start; spans[index].end = end; spans[index].edited = true;
  if (before) { before.end = start; before.edited = true; }
  if (after) { after.start = end; after.edited = true; }
  for (const span of spans) { span.beat_start = Number(beatPositionAtTime(bar,span.start).toFixed(3)); span.beat_end = Number(beatPositionAtTime(bar,span.end-.0001).toFixed(3)); span.position = sixteenthLabel(bar,span.start); span.unit = "manual"; }
  replaceBarSpans(bar,spans);
  selectedEventIndex = events.findIndex((event) => Number(event.bar) === Number(bar.bar) && Math.abs(Number(event.start)-start)<.001);
  renderStable(); updateActive(); autosaveDraft("boundary"); toast("边界已调整，相邻片段保持连续");
}

function canAdjustSelectedBoundary() {
  const item = events[selectedEventIndex];
  const bar = barByNumber(item?.bar);
  return !selectedBeatTarget && !!item && !!bar && !["uncovered","incomplete"].includes(bar.grid_status) && Number(item.start) >= Number(bar.start)-.0001 && Number(item.end) <= Number(bar.end)+.0001;
}

function replaceTimeRangeInBar(bar, start, end, parsed, edited = true, confidence = 1) {
  const spans = [];
  for (const span of bar.spans || []) {
    if (Number(span.end) <= start || Number(span.start) >= end) { spans.push({ ...span }); continue; }
    if (Number(span.start) < start - .001) spans.push({ ...span, end: start, beat_end: Number(beatPositionAtTime(bar,start-.0001).toFixed(3)) });
    if (Number(span.end) > end + .001) spans.push({ ...span, start: end, beat_start: Number(beatPositionAtTime(bar,end).toFixed(3)), position: sixteenthLabel(bar,end) });
  }
  spans.push(eventFromTimeRange(bar,start,end,parsed,confidence,edited,"beat"));
  replaceBarSpans(bar,spans);
}

function splitAtCurrentTime() {
  if (!canEditTimeline()) return;
  if (!canAdjustSelectedBoundary()) { toast("请先选中位于单个小节内的和弦片段。"); return; }
  const item = events[selectedEventIndex]; const bar = barByNumber(item?.bar); const time = analysisTime();
  if (!item || !bar || time < Number(item.start)+.02 || time > Number(item.end)-.02) { toast("将播放头移动到选中片段内部再切分。"); return; }
  const parsed = parsedFromItem(item);
  const parts = [eventFromTimeRange(bar,item.start,time,parsed,1,true,"manual"),eventFromTimeRange(bar,time,item.end,parsed,1,true,"manual")];
  replaceBarSpans(bar,[...(bar.spans || []).filter((span) => !sameSpan(span,item)),...parts]);
  selectedEventIndex = events.findIndex((event) => sameSpan(event,parts[1]));
  selectedBeatTarget = null;
  renderStable(); updateActive(); autosaveDraft("split-playhead");
}

function isCompatibleAnalysis(value) {
  if (!value || !Array.isArray(value.bars) || !Array.isArray(value.events)) return false;
  if (!Number.isFinite(Number(value.duration)) || Math.abs(Number(value.duration)-playbackDuration) > .2) return false;
  const importedOffset = Number(value.audio_offset ?? 0);
  if (!Number.isFinite(importedOffset) || Math.abs(importedOffset-playbackStart) > .005) return false;
  if (value.audio_file && DATA.audio_file && String(value.audio_file) !== String(DATA.audio_file)) return false;
  const validItem = (item) => item && typeof item === "object" && Number.isFinite(Number(item.start)) && Number.isFinite(Number(item.end)) && Number(item.end) > Number(item.start) && Number(item.start) >= 0 && Number(item.end) <= playbackDuration+.1;
  if (!value.events.every(validItem)) return false;
  if (!value.bars.every((bar) => validItem(bar) && Number.isFinite(Number(bar.bar)) && Array.isArray(bar.beats) && Array.isArray(bar.spans) && bar.beats.every((beat) => validItem(beat) && Number.isFinite(Number(beat.beat))) && bar.spans.every(validItem))) return false;
  if (value.structure?.sections && (!Array.isArray(value.structure.sections) || !value.structure.sections.every((section) => validItem(section) && Number.isFinite(Number(section.start_bar)) && Number.isFinite(Number(section.end_bar))))) return false;
  return true;
}

async function importCompleteAnalysis(file) {
  if (!file) return;
  try {
    const incoming = JSON.parse(await file.text());
    if (!isCompatibleAnalysis(incoming)) throw new Error("文件必须是同一音频、时长与偏移一致的完整 analysis JSON。");
    for (const key of Object.keys(DATA)) delete DATA[key];
    Object.assign(DATA,cloneData(incoming));
    dirtyEvents.clear(); selectedBeatTarget = null; selectedEventIndex = 0;
    refreshDerived(); renderStable(); updateActive(); autosaveDraft("import");
    byId("importStatus").textContent = `已导入 ${file.name}${timelineAudit.conflicted ? " · 检测到时间轴冲突，请审阅提示" : ""}`;
    toast(timelineAudit.conflicted ? "文件已加载，但时间轴存在冲突。请先审阅谱面上的冲突提示。" : "完整标注已导入，可撤销恢复",5000);
  } catch (error) { byId("importStatus").textContent = error.message || "导入失败"; toast(error.message || "导入失败",6000); }
  byId("importAnalysis").value = "";
}

function handleWorkbenchShortcut(event) {
  if (event.key === "Escape" && document.body.classList.contains("inspector-open")) { event.preventDefault(); setMobileInspector(false); return; }
  if (event.key === "Tab" && document.body.classList.contains("inspector-open")) {
    const focusable = [...document.querySelector(".inspector").querySelectorAll('button:not(:disabled),input:not(:disabled),textarea:not(:disabled),select:not(:disabled),summary,[tabindex="0"]')].filter((node) => node.offsetParent !== null);
    const first = focusable[0], last = focusable[focusable.length-1];
    if (event.shiftKey && event.target === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && event.target === last) { event.preventDefault(); first?.focus(); }
    return;
  }
  if (event.key === "Escape" && isTextEditingTarget(event.target)) { event.preventDefault(); event.target.blur(); barsEl.focus({ preventScroll: true }); return; }
  if (isTextEditingTarget(event.target) || event.altKey || byId("shortcutDialog").open || byId("seekDialog").open || byId("keyCandidatesDialog").open) return;
  const key = String(event.key).toLowerCase();
  if ((event.ctrlKey || event.metaKey) && (key === "z" || key === "y")) { event.preventDefault(); moveHistory(key === "y" || event.shiftKey ? 1 : -1); return; }
  if (event.ctrlKey || event.metaKey) return;
  if ((event.code === "Space" || event.key === "Enter") && event.target.closest("button,a,summary")) return;
  if (event.code === "Space") { event.preventDefault(); playBtn.click(); }
  else if (event.key === "Enter" && !event.target.closest("button, a, summary")) { event.preventDefault(); editSelectedInput(); }
  else if (key === "[") { event.preventDefault(); setLoopPoint("a"); }
  else if (key === "]") { event.preventDefault(); setLoopPoint("b"); }
  else if (key === "l") { event.preventDefault(); toggleLoop(); }
  else if (key === "g") { event.preventDefault(); openSeekDialog(); }
  else if (key === "?") { event.preventDefault(); byId("shortcutDialog").showModal(); }
}

function initializeWorkbench() {
  try { document.documentElement.dataset.theme = localStorage.getItem("chord-studio-theme") || "light"; } catch (error) {}
  byId("themeToggle").addEventListener("click", () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; try { localStorage.setItem("chord-studio-theme",theme); } catch (error) {} drawOverview(); });
  byId("undoEdit").addEventListener("click",() => moveHistory(-1)); byId("redoEdit").addEventListener("click",() => moveHistory(1));
  byId("showInspector").addEventListener("click",() => setMobileInspector(true)); byId("dismissInspector").addEventListener("click",() => setMobileInspector(false)); byId("inspectorBackdrop").addEventListener("click",() => setMobileInspector(false));
  window.addEventListener("resize",() => { if (window.innerWidth > 700 && document.body.classList.contains("inspector-open")) setMobileInspector(false); });
  byId("playbackRate").addEventListener("change",() => { audio.playbackRate = Number(byId("playbackRate").value); audio.preservesPitch = true; });
  byId("previousBar").addEventListener("click",() => jumpBar(-1)); byId("nextBar").addEventListener("click",() => jumpBar(1)); byId("jumpBarButton").addEventListener("click",() => jumpBar());
  byId("jumpBar").addEventListener("keydown",(event) => { if (event.key === "Enter") { event.preventDefault(); jumpBar(); } });
  byId("densitySelect").addEventListener("change",() => { barsEl.dataset.density = byId("densitySelect").value; });
  byId("showBeats").addEventListener("change",() => barsEl.classList.toggle("hide-beats",!byId("showBeats").checked));
  byId("reviewOnly").addEventListener("change",() => applyReviewFilter(true));
  byId("followPlayback").addEventListener("click",() => { followEnabled = !followEnabled; followSuspendedUntil = 0; lastFollowedBar = null; byId("followPlayback").setAttribute("aria-pressed",String(followEnabled)); if (followEnabled) revealBar(beatFor(analysisTime())?.bar, false); });
  window.addEventListener("wheel",() => { followSuspendedUntil = Date.now()+5000; },{ passive:true }); window.addEventListener("touchmove",() => { followSuspendedUntil = Date.now()+5000; },{ passive:true });
  document.querySelectorAll("[data-panel]").forEach((button) => { button.tabIndex = button.getAttribute("aria-selected") === "true" ? 0 : -1; button.addEventListener("click",() => activateInspector(button.dataset.panel)); button.addEventListener("keydown",(event) => { if (!["ArrowLeft","ArrowRight"].includes(event.key)) return; event.preventDefault(); event.stopPropagation(); const tabs = [...document.querySelectorAll("[data-panel]")]; const next = tabs[(tabs.indexOf(button)+(event.key === "ArrowRight" ? 1 : tabs.length-1))%tabs.length]; activateInspector(next.dataset.panel); next.focus(); }); });
  barsEl.addEventListener("dblclick",(event) => { if (event.target.closest(".bar")) editSelectedInput(); });
  [eventList,suggestionList,pathProposalList].forEach((container) => container.addEventListener("keydown",(event) => { if (event.target.closest("button,input,select,textarea")) return; if (event.key === "Enter" || event.code === "Space") { const node = event.target.closest('[role="button"]'); if (node) { event.preventDefault(); event.stopPropagation(); node.click(); } } }));
  byId("editAtPlayhead").addEventListener("click",() => { const item = activeFor(analysisTime()); if (item) { selectEventByObject(item); revealBar(item.bar); } });
  byId("auditionSelection").addEventListener("click",() => { const item = selectedCellItem(); if (!item) return; loopRange.enabled = false; renderLoop(); seekToAnalysis(item.start,true); auditionEnd = Number(item.end); });
  byId("applyBoundary").addEventListener("click",adjustSelectedBoundary); byId("splitAtPlayhead").addEventListener("click",splitAtCurrentTime);
  byId("rebuildEventTimeline").addEventListener("click",rebuildEventTimeline);
  byId("loopToggle").addEventListener("click",toggleLoop); byId("setLoopA").addEventListener("click",() => setLoopPoint("a")); byId("setLoopB").addEventListener("click",() => setLoopPoint("b"));
  byId("loopSelection").addEventListener("click",() => { const item = selectedCellItem(); if (item) setLoop(item.start,item.end,true); });
  byId("clearLoop").addEventListener("click",() => { loopRange = { a:null,b:null,enabled:false }; renderLoop(); });
  [byId("sectionTimeline"),byId("sectionList")].forEach((container) => container.addEventListener("click",(event) => { const node = event.target.closest("[data-section]"); if (node) selectSection(node.dataset.section); }));
  byId("loopList").addEventListener("click",(event) => { const node = event.target.closest("[data-loop]"); const loop = node ? DATA.structure?.loops?.[Number(node.dataset.loop)] : null; if (loop) { setLoop(loop.start,loop.end,true); revealBar(loop.start_bar); } });
  byId("sectionSelect").addEventListener("change",() => selectSection(byId("sectionSelect").value,false));
  byId("saveSection").addEventListener("click",() => saveSection(false)); byId("addSection").addEventListener("click",() => saveSection(true)); byId("deleteSection").addEventListener("click",deleteSection);
  byId("loopSection").addEventListener("click",() => { const section = sections().find((row) => row.id === selectedSectionId); if (section) setLoop(section.start,section.end,true); });
  byId("importAnalysis").addEventListener("change",() => importCompleteAnalysis(byId("importAnalysis").files[0]));
  byId("exportSections").addEventListener("click",() => { const quote = (value) => `"${String(value ?? "").replace(/"/g,'""')}"`; const fields = ["id","name","label","role","start","end","start_bar","end_bar","source"]; const lines = [fields.join(","),...sections().map((section) => fields.map((field) => quote(section[field])).join(","))]; downloadText(`${baseExportName()}_sections.csv`,"\uFEFF"+lines.join("\n"),"text/csv;charset=utf-8"); });
  byId("shortcutHelp").addEventListener("click",() => byId("shortcutDialog").showModal()); byId("closeShortcutHelp").addEventListener("click",() => byId("shortcutDialog").close());
  currentTimeEl.addEventListener("click",openSeekDialog); byId("closeSeekDialog").addEventListener("click",() => byId("seekDialog").close());
  byId("keyCandidatesButton").addEventListener("click",openKeyCandidates); byId("closeKeyCandidates").addEventListener("click",() => byId("keyCandidatesDialog").close());
  byId("printScore").addEventListener("click",() => { preparePrint(); window.print(); });
  window.addEventListener("beforeprint",preparePrint);
  byId("applySeekTime").addEventListener("click",() => applyPreciseSeek()); byId("auditionSeekTime").addEventListener("click",() => applyPreciseSeek(true));
  byId("seekTimeInput").addEventListener("keydown",(event) => { if (event.key === "Enter") { event.preventDefault(); applyPreciseSeek(); } });
  byId("localAudioFile").addEventListener("change",() => { const file = byId("localAudioFile").files[0]; if (!file) return; audio.pause(); if (localAudioUrl) URL.revokeObjectURL(localAudioUrl); localAudioUrl = URL.createObjectURL(file); audio.src = localAudioUrl; byId("sessionStatus").textContent = `已连接 ${file.name}`; loadWaveform(file); });
  audio.addEventListener("seeked",updateActive); audio.addEventListener("ratechange",updateActive); audio.addEventListener("ended",updateActive);
  audio.addEventListener("play",() => { playBtn.setAttribute("aria-label","暂停"); clearInterval(playbackGuardTimer); playbackGuardTimer = setInterval(enforcePlaybackRange,25); });
  audio.addEventListener("pause",() => { playBtn.setAttribute("aria-label","播放"); clearInterval(playbackGuardTimer); playbackGuardTimer = null; });
  audio.addEventListener("ended",() => { if (loopRange.enabled && loopRange.a != null && loopRange.b != null) seekToAnalysis(loopRange.a,true); });
  audio.addEventListener("loadedmetadata",() => { if (Number.isFinite(audio.duration) && audio.duration+.5 < playbackEnd) toast("音频时长短于分析时间轴，请确认选择的是对应版本。",7000); });
  const track = byId("overviewTrack"); let pointer = null;
  const pointerTime = (event) => { const rect = track.getBoundingClientRect(); return clamp((event.clientX-rect.left)/rect.width,0,1)*playbackDuration; };
  track.addEventListener("pointerdown",(event) => { if (event.button !== 0) return; pointer = { start:pointerTime(event),x:event.clientX,moved:false }; track.setPointerCapture(event.pointerId); track.focus({ preventScroll:true }); });
  track.addEventListener("pointermove",(event) => { if (!pointer) return; if (Math.abs(event.clientX-pointer.x)>4) pointer.moved = true; if (pointer.moved) { const time = pointerTime(event); loopRange = { a:Math.min(pointer.start,time), b:Math.max(pointer.start,time), enabled:false }; renderLoop(); } });
  track.addEventListener("pointerup",(event) => { if (!pointer) return; const time = pointerTime(event); if (pointer.moved) setLoop(Math.min(pointer.start,time),Math.max(pointer.start,time)); else { seekToAnalysis(time); const item = activeFor(time); if (item) { selectEventByObject(item); revealBar(item.bar); } } pointer = null; });
  track.addEventListener("pointercancel",() => { pointer = null; });
  track.addEventListener("keydown",(event) => { if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); event.stopPropagation(); seekToAnalysis(analysisTime()+(event.key === "ArrowRight" ? 1 : -1)*(event.shiftKey ? 5 : .25)); } else if (event.key === "Home" || event.key === "End") { event.preventDefault(); seekToAnalysis(event.key === "Home" ? 0 : playbackDuration); } });
  if (typeof ResizeObserver !== "undefined") { new ResizeObserver(drawOverview).observe(byId("overviewCanvas")); new ResizeObserver(() => document.documentElement.style.setProperty("--header-height",`${document.querySelector(".topbar").offsetHeight}px`)).observe(document.querySelector(".topbar")); }
  window.addEventListener("beforeunload",() => { if (localAudioUrl) URL.revokeObjectURL(localAudioUrl); });
  window.addEventListener("hashchange",openLinkedTimeRange);
}

initializeWorkbench();
restoreAutosaveDraft();
refreshDerived();
render();
selectEvent(0);
recordHistory("initial");
updateActive();
openLinkedTimeRange();
