"use strict";
/* Result identities and task progress are independent of the displayed model. */
const GuitarParts = (() => {
  let catalog = { results: [], modes: [] }, serial = 0, error = "";
  let job = null, polling = false, notice = "";
  const model = () => state.manifest?.model_key || "bs_roformer_sw";
  const mode = () => readPreference().guitarModes?.[model()] || "timbre";
  const identity = (m = model(), kind = mode()) => `${m}:${kind}`;
  const expanded = () => readPreference().guitarExpanded?.[identity()] ?? true;
  const saveExpanded = (value, m = model(), kind = mode()) => savePreference({
    guitarExpanded: { ...readPreference().guitarExpanded, [identity(m, kind)]: value }
  });
  const saveMode = (m, kind) => savePreference({ guitarModes: { ...readPreference().guitarModes, [m]: kind } });
  const label = kind => catalog.modes.find(r => r.id === kind)?.label || (kind === "duet" ? "旋律／伴奏吉他" : "木吉他／电吉他");
  const sleep = () => new Promise(resolve => setTimeout(resolve, 2000));

  async function refresh() {
    const request = ++serial;
    try {
      const data = await api(`/api/projects/${PROJECT}/stems/guitar-parts-catalog`);
      if (request === serial) { catalog = data; error = ""; }
    } catch (e) { if (request === serial) error = e.status === 404 ? "吉他细分服务尚未载入，请重新打开应用。" : e.message; }
  }
  function apply() {
    state.guitarParts = error ? null : catalog.results.find(r => r.model === model() && r.mode === mode()) || null;
    state.expandGuitars = !!state.guitarParts && expanded();
  }
  async function show(m, kind) {
    saveMode(m, kind); saveExpanded(true, m, kind);
    if (model() !== m) await switchModel(m);
    if (model() !== m) return;
    apply(); rebuildTracks();
    document.querySelector('.track-head[data-track="acoustic_guitar"], .track-head[data-track="guitar_one"]')?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  function render() {
    if (!state.manifest) return;
    const hasParent = state.manifest.tracks.some(t => t.id === "guitar");
    $("guitarSource").textContent = hasParent ? `来源：${modelLabel(model())} · 完整吉他轨` : "来源：六轨精细版的吉他轨";
    const select = $("guitarMode");select.replaceChildren();
    for (const row of catalog.modes) {
      const option = document.createElement("option"); option.value = row.id;
      option.textContent = row.label + (row.ready ? "" : "（模型未就绪）");option.disabled = !row.ready;
      select.appendChild(option);
    }
    select.value = mode(); select.disabled = !!error;
    select.onchange = () => { saveMode(model(), select.value); apply(); rebuildTracks(); };
    const recipe = catalog.modes.find(r => r.id === mode());
    $("guitarNote").textContent = (recipe?.note || "") + " 结果展开在下方音轨列表，点 S 独奏、点 ↓ 下载；收起恢复完整吉他。";
    $("guitarStatus").textContent = error ? `读取结果失败：${error} 已有文件不会被删除。`
      : !hasParent ? "当前版本没有独立吉他轨。可以切到六轨精细，或直接打开下方已保存结果。"
      : state.guitarParts ? `已保存 · ${label(mode())} · ${state.expandGuitars ? "已在下方展开" : "已收起"}`
      : "本方案暂无有效结果。不同细分方式分别保存，切换不会删除已有音轨。";
    $("guitarParent").hidden = hasParent;
    $("guitarParent").onclick = () => switchModel("bs_roformer_sw");
    const generate = $("guitarGenerate");generate.hidden = !hasParent;
    generate.disabled = !!job || !!error || !recipe?.ready;
    generate.textContent = job ? "吉他细分进行中" : state.guitarParts ? "重新生成本方案" : "生成吉他细分";
    generate.onclick = () => run(!!state.guitarParts);
    const expand = $("guitarExpand");expand.hidden = !state.guitarParts;
    expand.textContent = state.expandGuitars ? "收起吉他细分" : "展开结果";
    expand.onclick = () => { saveExpanded(!state.expandGuitars);apply();rebuildTracks(); };
    $("guitarJobStatus").hidden = !job && !notice;
    $("guitarJobStatus").textContent = job
      ? `${modelLabel(job.model)} · ${label(job.guitar_mode)} · ${job.stage_label || "正在处理"}。切换或刷新不影响任务。`
      : notice;
    const results = $("guitarSavedResults");results.replaceChildren();
    for (const result of catalog.results) {
      const button = document.createElement("button");button.className = "secondary";
      button.dataset.model = result.model;button.dataset.mode = result.mode;button.disabled = !!error;
      button.textContent = `查看结果 · ${modelLabel(result.model)} · ${label(result.mode)}`;
      button.onclick = () => show(result.model, result.mode);results.appendChild(button);
    }
  }
  async function run(force) {
    if (job) return;
    job = { model: model(), guitar_mode: mode(), stage_label: "正在提交任务" };notice = "";render();
    try {
      const row = await api(`/api/projects/${PROJECT}/guitar-parts`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: job.model, mode: job.guitar_mode, force }) });
      if (row.profile !== "guitar_parts") throw new Error("当前有其他任务在运行，请等它结束后再细分吉他。");
      job = row; savePreference({ guitarJobId: row.id });await watch();
    } catch (e) { job = null;notice = e.message;toast(e.message); }
    finally { render(); }
  }
  async function resume() {
    for (const id of [...new Set([catalog.active_job, state.project?.active_job, readPreference().guitarJobId].filter(Boolean))]) {
      try {
        const row = await api(`/api/jobs/${id}`);
        if (row.profile !== "guitar_parts") continue;
        job = row;savePreference({ guitarJobId: row.id });render();void watch();return;
      } catch (e) {
        if (e.status === 404) { savePreference({ guitarJobId: null });continue; }
        notice = `任务状态暂时无法恢复：${e.message}，请刷新重试。`;render();return;
      }
    }
  }
  async function watch() {
    if (polling) return;polling = true;
    try {
      while (job?.id) {
        let row;
        try { row = await api(`/api/jobs/${job.id}`); }
        catch (e) {
          if (e.status === 404) throw e;
          job.stage_label = "进度读取中断，正在重连";render();await sleep();continue;
        }
        job = row;render();
        if (["failed", "interrupted"].includes(row.status)) throw new Error(row.error || "吉他细分失败，请重试。");
        if (row.status === "completed") {
          await refresh();
          if (error) { job.stage_label = "任务完成，正在重试读取结果";render();await sleep();continue; }
          const result = catalog.results.find(r => r.model === row.model && r.mode === row.guitar_mode);
          if (!result) throw new Error("任务已结束，但结果不可用，源吉他或模型可能已更新。请重新生成。");
          saveExpanded(true, row.model, row.guitar_mode);
          notice = `${label(row.guitar_mode)}已保存，可点「查看结果」试听和下载。`;
          job = null;savePreference({ guitarJobId: null });apply();rebuildTracks();toast(notice);break;
        }
        await sleep();
      }
    } catch (e) { notice = e.message;job = null;savePreference({ guitarJobId: null });toast(e.message); }
    finally { polling = false;render(); }
  }
  return { refresh, apply, render, resume };
})();
