"use strict";
/* Unified pipeline: analyse -> separate -> optional sub-splits -> harmonic fusion. */
(function () {
  const panel = document.getElementById("pipelinePanel");
  const startBtn = document.getElementById("pipeStart");
  const modelSelect = document.getElementById("pipeModel");
  if (!panel || !startBtn) return;
  let configLoaded = false;
  let running = false;

  async function ensureModels() {
    if (configLoaded) return;
    try {
      const config = await api("/api/config");
      const models = config.stems_models || [];
      modelSelect.replaceChildren();
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = `${model.label}（${model.stems} 轨）`;
        option.title = model.note || "";
        if (model.id === "bs_roformer_sw") option.selected = true;
        modelSelect.appendChild(option);
      }
      configLoaded = true;
    } catch (error) {
      toast(error.message);
    }
  }

  function openPanel() {
    panel.hidden = !panel.hidden;
    if (!panel.hidden) ensureModels();
  }

  async function ensureProject() {
    if (state.project) return state.project.id;
    if (!state.file) throw new Error("请先选择音乐，或打开一个已有项目");
    $("inputStatus").textContent = "正在上传音频…";
    const form = new FormData();
    form.append("file", state.file);
    state.project = await api("/api/upload", { method: "POST", body: form });
    state.file = null;
    return state.project.id;
  }

  async function start() {
    if (running) return;
    running = true;
    startBtn.disabled = true;
    const original = startBtn.textContent;
    startBtn.textContent = "正在运行…";
    try {
      const id = await ensureProject();
      const body = {
        model: modelSelect.value || "bs_roformer_sw",
        harmony: document.getElementById("pipeHarmony").checked,
        harmony_method: "stem_evidence_v2",
        vocal_parts: document.getElementById("pipeVocal").checked,
        guitar_parts: document.getElementById("pipeGuitar").checked,
        drums_parts: document.getElementById("pipeDrums").checked,
        guitar_mode: document.getElementById("pipeGuitarMode")?.value || "timbre",
        harmony_default: Boolean(document.getElementById("pipeHarmonyDefault")?.checked),
        force_stems: false,
      };
      const job = await post(`/api/projects/${id}/pipeline`, body);
      if (job.profile !== "pipeline") throw new Error("当前项目有任务在运行，请完成后再开始完整流程。");
      $("inputStatus").textContent = "完整流程已提交：分析 → 分轨 → 细分 → 融合";
      toast("完整流程开始，约 3–6 分钟；可以继续浏览其他页面");
      if (!await waitForCompletion(job.id, state.navigation)) return;
      $("inputStatus").textContent = body.harmony
        ? (body.harmony_default ? "完成。默认时间轴已切到「分轨构成音和弦 · V2」。"
            : "完成。请在和弦来源中选择「分轨构成音和弦 · V2」与原结果比较。")
        : "完成。分轨结果已更新。";
    } catch (error) {
      $("inputStatus").textContent = error.message;
      toast(error.message);
    } finally {
      running = false;
      startBtn.disabled = false;
      startBtn.textContent = original;
    }
  }

  startBtn.addEventListener("click", start);
  document.getElementById("pipeOpenResult")?.addEventListener("click", () => {
    const setup = document.getElementById("beAnalysisSetup");
    if (setup) setup.open = true;
    panel.hidden = false;
    ensureModels();panel.scrollIntoView({block: "center"});
  });
  async function waitForCompletion(id, navigation) {
    watchJob(id, navigation);
    for (;;) {
      if (navigation !== state.navigation) return false;
      const job = await api(`/api/jobs/${id}`);
      if (job.status === "completed") return true;
      if (["failed", "interrupted"].includes(job.status)) throw new Error(job.error || "识别未完成");
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }
  document.getElementById("pipeChordsOnly")?.addEventListener("click", async event => {
    const button = event.currentTarget;
    if (running) return;
    running = true;button.disabled = true;startBtn.disabled = true;
    try {
      const id = await ensureProject();
      const job = await post(`/api/projects/${id}/fusion`, {model: modelSelect.value || "bs_roformer_sw"});
      if (job.profile !== "fusion") throw new Error("当前项目有任务在运行，请完成后再识别分轨和弦。");
      $("inputStatus").textContent = "正在识别分轨和弦；将复用已有分析和分轨。";
      if (!await waitForCompletion(job.id, state.navigation)) return;
      $("inputStatus").textContent = "完成。请在和弦来源中选择「分轨构成音和弦 · V2」试听比较。";
    } catch (error) { toast(error.message);$("inputStatus").textContent = error.message; }
    finally { running = false;button.disabled = false;startBtn.disabled = false; }
  });
  window.openPipelinePanel = openPanel;
})();
