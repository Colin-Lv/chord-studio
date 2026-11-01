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
        vocal_parts: document.getElementById("pipeVocal").checked,
        guitar_parts: document.getElementById("pipeGuitar").checked,
        drums_parts: document.getElementById("pipeDrums").checked,
        guitar_mode: document.getElementById("pipeGuitarMode")?.value || "timbre",
        harmony_default: Boolean(document.getElementById("pipeHarmonyDefault")?.checked),
        force_stems: false,
      };
      const job = await post(`/api/projects/${id}/pipeline`, body);
      $("inputStatus").textContent = "完整流程已提交：分析 → 分轨 → 细分 → 融合";
      toast("完整流程开始，约 3–6 分钟；可以继续浏览其他页面");
      await watchJob(job.id, state.navigation);
      $("inputStatus").textContent = body.harmony
        ? "完成。默认时间轴已切到「分轨融合（实验）」，可在来源里切回。"
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
  window.openPipelinePanel = openPanel;
})();
