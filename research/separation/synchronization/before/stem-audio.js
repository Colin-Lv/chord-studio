"use strict";
// Complete bounded downloads before playing: long HTTP media streams otherwise
// occupy Chromium's per-origin connections and starve the remaining stems.
const StemAudio = (() => {
  let current = null;
  function dispose() {
    if (!current) return;
    const old = current;current = null;old.controller.abort();
    for (const node of old.nodes) { node.pause();node.removeAttribute("src");node.load(); }
    for (const url of old.urls) URL.revokeObjectURL(url);
  }
  function ready(node, signal) {
    if (node.readyState >= 2) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const finish = error => {
        clearTimeout(timer);node.removeEventListener("loadeddata", loaded);
        node.removeEventListener("error", failed);signal.removeEventListener("abort", aborted);
        error ? reject(error) : resolve();
      };
      const loaded = () => finish();
      const failed = () => finish(new Error("试听音频无法解码，请重新加载。"));
      const aborted = () => finish(new DOMException("Cancelled", "AbortError"));
      const timer = setTimeout(() => finish(new Error("试听音频准备超时，请重试。")), 30000);
      node.addEventListener("loadeddata", loaded);node.addEventListener("error", failed);
      signal.addEventListener("abort", aborted, { once: true });
      if (signal.aborted) aborted();
    });
  }
  function load(nodes) {
    if (current) return current.promise;
    const job = { nodes, urls: [], controller: new AbortController(), promise: null };
    current = job;let cursor = 0;
    async function worker() {
      while (cursor < nodes.length) {
        const node = nodes[cursor++];
        const response = await fetch(node.dataset.sourceUrl, { signal: job.controller.signal });
        if (!response.ok) throw new Error(`读取试听音频失败（${response.status}），请重试。`);
        const blob = await response.blob();
        if (job.controller.signal.aborted) throw new DOMException("Cancelled", "AbortError");
        const url = URL.createObjectURL(blob);job.urls.push(url);
        node.src = url;node.preload = "auto";node.load();
        await ready(node, job.controller.signal);
      }
    }
    job.promise = Promise.all(Array.from({ length: Math.min(3, nodes.length) }, worker)).catch(error => {
      if (current === job) dispose();
      throw error;
    });
    return job.promise;
  }
  window.addEventListener("pagehide", dispose);
  return { load, dispose };
})();
