"use strict";
/* Shared across workbench, stems and MIDI pages; cancellation is server-owned. */
(() => {
  const host = document.querySelector('header');
  if (!host) return;
  const button = document.createElement('button');
  button.className = 'tq-open'; button.type = 'button'; button.textContent = '任务队列';
  button.setAttribute('aria-haspopup', 'dialog'); host.append(button);
  const dialog = document.createElement('dialog'); dialog.className = 'tq-dialog';
  dialog.setAttribute('aria-labelledby', 'tq-title');
  dialog.innerHTML = '<div class="tq-heading"><div><h2 id="tq-title">任务队列</h2><p>所有歌曲的任务按提交顺序运行。</p></div><button type="button" class="tq-close" aria-label="关闭任务队列">×</button></div><p class="tq-message" role="status"></p><div class="tq-active"></div><details class="tq-history"><summary>最近结束的任务</summary><div class="tq-recent"></div></details><p class="tq-note">取消后，已经保存的分析与分轨结果仍可使用。运行中的任务停止并释放资源后，才会开始下一项。</p>';
  document.body.append(dialog);
  const active = dialog.querySelector('.tq-active'), recent = dialog.querySelector('.tq-recent');
  const message = dialog.querySelector('.tq-message');
  const labels = {queued:'排队中', running:'进行中', cancelling:'正在停止', cancelled:'已取消', completed:'已完成', failed:'失败', interrupted:'已中断'};
  const profiles = {retained10:'和声识别', v1:'和声识别', v2:'和声识别', madmom:'和声识别', stems:'音轨分离', pipeline:'完整流程', fusion:'分轨和弦识别', vocal_parts:'主唱／和声细分', guitar_parts:'吉他细分', drums_parts:'鼓组细分', piano:'钢琴扒谱', reconstruction:'整曲重构'};
  let timer, reading = false; const pending = new Set();
  async function api(url, method='GET') {
    const response = await fetch(url, {method, headers:{'X-Harmony-Client':'studio'}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '暂时无法读取任务，请重试');
    return data;
  }
  function el(tag, cls, text) {
    const n = document.createElement(tag); n.className = cls; n.textContent = text; return n;
  }
  function render(rows, container) {
    const focused = document.activeElement?.dataset?.cancel;
    container.replaceChildren(...rows.map(job => {
      const row = el('article', 'tq-row', '');
      const content = el('div', 'tq-content', '');
      content.append(el('strong','',job.project_title || '未命名歌曲'));
      content.append(el('p','tq-type',profiles[job.profile] || '分析任务'));
      content.append(el('p','tq-stage',job.status === 'queued' ? `等待顺序 · 第 ${job.queue_position} 项` : (job.stage_label || labels[job.status])));
      if (job.error) content.append(el('p','tq-error',job.error));
      const side = el('div','tq-side','');
      side.append(el('span',`tq-state tq-${job.status}`,labels[job.status] || job.status));
      if (['queued','running','cancelling'].includes(job.status)) {
        const cancel = el('button','tq-cancel',job.status === 'queued' ? '取消排队' : '停止任务');
        cancel.type = 'button'; cancel.dataset.cancel = job.id;
        cancel.disabled = job.status === 'cancelling' || pending.has(job.id);
        cancel.onclick = async () => {
          pending.add(job.id); cancel.disabled = true; message.textContent = '正在提交取消请求…';
          try {
            const result = await api(`/api/jobs/${job.id}/cancel`, 'POST');
            message.textContent = result.status === 'cancelled' ? '已取消排队。' : result.status === 'cancelling' ? '正在停止任务，已有结果会保留。' : '该任务已经结束。';
            window.dispatchEvent(new CustomEvent('chord-job-updated', {detail:result}));
          } catch (error) { message.textContent = error.message; }
          finally {pending.delete(job.id); await refresh();}
        };
        side.append(cancel);
      }
      row.append(content,side); return row;
    }));
    if (focused) container.querySelector(`[data-cancel="${focused}"]`)?.focus();
  }
  async function refresh() {
    clearTimeout(timer);
    if (reading) return;
    reading = true;
    try {
      const data = await api('/api/jobs');
      button.textContent = `任务队列${data.active.length ? ` · ${data.active.length}` : ''}`;
      if (dialog.open) {
        render(data.active,active); render(data.recent,recent);
        if (!data.active.length) active.append(el('p','tq-empty','没有进行中或排队的任务。'));
        if (!data.recent.length) recent.append(el('p','tq-empty','暂无任务记录。'));
      }
    } catch (error) {if (dialog.open) message.textContent = error.message;}
    finally {reading = false; timer = setTimeout(refresh, dialog.open ? 1500 : 8000);}
  }
  button.onclick = () => {dialog.showModal(); message.textContent = ''; refresh();};
  dialog.querySelector('.tq-close').onclick = () => dialog.close();
  dialog.addEventListener('click', e => {if (e.target === dialog) {const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  document.addEventListener('visibilitychange', () => {if (!document.hidden) refresh();});
  refresh();
})();
