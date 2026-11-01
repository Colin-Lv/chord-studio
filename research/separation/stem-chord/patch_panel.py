"""Add the unified pipeline panel to the workbench."""
from __future__ import annotations

from pathlib import Path

FRONTEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend")

html = FRONTEND / "index.html"
text = html.read_text(encoding="utf-8")

ANCHOR = '<p id="inputStatus" class="status" role="status">选择音频后可以立即试听。</p>'
PANEL = ANCHOR + '''
        <div id="pipelinePanel" class="pipeline-panel" hidden>
          <div class="pipeline-head"><strong>完整流程</strong><span class="muted">一次设置好：和声分析 → 音轨分离 → 细分 → 融合</span></div>
          <div class="pipeline-grid">
            <label class="pipe-field">分轨模型<select id="pipeModel"></select></label>
            <label class="pipe-check"><input type="checkbox" id="pipeHarmony" checked> 同时识别和声（分轨融合）</label>
            <label class="pipe-check"><input type="checkbox" id="pipeVocal"> 人声细分（主唱 / 和声）</label>
            <label class="pipe-check"><input type="checkbox" id="pipeGuitar"> 吉他细分</label>
            <label class="pipe-field">吉他方案<select id="pipeGuitarMode"><option value="timbre">木吉他与电吉他</option><option value="duet">旋律与伴奏</option></select></label>
            <label class="pipe-check"><input type="checkbox" id="pipeDrums"> 鼓组细分（六件）</label>
            <label class="pipe-check"><input type="checkbox" id="pipeHarmonyDefault" checked> 融合结果设为默认时间轴</label>
          </div>
          <div class="pipeline-foot"><button id="pipeStart" class="primary">开始完整流程 <span>→</span></button>
            <span class="muted">约 3–6 分钟；分轨和细分结果按音频缓存，重复运行会复用。</span></div>
        </div>'''
assert text.count(ANCHOR) == 1
text = text.replace(ANCHOR, PANEL)

OLD_SCRIPT = '<script defer src="/assets/app.js"></script>'
NEW_SCRIPT = '<script defer src="/assets/app.js"></script>\n  <script defer src="/assets/pipeline.js"></script>'
if NEW_SCRIPT not in text:
    assert text.count(OLD_SCRIPT) == 1
    text = text.replace(OLD_SCRIPT, NEW_SCRIPT)
html.write_text(text, encoding="utf-8")
print("index.html: pipeline panel")

app = FRONTEND / "app.js"
text = app.read_text(encoding="utf-8")
OLD = '$("separate").addEventListener("click",()=>beginSeparation().catch(error=>{toast(error.message);$("inputStatus").textContent=error.message;}));'
NEW = ('$("separate").addEventListener("click",()=>{'
       'if(window.openPipelinePanel){window.openPipelinePanel();return;}'
       'beginSeparation().catch(error=>{toast(error.message);$("inputStatus").textContent=error.message;});'
       '});')
assert text.count(OLD) == 1, text.count(OLD)
app.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("app.js: separate button opens the panel")

css = FRONTEND / "style.css"
text = css.read_text(encoding="utf-8")
if ".pipeline-panel" not in text:
    text += """
.pipeline-panel { margin-top: 14px; border: 1px solid #304043; border-radius: 12px; padding: 14px 16px; background: #16201f; }
.pipeline-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }
.pipeline-head strong { font-size: 14px; }
.pipeline-head .muted { font-size: 12px; }
.pipeline-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px 18px; }
.pipe-field { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); }
.pipe-field select { background: #16201f; color: var(--text); border: 1px solid #3d4f53; border-radius: 7px; padding: 6px 9px; font-size: 12px; }
.pipe-check { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text); }
.pipe-check input { accent-color: #b6e7ce; }
.pipeline-foot { display: flex; align-items: center; gap: 14px; margin-top: 12px; flex-wrap: wrap; }
.pipeline-foot .muted { font-size: 11px; }
"""
    css.write_text(text, encoding="utf-8")
    print("style.css: pipeline styles")
