"""Refuse to queue a separation while another (analysis) job owns the serial queue."""
from __future__ import annotations

from pathlib import Path

FRONTEND = Path(r"C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend")

app = FRONTEND / "app.js"
text = app.read_text(encoding="utf-8")
old = 'const job=await post(`/api/projects/${id}/separate`,{force:false});'
new = ('const job=await post(`/api/projects/${id}/separate`,{force:false});'
       'if(job.profile&&job.profile!=="stems")throw new Error("当前有分析任务在跑，等它结束再点「音轨分离」");')
assert text.count(old) == 1, text.count(old)
app.write_text(text.replace(old, new), encoding="utf-8")

stems = FRONTEND / "stems.js"
text = stems.read_text(encoding="utf-8")
old2 = """    const job = await api(`/api/projects/${PROJECT}/separate`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: !!force }),
    });
    await followJob(job.id);"""
new2 = """    const job = await api(`/api/projects/${PROJECT}/separate`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: !!force }),
    });
    if (job.profile && job.profile !== "stems") {
      throw new Error("当前有分析任务在跑，等它结束再点「开始分离」");
    }
    await followJob(job.id);"""
assert text.count(old2) == 1, text.count(old2)
stems.write_text(text.replace(old2, new2), encoding="utf-8")
print("guards added")
