#!/usr/bin/env python
"""Build the local review library from completed demo reports, without model calls."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from string import Template
from urllib.parse import quote

WORKSPACE = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("studio_assets")
DEFAULT_REPORT = WORKSPACE / "benchmarks/reports/session_20260926/full_song_demos/report.json"
SONGS = {
    "qimeidi": ("凄美地", "郭顶"),
    "gurenge": ("紅蓮華", "LiSA"),
    "pingfan": ("平凡之路", "朴树"),
    "yorushika": ("だから僕は音楽を辞めた", "ヨルシカ"),
}
PALETTE = ["#aac7b8", "#ccbbce", "#bdd0d8", "#d6c7b2", "#b8c5a8", "#cbbdd5"]


def local_url(path: Path, parent: Path) -> str:
    """Encode local paths including spaces, Unicode, percent and hash characters."""
    try:
        value = Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix()
        return quote(value, safe="/")
    except ValueError:
        return path.resolve().as_uri()


def report_path(value: str | None, folder: Path) -> Path | None:
    if not value:
        return None
    path = Path(value.replace("\\", "/"))
    choices = [path] if path.is_absolute() else [WORKSPACE / path, folder / path, folder / path.name]
    return next((item.resolve() for item in choices if item.is_file()), None)


def duration_label(value: float) -> str:
    seconds = max(0, round(float(value)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def disagreement_list(song: dict | None, primary: Path | None, folder: Path, models: tuple[str, str]) -> str:
    if not song or not primary:
        return ""
    names = {"root":"根音", "quality":"性质 / 扩展音", "triad":"和弦性质", "extension":"扩展音", "bass":"低音", "coverage":"覆盖", "label":"标签"}
    links = []
    for segment in song.get("segments", []):
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= float(song.get("duration", 0))+.001:
            continue
        url = local_url(primary, folder) + f"#t={start:.6f}&end={end:.6f}"
        left = html.escape(str(segment["primary_label"])) if segment.get("primary_label") is not None else "未覆盖"
        right = html.escape(str(segment["comparison_label"])) if segment.get("comparison_label") is not None else "未覆盖"
        kind = html.escape(names.get(segment.get("kind"), "标签"))
        tenth_start, tenth_end = round(start*10), round(end*10)
        time = f"{tenth_start//600:02d}:{tenth_start%600/10:04.1f}–{tenth_end//600:02d}:{tenth_end%600/10:04.1f}"
        links.append(f'<li><a href="{html.escape(url,quote=True)}"><span class="difference-time">{time}<small>{kind}分歧</small></span><span class="difference-labels"><strong>{left}</strong><span>↔</span><strong>{right}</strong></span></a></li>')
    if not links:
        return ""
    first, second = (html.escape(value) for value in models)
    minimum = float(song.get("display_min_duration", .5))
    more = f'<details class="more-differences"><summary>展开其余 {len(links)-8} 段</summary><ul>{"".join(links[8:])}</ul></details>' if len(links)>8 else ""
    return f'<details class="disagreements"><summary>模型分歧试听 · {len(links)} 段</summary><p>左：{first}<br>右：{second}<br>仅列出至少 {minimum:g} 秒的不同标签，不判断哪方正确。</p><ul>{"".join(links[:8])}</ul>{more}<p>点击定位后按空格播放，按 L 开启区间循环；标注不会自动改变。未覆盖表示缺少预测，不等于无和弦 N。</p></details>'


def button(path: Path | None, folder: Path, text: str, kind: str) -> str:
    if path is None:
        return f'<span class="button disabled">{html.escape(text)} · 尚未生成</span>'
    return f'<a class="button {kind}" href="{html.escape(local_url(path, folder))}">{html.escape(text)}<span aria-hidden="true">↗</span></a>'


def recommended_primary_report(session: Path, expected_count: int) -> Path:
    """Use the final collection once every expected song has a usable output."""
    final = session / "madmom_user4_final/report.json"
    try:
        data = json.loads(final.read_text(encoding="utf-8-sig"))
        songs = data.get("songs", [])
        if len(songs) == expected_count and all(row.get("status") == "complete" and report_path(row.get("html"), final.parent) for row in songs):
            return final
    except (OSError, ValueError, TypeError):
        pass
    return session / "madmom_user4_workbenches/report.json"


def build_index(report: Path, reference: Path | None = None, primary_report: Path | None = None) -> Path:
    report = report.resolve()
    payload = json.loads(report.read_text(encoding="utf-8-sig"))
    rows = payload.get("demos", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("The report must contain completed demos")
    folder, session = report.parent, report.parent.parent
    primary_report = primary_report or recommended_primary_report(session, len(rows))
    primary_data = json.loads(primary_report.read_text(encoding="utf-8-sig")) if primary_report.is_file() else {}
    primary_rows = {row["id"]: row for row in primary_data.get("songs", []) if row.get("status") == "complete" and report_path(row.get("html"), primary_report.parent)}
    disagreement_path = session / "model_disagreements.json"
    disagreement_data = json.loads(disagreement_path.read_text(encoding="utf-8-sig")) if disagreement_path.is_file() else {}
    disagreements = {row["id"]: row for row in disagreement_data.get("songs", [])}
    disagreement_models = (str(disagreement_data.get("primary_model", "当前默认流程")),str(disagreement_data.get("comparison_model", "SheetSage2")))
    cards = []
    for index, row in enumerate(rows):
        native = report_path(row.get("native_html"), folder)
        review = report_path(row.get("review_html"), folder)
        analysis = report_path(row.get("native_json"), folder)
        data = json.loads(analysis.read_text(encoding="utf-8-sig")) if analysis else {}
        primary_row = primary_rows.get(row.get("id"))
        primary_html = report_path(primary_row.get("html"), primary_report.parent) if primary_row else None
        primary_json = report_path(primary_row.get("prediction"), primary_report.parent) if primary_row else None
        shown_data = json.loads(primary_json.read_text(encoding="utf-8-sig")) if primary_json else data
        shown_row = primary_row or row
        fallback = str(data.get("display_title") or row.get("id") or "未命名歌曲").split(" · ")[0]
        title, artist = SONGS.get(row.get("id"), (fallback, "本地音频"))
        sections = shown_data.get("structure", {}).get("sections", [])
        strip = "".join(
            f'<span style="flex:{max(.01, float(section["end"])-float(section["start"])):.4f};background:{PALETTE[position % len(PALETTE)]}" title="{html.escape(str(section.get("name") or "段落建议"), quote=True)}"></span>'
            for position, section in enumerate(sections)
            if section.get("start") is not None and section.get("end") is not None
        )
        download_path = primary_json or analysis
        download = f'<a class="download" href="{html.escape(local_url(download_path, folder))}" download>下载对应标注 JSON ↓</a>' if download_path else ""
        if primary_html:
            actions = f'{button(primary_html, folder, "打开工作台", "primary")}<details class="comparison"><summary>其他模型对照 · 2 种</summary><div class="song-actions">{button(native, folder, "SheetSage2 原生", "secondary")}{button(review, folder, "专家拍格 · 待审", "secondary")}</div></details>'
        else:
            actions = button(native, folder, "原生工作台", "primary") + button(review, folder, "专家拍格 · 待审", "secondary")
        actions += disagreement_list(disagreements.get(row.get("id")),primary_html,folder,disagreement_models)
        route = "当前默认分析" if primary_html else "SheetSage2 音频候选"
        cards.append(f'''<article class="song"><div class="song-top"><span class="song-number">{index + 1:02d}</span><span>{duration_label(shown_row.get("duration", row.get("duration", 0)))}</span></div><div class="structure-strip" aria-label="段落建议分布">{strip}</div><p class="artist">{html.escape(artist)}</p><h3>{html.escape(title)}</h3><p class="route-label">{route}</p><p class="counts">{int(shown_row.get("event_count", 0))} 个和弦片段 · {int(shown_row.get("role_section_count", 0))} 段结构建议</p><div class="song-actions">{actions}</div>{download}</article>''')
    resource_rows = [
        ("使用说明", "运行、导出与标注入口", WORKSPACE / "README_chord_recognizer.md"),
        ("工作台变更与验证", "交互、音画同步与人工校正", WORKSPACE / "docs/studio_ui_20260926.md"),
        ("本次项目审查", "模型能力、评测与已知限制", WORKSPACE / "PROJECT_REPORT.md"),
        ("四首整曲分析记录", "查看各版本与分析来源", report),
        ("人工标注曲谱 PDF", "《凄美地》独立参考示例", session / "qimeidi_score.pdf"),
    ]
    if primary_rows:
        resource_rows.insert(0, ("默认流程分析记录", "四首歌的版本与输出来源", primary_report))
    if reference and reference.is_file():
        resource_rows.insert(0, ("当前 madmom 路线", "与新模型候选对照试听", reference.resolve()))
    resources = "".join(f'<a class="resource" href="{html.escape(local_url(path, folder))}"><span><strong>{html.escape(title)}</strong><small>{html.escape(detail)}</small></span><span class="arrow" aria-hidden="true">↗</span></a>' for title, detail, path in resource_rows if path.is_file())
    preview_path = session / ("studio_desktop.png" if primary_rows else "studio_expert_grid_review.png")
    if not preview_path.is_file():
        preview_path = session / "studio_desktop.png"
    preview = f'<a class="preview" href="{html.escape(local_url(preview_path, folder))}" aria-label="查看完整工作台截图"><span class="preview-top"><span class="window-dots" aria-hidden="true"><i></i><i></i><i></i></span><span>播放 · 结构 · 人工标注</span><span aria-hidden="true">↗</span></span><img src="{html.escape(local_url(preview_path, folder))}" alt="工作台实际画面：歌曲结构、小节和弦与同步标注面板" width="1440" height="800"></a>' if preview_path.is_file() else '<div></div>'
    if primary_rows:
        guide_rows = [("默认工作台", "点「打开工作台」，播放并点选和弦或段落；修改后用「导出标注」保存。主歌、副歌等角色为专家模型建议，需结合听感确认。"), ("SheetSage2 对照", "独立模型直接分析音频，可与默认流程对照和弦及局部拍号。两个模型的差异为人工试听提供线索。"), ("专家拍格 · 待审", "重新排布 SheetSage2 的显示拍格，保留其和弦时间轴。先核对强拍再确认；修改可撤销，请导出 JSON 保存。")]
    else:
        guide_rows = [("原生工作台", "保留 SheetSage2 给出的拍号与拍点，查看原始分析。缺失拍点会明确标示；和弦及段落功能均需结合试听判断。"), ("专家拍格 · 待审", "使用另一模型的小节建议辅助阅读，和弦起止时间保持相同。试听强拍位置后再确认；修改可撤销，请导出 JSON 保存。")]
    guide = "".join(f'<div><h3><span>{index+1:02d}</span>{html.escape(title)}</h3><p>{html.escape(detail)}</p></div>' for index,(title,detail) in enumerate(guide_rows))
    session_date = session.name.removeprefix("session_")
    if len(session_date) == 8 and session_date.isdigit():
        session_date = f"{session_date[:4]}.{session_date[4:6]}.{session_date[6:]}"
    page = Template((ASSETS / "review_index.html").read_text(encoding="utf-8")).substitute(
        css=(ASSETS / "review_index.css").read_text(encoding="utf-8"),
        song_count=len(rows), total_duration=duration_label(sum(float(row.get("duration", 0)) for row in rows)),
        song_cards="".join(cards), resources=resources, preview=preview,
        date=html.escape(session_date), guide=guide, guide_class="three" if primary_rows else "",
        view_heading="多模型对照" if primary_rows else "两种视图", view_detail="试听与人工复核" if primary_rows else "相同的和弦时间轴",
        library_note="默认流程可直接试听<br>独立模型与专家拍格可展开对照" if primary_rows else "SheetSage2 音频候选<br>段落与拍格由独立模型提供建议",
    )
    output = folder / "index.html"
    output.write_text(page, encoding="utf-8")
    target = html.escape(local_url(output, session), quote=True)
    (session / "index.html").write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={target}"><title>Chord Studio</title><p><a href="{target}">打开整曲和声工作区</a></p></html>', encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reference", type=Path, help="Completed madmom/reference HTML to include")
    parser.add_argument("--primary-report", type=Path, help="Completed madmom workbench report; defaults to the session's madmom_user4_workbenches/report.json")
    args = parser.parse_args()
    if args.reference and not args.reference.is_file():
        parser.error(f"Reference page not found: {args.reference}")
    print(build_index(args.report, args.reference, args.primary_report))


if __name__ == "__main__":
    main()
